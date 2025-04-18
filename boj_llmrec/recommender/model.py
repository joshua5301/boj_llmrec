import torch
from .dataset import Dataset

class MultiVAE(torch.nn.Module):
    def __init__(self, dataset: Dataset) -> None:
        super(MultiVAE, self).__init__()
        self.dataset = dataset
        self.user_item_matrix = torch.tensor(self.dataset.user_item_matrix.todense(), dtype=torch.float32)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.encoder_dim = [self.dataset.item_cnt] + [2000, 300]
        self.decoder_dim = self.encoder_dim[::-1]
        self.decoder_dim[0] = self.decoder_dim[0] // 2

        self.encoder_layers = torch.nn.ModuleList(
            torch.nn.Linear(self.encoder_dim[i], self.encoder_dim[i+1]) for i in range(len(self.encoder_dim) - 1)
        )
        self.decoder_layers = torch.nn.ModuleList(
            torch.nn.Linear(self.decoder_dim[i], self.decoder_dim[i+1]) for i in range(len(self.decoder_dim) - 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.nn.functional.normalize(input)
        input = self.dropout(input)

        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)
        recon_input = self.decode(z)
        return recon_input, mu, log_var

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)
        x = self.encoder_layers[-1](x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers[:-1]:
            z = layer(z)
            z = torch.tanh(z)
        z = self.decoder_layers[-1](z)
        return z

    def get_topk(self, k: int) -> torch.Tensor:
        scores, _, _ = self.forward(self.user_item_matrix)
        trues = self.dataset.train_interactions.groupby('user_id')['item_id'].apply(list)
        for user_id, items in trues.items():
            scores[user_id, items] = -1e9
        topk = torch.topk(scores, k=k, dim=1).indices
        return topk

    def to(self, device: str):
        super(MultiVAE, self).to(device)
        self.user_item_matrix = self.user_item_matrix.to(device)
        return self