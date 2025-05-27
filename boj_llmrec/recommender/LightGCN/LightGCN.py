import torch
import numpy as np
from ..dataset import Dataset

class LightGCN(torch.nn.Module):
    def __init__(self, dataset: Dataset) -> None:
        super(LightGCN, self).__init__()
        self.dataset = dataset

        self.user_embedding = torch.nn.Embedding(
            self.dataset.user_cnt, 128
        )
        self.item_embedding = torch.nn.Embedding(
            self.dataset.item_cnt, 128
        )
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.aggregator = self.get_aggregator()

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_embedding, item_embedding = self.get_embeddings()
        user_embedding = user_embedding[user_indices]
        item_embedding = item_embedding[item_indices]
        return torch.sum(user_embedding * item_embedding, dim=1)
    
    def get_topk(self, k: int) -> torch.Tensor:
        user_embedding, item_embedding = self.get_embeddings()
        scores = user_embedding @ item_embedding.T
        trues = self.dataset.train_interactions.groupby('user_id')['item_id'].apply(list)
        for user_id, items in trues.items():
            scores[user_id, items] = -1e8
        topk = torch.topk(scores, k=k, dim=1).indices
        return topk

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        full_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings.append(full_embedding)
        for _ in range(1):
            full_embedding = torch.sparse.mm(self.aggregator, full_embedding)
            embeddings.append(full_embedding)
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        final_user_embedding, final_item_embedding = torch.split(
            final_embedding, [self.dataset.user_cnt, self.dataset.item_cnt])
        return final_user_embedding, final_item_embedding
    
    def get_aggregator(self) -> torch.Tensor:
        coo = self.dataset.normalized_matrix.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float)
        aggregator = torch.sparse_coo_tensor(indices, values, size=coo.shape)
        return aggregator
    
    def to(self, device: torch.device):
        super(LightGCN, self).to(device)
        self.aggregator = self.aggregator.to(device)
        self.user_embedding = self.user_embedding.to(device)
        self.item_embedding = self.item_embedding.to(device)
        return self