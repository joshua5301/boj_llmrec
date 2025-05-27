import torch
from ..dataset import Dataset
from .MultiVAE import MultiVAE
from ..utils import vae_reg_loss, vae_bce_loss, recall

class MultiVAETrainer:
    def __init__(self, dataset: Dataset, model: MultiVAE) -> None:
        self.dataset = dataset
        self.model = model
        if self.dataset.test_interactions is not None:
            grouped = self.dataset.test_interactions.groupby('user_id')['item_id'].apply(list)
            self.all_true = [grouped.get(user_id, []) for user_id in range(self.dataset.user_cnt)]

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        user_info = torch.tensor(self.dataset.user_item_matrix.todense()).to(device)
        dataset = torch.utils.data.TensorDataset(user_info)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=512,
            shuffle=True,
        )
        self.model.to(device)

        for epoch in range(60):
            for (batch_user_info, ) in dataloader:
                optimizer.zero_grad()
                batch_user_info = batch_user_info.to(device).to(torch.float32)

                recon_users, mu, log_var = self.model(batch_user_info)
                reg_loss = vae_reg_loss(mu, log_var)
                bce_loss = vae_bce_loss(batch_user_info, recon_users)
                loss = (bce_loss + 0.05 * reg_loss).mean()
                loss.backward()
                optimizer.step()

            print(f'epoch: {epoch}')
            if epoch % 5 == 0 and self.dataset.test_interactions is not None:
                self.validate()

    def validate(self):
        if self.dataset.test_interactions is None:
            return
        self.model.eval()
        with torch.no_grad():
            pred = self.model.get_topk(10).to('cpu').numpy().tolist()
        self.model.train()
        print(recall(self.all_true, pred))
