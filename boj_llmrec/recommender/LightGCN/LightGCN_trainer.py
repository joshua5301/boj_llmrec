import torch
from tqdm import tqdm
from ..dataset import Dataset
from .LightGCN import LightGCN
from ..utils import bpr_loss, recall
from ..sampler import NegativeSampler

class LightGCNTrainer:

    def __init__(self, dataset: Dataset, model: LightGCN) -> None:
        self.dataset = dataset
        self.model = model
        if self.dataset.test_interactions is not None:
            grouped = self.dataset.test_interactions.groupby('user_id')['item_id'].apply(list)
            self.all_true = [grouped.get(user_id, []) for user_id in range(self.dataset.user_cnt)]
           
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        sampler = NegativeSampler(self.dataset, 100, 1)
        
        self.model.to(device)
        for epoch in range(1):
            total_loss = 0
            pairwise_samples = sampler.get_samples().to(device)
            dataset = torch.utils.data.TensorDataset(*pairwise_samples.T)
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=512,
                shuffle=True,
            )
            for users, pos_samples, *neg_samples_list in tqdm(dataloader):
                optimizer.zero_grad()
                pos_scores = self.model(users, pos_samples)
                neg_scores_list = []
                for neg_samples in neg_samples_list:
                    neg_scores_list.append(self.model(users, neg_samples))
                loss = bpr_loss(pos_scores, *neg_scores_list)
                loss.backward()
                optimizer.step() 
                total_loss += loss.item()
                
            print(f'avg_loss: {total_loss / len(dataloader)}')
            if (self.dataset.test_interactions is not None and 
                epoch % 1 == 0):
                self.validate()

    def validate(self):
        if self.dataset.test_interactions is None:
            return
        self.model.eval()
        with torch.no_grad():
            pred = self.model.get_topk(10).to('cpu').numpy().tolist()
        self.model.train()
        print(recall(self.all_true, pred))
