import os
import numpy as np
import pandas as pd
import torch

from .dataset import Dataset
from .encoder import Encoder
from .splitter import Splitter
from .downloader import DataDownloader
from .model import MultiVAE
from .trainer import MultiVAETrainer

class Recommender:

    def __init__(self, data_path: str) -> None:
        self.solved_info = pd.read_csv(os.path.join(data_path, 'solved_info.csv'), index_col=0)
        self.solved_info.columns = ['user_id', 'item_id']
        self.model = None
        self.encoder = None

    def train_model(self) -> None:
        self.encoder = Encoder()
        train_df = self.encoder.fit_transform(self.solved_info)
        train_df['user_id'] = train_df['user_id'].astype(int)
        train_df['item_id'] = train_df['item_id'].astype(int)
        dataset = Dataset(train_df, None, None, None)
        self.model = MultiVAE(dataset)
        trainer = MultiVAETrainer(dataset, self.model)
        trainer.train()
    
    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path: str) -> None:
        self.encoder = Encoder()
        train_df = self.encoder.fit_transform(self.solved_info)
        train_df['user_id'] = train_df['user_id'].astype(int)
        train_df['item_id'] = train_df['item_id'].astype(int)
        dataset = Dataset(train_df, None, None, None)
        self.model = MultiVAE(dataset)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

    def recommend(self, user_handle: str) -> list:
        downloader = DataDownloader()
        problems = downloader.get_top_100_problems(user_handle)
        solved_ids = []
        for problem in problems:
            solved_ids.append(problem['problemId'])
        solved_ids = self.encoder.item_encoder.transform(np.array(solved_ids).reshape(-1, 1)).ravel()
        solved_ids = [id for id in solved_ids if id >= 0]
        is_solved = np.zeros(self.model.dataset.item_cnt)
        is_solved[solved_ids] = 1
        is_solved = torch.tensor(is_solved, dtype=torch.float32).unsqueeze(0)
        scores, _, _ = self.model.forward(is_solved)
        scores = scores.squeeze().to('cpu').detach().numpy()
        mask = np.zeros_like(scores, dtype=bool)
        mask[solved_ids] = True
        scores[mask] = -np.inf
        sorted_problem_ids = np.argsort(scores)[::-1]
        sorted_problem_ids = self.encoder.item_encoder.inverse_transform(sorted_problem_ids.reshape(-1, 1)).ravel()
        return sorted_problem_ids.tolist()