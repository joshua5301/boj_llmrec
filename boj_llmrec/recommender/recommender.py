import os
import numpy as np
import pandas as pd
import torch
import requests
import json

from .dataset import Dataset
from .encoder import Encoder
from .splitter import Splitter
from .downloader import DataDownloader
from .MultiVAE import MultiVAE
from .MultiVAE import MultiVAETrainer
from .LightGCN import LightGCN
from .LightGCN import LightGCNTrainer

class Recommender:

    def __init__(self, data_path: str) -> None:
        self.solved_info = pd.read_csv(os.path.join(data_path, 'solved_info.csv'), index_col=0)
        self.solved_info.columns = ['user_id', 'item_id']
        self.problem_info = pd.read_csv(os.path.join(data_path, 'problem_info.csv'))

        self.top_100_info = {}
        top_100_path = os.path.join(data_path, 'top_100_for_demo')
        for filename in os.listdir(top_100_path):
            if filename.startswith("top_100_") and filename.endswith(".json"):
                username = filename[len("top_100_"):-len(".json")]

                file_path = os.path.join(top_100_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)['items']
                    self.top_100_info[username] = data
        
        self._init_recommender()

    def _init_recommender(self) -> None:
        self.encoder = Encoder()
        train_df = self.encoder.fit_transform(self.solved_info)
        train_df['user_id'] = train_df['user_id'].astype(int)
        train_df['item_id'] = train_df['item_id'].astype(int)
        self.dataset = Dataset(train_df, None, None, None)
        self.lightgcn_model = LightGCN(self.dataset)
        self.multivae_model = MultiVAE(self.dataset)

    def train_model(self, model_type: str) -> None:
        if model_type == 'LightGCN':
            trainer = LightGCNTrainer(self.dataset, self.lightgcn_model)
            trainer.train()
        elif model_type == 'MultiVAE':
            trainer = MultiVAETrainer(self.dataset, self.multivae_model)
            trainer.train()

    def save_model(self, model_path: str, model_type: str) -> None:
        if model_type == 'LightGCN':
            model = self.lightgcn_model
        elif model_type == 'MultiVAE':
            model = self.multivae_model
        torch.save(model.state_dict(), model_path)

    def load_model(self, model_path: str, model_type: str) -> None:
        if model_type == 'LightGCN':
            self.lightgcn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
            self.lightgcn_model.eval()
        elif model_type == 'MultiVAE':
            self.multivae_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
            self.multivae_model.eval()

    def get_recommended_problems(self, user_handle: str) -> list:
        downloader = DataDownloader()
        try:
            problems = downloader.get_top_100_problems(user_handle)
        except requests.exceptions.HTTPError as e:
            if user_handle in self.top_100_info:
                problems = self.top_100_info[user_handle]
                print(f"Using cached top 100 problems for {user_handle}.")
            else:
                problems = []
                print(f"Error fetching top 100 problems for {user_handle}: {e}")
            
        solved_ids = []
        for problem in problems:
            solved_ids.append(problem['problemId'])
        if solved_ids:
            solved_ids = self.encoder.item_encoder.transform(np.array(solved_ids).reshape(-1, 1)).ravel()
            solved_ids = [id for id in solved_ids if id >= 0]
        is_solved = np.zeros(self.multivae_model.dataset.item_cnt)
        is_solved[solved_ids] = 1
        is_solved = torch.tensor(is_solved, dtype=torch.float32).unsqueeze(0)
        scores, _, _ = self.multivae_model.forward(is_solved)
        scores = scores.squeeze().to('cpu').detach().numpy()
        mask = np.zeros_like(scores, dtype=bool)
        mask[solved_ids] = True
        scores[mask] = -np.inf
        sorted_problem_ids = np.argsort(scores)[::-1]
        sorted_problem_ids = self.encoder.item_encoder.inverse_transform(sorted_problem_ids.reshape(-1, 1)).ravel()
        sorted_df = self.problem_info.set_index('problemId', drop=True).loc[sorted_problem_ids].reset_index()
        return sorted_df

    def get_similar_problems(self, problem_id: int) -> list:
        problem_id = self.encoder.item_encoder.transform(np.array([problem_id]).reshape(-1, 1)).item()
        if problem_id < 0:
            raise ValueError("Problem ID not found in the dataset.")
        target_embedding = self.lightgcn_model.item_embedding.weight[problem_id].unsqueeze(0)
        all_embeddings = self.lightgcn_model.item_embedding.weight
        similarities = torch.nn.functional.cosine_similarity(target_embedding, all_embeddings)
        similarities = similarities.cpu().detach().numpy()
        sorted_problem_ids = np.argsort(similarities)[::-1]
        sorted_problem_ids = self.encoder.item_encoder.inverse_transform(sorted_problem_ids.reshape(-1, 1)).ravel()
        sorted_df = self.problem_info.set_index('problemId', drop=True).loc[sorted_problem_ids].reset_index()
        return sorted_df
    
    def get_other_user_problems(self, recommended_problems: pd.DataFrame, base_user_handle: str, target_user_handle: str) -> list:
        downloader = DataDownloader()
        try:
            base_user_problems = downloader.get_top_100_problems(base_user_handle)
            target_user_problems = downloader.get_top_100_problems(target_user_handle)
        except requests.exceptions.HTTPError as e:
            if base_user_handle in self.top_100_info:
                base_user_problems = self.top_100_info[base_user_handle]
            else:
                base_user_problems = []
            if target_user_handle in self.top_100_info:
                target_user_problems = self.top_100_info[target_user_handle]
            else:
                target_user_problems = []
        base_problem_ids = {problem['problemId'] for problem in base_user_problems}
        target_problem_ids = {problem['problemId'] for problem in target_user_problems}
        other_problem_ids = list(target_problem_ids - base_problem_ids)
        other_problem_ids = [id for id in other_problem_ids if id >= 0]
        other_user_problems = self.problem_info.set_index('problemId', drop=True).loc[other_problem_ids].reset_index()
        sorted_df = recommended_problems[recommended_problems['problemId'].isin(other_user_problems['problemId'])].copy()
        return sorted_df
