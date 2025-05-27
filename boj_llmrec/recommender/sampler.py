import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix

from .dataset import Dataset

class NegativeSampler:
    def __init__(self, dataset: Dataset, sample_num_per_user: int, negative_sample_num: int) -> None:
        self.dataset = dataset
        self.sample_num_per_user = sample_num_per_user
        self.negative_sample_num = negative_sample_num

    def get_samples(self) -> torch.Tensor:
        pairwise_samples = []

        for user in tqdm(range(self.dataset.user_cnt)):
            adj: csr_matrix = self.dataset.user_item_matrix
            all_items = np.arange(self.dataset.item_cnt)
            positive_items = adj.indices[adj.indptr[user]: adj.indptr[user + 1]]
            negative_items = np.setdiff1d(all_items, positive_items)

            for _ in range(self.sample_num_per_user):
                cur_positive_item = np.random.choice(positive_items)
                cur_negative_items = np.random.choice(negative_items, size=self.negative_sample_num)
                pairwise_samples.append([user, cur_positive_item, *cur_negative_items])
        return torch.tensor(pairwise_samples, dtype=torch.long)
