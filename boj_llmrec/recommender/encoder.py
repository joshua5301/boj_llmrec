import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class Encoder:

    def __init__(self) -> None:
        self.user_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        # self.user_encoder = OrdinalEncoder(dtype=int, handle_unknown='error')
        self.item_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        # self.item_encoder = OrdinalEncoder(dtype=int, handle_unknown='error')

    def fit(self, interactions: pd.DataFrame) -> pd.DataFrame:
        self.user_encoder.fit(
            interactions['user_id'].values.reshape(-1, 1)
        )
        self.item_encoder.fit(
            interactions['item_id'].values.reshape(-1, 1)
        )
        return self

    def transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions.loc[:, 'user_id'] = self.user_encoder.transform(
            interactions['user_id'].values.reshape(-1, 1)
        ).ravel()
        interactions.loc[:, 'item_id'] = self.item_encoder.transform(
            interactions['item_id'].values.reshape(-1, 1)
        ).ravel()
        interactions = interactions[
            (interactions['user_id'] != -1) &
            (interactions['item_id'] != -1)
        ]
        return interactions

    def fit_transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        return self.fit(interactions).transform(interactions)

    def inverse_transform(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions.loc[:, 'user_id'] = self.user_encoder.inverse_transform(
            interactions['user_id'].values.reshape(-1, 1)
        ).ravel()
        interactions.loc[:, 'item_id'] = self.item_encoder.inverse_transform(
            interactions['item_id'].values.reshape(-1, 1)
        ).ravel()
        return interactions
