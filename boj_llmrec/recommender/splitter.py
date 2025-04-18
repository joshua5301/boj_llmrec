import numpy as np
import pandas as pd

class Splitter:
    
    def leave_n_out_split(self, interactions: pd.DataFrame, n: int = 20, is_random: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        grouped = interactions.groupby('user_id')
        train_users, train_items = [], []
        test_users, test_items = [], []

        for user, group in grouped:
            item_indices = group['item_id'].values
            if len(item_indices) <= n:
                continue
            if is_random:
                test_indices = np.random.choice(item_indices, replace=False, size=n)
            else:
                test_indices = item_indices[-n:]
            train_indices = list(set(item_indices) - set(test_indices))

            train_users.extend([user] * len(train_indices))
            test_users.extend([user] * len(test_indices))
            train_items.extend(train_indices)
            test_items.extend(test_indices)

        train_interactions = pd.DataFrame({'user_id': train_users, 'item_id': train_items})
        test_interactions = pd.DataFrame({'user_id': test_users, 'item_id': test_items})
        return train_interactions, test_interactions
