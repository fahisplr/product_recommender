import torch
from torch.utils.data import Dataset

class ProductRankingDataset(Dataset):
    def __init__(self, dataframe):
        self.user_indices = dataframe['user_index'].values
        self.item_indices = dataframe['item_index'].values
        self.features = dataframe[['recency_days', 'sequence_score', 'total_user_txns']].values.astype('float32')
        self.labels = dataframe['buy_count'].values.astype('float32')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.user_indices[idx], dtype=torch.long),
            'item': torch.tensor(self.item_indices[idx], dtype=torch.long),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }
