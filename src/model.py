import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductRankingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64):
        super(ProductRankingModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP to combine embeddings and features
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 3, hidden_dim),  # 2 embeddings + 3 numerical features
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: predicted score
        )

    def forward(self, user, item, features):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        x = torch.cat([user_emb, item_emb, features], dim=1)
        out = self.mlp(x)
        return out.squeeze(1)  # [batch_size]
