import torch
import torch.nn as nn
import pandas as pd

# Load sample data
ratings = pd.read_csv('../data/ratings.csv')

# Neural CF model
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(nn.Linear(embedding_dim*2,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, user, item):
        x = torch.cat([self.user_emb(user), self.item_emb(item)], dim=1)
        return self.fc(x)
