import torch
import torch.nn as nn


class EdgeNetwork(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        )

        self.hidden_dim = hidden_dim

    def forward(self, edge_feat):
        A = self.edge_mlp(edge_feat)
        return A.view(-1, self.hidden_dim, self.hidden_dim)
