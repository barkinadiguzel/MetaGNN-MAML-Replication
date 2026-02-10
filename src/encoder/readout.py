import torch
import torch.nn as nn


class Readout(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        g = torch.sum(h, dim=0)
        return self.mlp(g)
