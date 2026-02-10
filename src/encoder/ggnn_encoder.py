import torch
import torch.nn as nn
from .edge_network import EdgeNetwork


class GGNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, steps=3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.steps = steps

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_net = EdgeNetwork(edge_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat):
        h = self.node_proj(node_feat)

        src, dst = edge_index
        A = self.edge_net(edge_feat)

        for _ in range(self.steps):

            m = torch.zeros_like(h)

            for i in range(A.size(0)):
                w = src[i]
                v = dst[i]

                m[v] += torch.matmul(A[i], h[w])

            h = self.gru(m, h)

        return h
