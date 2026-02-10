import torch
import torch.nn as nn
from src.encoder.ggnn_encoder import GGNNEncoder
from src.encoder.readout import Readout


class MetaGNNModel(nn.Module):

    def __init__(self,
                 node_dim,
                 edge_dim,
                 hidden_dim,
                 out_dim):

        super().__init__()

        self.encoder = GGNNEncoder(
            node_dim,
            edge_dim,
            hidden_dim
        )

        self.predictor = Readout(
            hidden_dim,
            out_dim
        )

    def forward(self, node_feat, edge_index, edge_feat):

        h = self.encoder(node_feat, edge_index, edge_feat)
        y = self.predictor(h)

        return y
