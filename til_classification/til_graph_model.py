# til_graph_model.py
# Graph-based TIL classification using patch-wise spatial relationships.
# Each patch is treated as a node and edges encode spatial proximity or histological correlation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # Graph Attention Network
from torch_geometric.data import Data

class TILGraphClassifier(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, out_channels=4):
        """
        Initializes a graph neural network for TIL subtype classification.

        Args:
            in_channels (int): Size of input patch feature vectors.
            hidden_channels (int): Hidden embedding size in GNN.
            out_channels (int): Number of TIL subtype classes.
        """
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.gat2 = GATConv(hidden_channels * 2, out_channels, heads=1, concat=False)

    def forward(self, data):
        """
        Forward pass of GNN.

        Args:
            data (torch_geometric.data.Data): Contains:
                - x: node features (N, F)
                - edge_index: graph connectivity (2, E)

        Returns:
            Tensor: Node-level predictions (N, out_channels)
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))  # First GAT layer
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gat2(x, edge_index)          # Second GAT layer
        return F.log_softmax(x, dim=1)
