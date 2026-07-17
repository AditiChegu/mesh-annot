import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_geometric.nn import SplineConv

class SplineGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SplineConv(in_channels, 16, dim=3, kernel_size=5)
        self.conv2 = SplineConv(16, 32, dim=3, kernel_size=5)
        self.conv3 = SplineConv(32, 16, dim=3, kernel_size=5)
        self.conv4 = SplineConv(16, out_channels, dim=3, kernel_size=5)
    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, pseudo)
        x = F.relu(x)
        x = self.conv2(x, edge_index, pseudo)
        x = F.relu(x)
        x = self.conv3(x, edge_index, pseudo)
        x = F.relu(x)
        x = self.conv4(x, edge_index, pseudo)
        x = F.relu(x)

        return x

# class ExampleGCN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, 16)
#         self.conv2 = GCNConv(16, out_channels)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)

#         return x