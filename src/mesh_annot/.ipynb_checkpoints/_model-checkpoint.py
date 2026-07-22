import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_geometric.nn import SplineConv

# TODO Need to build the actual model. The idea is to do a two (secretly three) stream network.
# Let us start with talking about stream 2 because it is (relatively) straightfoward, 
# and is more of a problem of loading in data than using it.
# We need to load in the HCP data in a way that allows us to process it like 3D data.
# For some instruction on how to implement a 3D-CNN, look at this link
# https://www.geeksforgeeks.org/deep-learning/video-classification-with-a-3d-convolutional-neural-network/
# For stream 1 on the other hand, we need to resample the output of every layer onto the next lower
# resolution of available fsaverage maps (as a reminder, the options in decreasing order of resolution are
# fsaverage, fsaverage6, fsaverage5, fsaverage4, fsaverage3).
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