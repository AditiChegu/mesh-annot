import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_geometric.nn import SplineConv
from torch_geometric.nn import TopKPooling

from . import config as cfg

class SplineGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SplineConv(in_channels, 16, dim=3, kernel_size=5)
        self.conv2 = SplineConv(16, 32, dim=3, kernel_size=5)
        self.conv3 = SplineConv(32, 16, dim=3, kernel_size=5)
        self.conv4 = SplineConv(16, out_channels, dim=3, kernel_size=5)
        # TODO Added dropout, this is different from the usual torch.nn.Dropout
        # but it is how Ribeiro did it, so I'm just rolling with that for now.
        self.dropout = 0.2
        
    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, pseudo)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index, pseudo)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index, pseudo)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv4(x, edge_index, pseudo)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        return x

class SplineUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder
        self.conv1 = SplineConv(in_channels, 16, dim=3, kernel_size=5)
        self.pool1 = TopKPooling(16, ratio=0.5)
        
        self.conv2 = SplineConv(16, 32, dim=3, kernel_size=5)
        self.pool2 = TopKPooling(32, ratio=0.5)
        
        # Bottleneck
        self.bottleneck = SplineConv(32, 32, dim=3, kernel_size=5)
        
        # Decoder (input channels accounting for skip connections) 
        self.conv3 = SplineConv(32+32, 16, dim=3, kernel_size=5)
        self.conv4 = SplineConv(16+16, out_channels, dim=3, kernel_size=5)

    def forward(self, data):
        # Encoder
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x0 = self.conv1(x, edge_index, pseudo)
        x0 = F.relu(x0)
        skip0 = x0
        
        x0_pooled, edge_index0, pseudo0, batch0, perm0, score0 = self.pool1(x0, edge_index, edge_attr=pseudo, batch=batch)
        
        # Encoder
        x1 = self.conv2(x0_pooled, edge_index0, pseudo0)
        x1 = F.relu(x1)
        skip1 = x1
        
        x1_pooled, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool2(x1, edge_index0, edge_attr=pseudo0, batch=batch0)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x1_pooled, edge_index1, edge_attr1)
        x_bottleneck = F.relu(x_bottleneck)
        
        # Decoder
        x_up1 = torch.zeros_like(skip1)
        x_up1[perm1] = x_bottleneck
        x_up1 = torch.cat([x_up1, skip1], dim=-1) # Some places also just add this, like perm1 + skip1
        x2 = self.conv3(x_up1, edge_index0, pseudo0)
        x2 = F.relu(x2)
        
        # Decoder
        x_up0 = torch.zeros_like(skip0)
        x_up0[perm0] = x2
        x_up0 = torch.cat([x_up0, skip0], dim=-1)
        
        x = self.conv4(x_up0, edge_index, pseudo)
        # TODO Consider adding a relu here.
        return x