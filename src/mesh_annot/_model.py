import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_geometric.nn import SplineConv
from torch_geometric.nn import TopKPooling

# TODO Need to add pooling layers.
# We need to resample the output of every layer onto the next lower
# resolution of available fsaverage maps (as a reminder, the options in decreasing order of resolution are
# fsaverage, fsaverage6, fsaverage5, fsaverage4, fsaverage3).

# TODO Need to figure out if it is feasible to add pooling layers because if we pool the graph, then we lose
# information about the edge_attr i.e., the stuff the splineconv needs to work. 

# TODO Consider if a built-in GraphUNet is good enough.
# TODO I think whenever we do figure out if it is possible to combine splineconvs with pooling and move to that 
# two (three) stream architecture, we should look into pool.voxel_grid 

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

class EncoderBlock(nn.Module):
	@cfg.wrap_opts
	def __init__(self, in_channels, *, dropout, kernel_size):
		super().__init__()
		# TODO Implement the rest of this.
	def forward(self, x):
		# TODO Implement the rest of this.
		return x
class DecoderBlock(nn.Module):
	@cfg.wrap_opts
	def __init__(self, in_channels, dropout, kernel_size):
		super().__init__()
		# TODO Implement the rest of this.
	def forward(self, x):
		# TODO Implement the rest of this.
		return x
class SplineUNet(nn.Module):
	@cfg.wrap_opts
	def __init__(self, in_channels, dropout, kernel_size):
		super().__init__()
		# TODO Implement the rest of this.
	def forward(self, x):
		# TODO Implement the rest of this
		return x