from pathlib import Path
from collections import namedtuple

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import pandas as pd

from ._model import SplineGCN
from . import config as cfg

# TODO The rest of this
# NB Make sure that when we use the Dataloaders() function for the graph data,
# we use the torch_geometric version

# TODO Epoch Functions

# TODO Training Runs

# TODO Model Build Function
@cfg.wrap_opts
def model(
    *,
    properties=Ellipsis,
    outputs=Ellipsis,
    device=Ellipsis
):
    model = SplineGCN(len(properties), outputs)

    return model.to(device)
    
# TODO Main Training Loop Function