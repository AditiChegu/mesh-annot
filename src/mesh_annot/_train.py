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

# TODO Utilities
    # Need to calculate Dice and BCE loss.
    # Need to find a good combination to use such that we can optimize on it,
    # we can look to the visual-autolabel calculations for this. 
    # We also need to figure out how the loss is one of the outputs of the training loop.
    # Worth checking cortexae for this part.

# TODO Epoch Functions
    # TODO trn_epoch()
    # TODO val_epoch()
# TODO Training Runs

# TODO Model Build Function
@cfg.wrap_opts
def model(
    *,
    properties=Ellipsis,
    device=Ellipsis
):
    from ._model import SplineGCN
    model = SplineGCN(len(properties), 1)
    return model.to(device)
    
# TODO Main Training Loop Function
    # NB Make sure that when we use the Dataloaders() function for the graph data,
    # we use the torch_geometric version.