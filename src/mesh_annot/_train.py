import os
from pathlib import Path
from collections import namedtuple

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from ._model import SplineGCN
from . import config as cfg

# TODO Utilities
    # Maybe we should consider using BCE Loss for this instead of
    # MSE given that we will eventually be working on node-level classification
    # instead of graph-level i.e., property prediction

@cfg.wrap_opts
def add_noise(x, noise_std=Ellipsis):
    if noise_std is None:
        return x
    else:
        return x + noise_std * torch.randn_like(x)

# EPOCH FUNCTIONS
@cfg.wrap_opts
def trn_epoch(
    epochno,
    /,
    model,
    loader,
    optimizer,
    *,
    noise_std=Ellipsis,
    device=Ellipsis
):
    model.train()
    loss_total = 0.0
    # TODO Check if there is a better place to intialize these values
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(add_noise(batch, noise_std=noise_std))
        loss = criterion(output, batch.y)
        loss_total += loss.detach().item()
        
    batch_size = len(loader)
    return {
        'total loss' : loss_total / n
    }

# TRAINING RUNS
RunBase = namedtuple(
    'RunBase',
    ('hyperparams', 'model', 'history', 'best_loss')
)

class Run(RunBase):
    __slots__ = ()
    # TODO What doescls mean here?
    def __new__(
        cls,
        hyperparams,
        model,
        history,
        *,
        loss_colname = 'val_total'
    ):
        history = pd.DataFrame(history)
        best_loss = float(history[loss_colname].min())
        return RunBase.__new__(cls, hyperparams, model, history, best_loss)

    def save(
        self,
        dirpath,
        /,
        *,
        mkdir=True,
        model_filename='weights.pt',
        hyperparams_filename='hyperparams.json',
        history_filename='history.tsv'
    ):
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            if mkdir:
                dirpath.mkdir(exist_ok=True, parents=True)
            else:
                raise ValueError(f'{dirpath} does not exist and mkdir is False')
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(dirpath, model_filename))
        # Save the hyperparameters
        with os.path.join(dirpath, hyperparams_filename).open('w') as fl:
            json.dump(self.hyperparams, fl)
        # Save the history
        self.history.to_csv(os.path.join(dirpath, history_filename), sep='\t', index=False)

    @classmethod
    def load(
        cls,
        dirpath,
        /,
        *,
        no_model=False,
        model_filename='weights.pt',
        hyperparams_filename='hyperparams.json',
        history_filename='history.tsv',
        loss_colname='val_total',
        device=Ellipsis
    ):
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            raise ValueError(f'Given directory {dirpath} does not exist')
        if device is Ellipsis:
            from .config import device as device

        # Load the hyperparameters
        with os.path.json(dirpath, hyperparams_filename).open('r') as fl:
            hp = json.load(fl)
        hp = cfg.hyperparams(**hp)

        if no_model:
            mdl = None
        else:
            mdl = model(
                properties=hp['properties'],
                latent_dim=hp['latent_dim'],
                dropout=hp['dropout'],
                device=device
            )
            weights = torch.load(os.path.join(dirpath, model_filename), weights_only=True)
            mdl.load_state_dict(weights)

        history = pd.read_csv(os.path.join(dirpath, history_filename), sep='\t')
        return cls(hp, mdl, history, loss_colname=loss_colname)

    def loadrun(
        dirpath,
        /,
        *,
        no_model=False,
        model_filename='weights.pt',
        hyperparams_filename='hyperparams.json',
        history_filename='history.tsv',
        loss_colname='val_total',
        device=Ellipsis
    ):
        return Run.load(
            dirpath,
            no_model=no_model,
            model_filename=model_filename,
            hyperparams_filename=hyperparams_filename,
            history_filename=history_filename,
            loss_colname=loss_colname,
            device=device
        )

# MODEL BUILD FUNCTION
@cfg.wrap_opts
def model(
    *,
    properties=Ellipsis,
    target=Ellipsis,
    dropout=Ellipsis,
    device=Ellipsis
):
    from ._model import SplineGCN
    mdl = SplineGCN(
        in_channels= 3 + len(properties),
        out_channels= len(target),
        dropout=dropout
    )
    return mdl.to(device)

# TRAINING FUNCTION
@cfg.wrap_opts
def train(
    trn_dset,
    val_dset,
    *,
    properties=Ellipsis,
    epochs=Ellipsis,
    batch_size=Ellipsis,
    kernel_size=Ellipsis,
    dropout=Ellipsis,
    device=Ellipsis,
    lr=Ellipsis,
    noise_std=Ellipsis,
    target=Ellipsis,
    hemisphere=Ellipsis,
    # wrap_opts() function puts all the hyperparameters in this option.
    hyperparams=Ellipsis,
    # Controls side-effects, but also are not hyperparameters.
    log=print,
    loss_only=False,
    savedir=None,
    mkdir=True,
    # If you want to continue a previous run (all hyperparams are ignored
    # in this case and the provided run's hyperparams are used instead)
    resume=None,
    # Start with an existing model
    initmodel=None,
    # These are for saving the run if savedir is not None
    model_filename='weights.pt',
    hyperparams_filename='hyperparams.json',
    history_filename='history.tsv'
):
    from . import _train as self
    # TODO Finish this function
    