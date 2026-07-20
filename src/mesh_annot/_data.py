from pathlib import Path
import torch
from . import config as cfg

class GraphDataset(torch.utils.data.Dataset):
    @staticmethod
    def _prop_list(base_path):
        return [
            p.name
            for p in base_path.iter_dir()
            if p.is_dir() and not p.name.startswith('.')
        ]
    # TODO Add the rest of the helper methods
    def __init__(
        self,
        base_path,
        sids,
        *,
        properties=Ellipsis,
        graph_size=Ellipsis,
        device=Ellipsis,
        raters=None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        self.device = device

        base_path = Path(base_path)
        self.base_path = base_path

        self.sids = sids

        self.raters = raters
        if raters is None:
            raters = (None,)

        if isinstance(properties, str):
            properties = (properties,)

        # Allocate the dataset space
        nprops = len(properties)
        nsids = len(sids)
        nraters = len(raters)

        # TODO Figure out how to load in multiple subjects and interpolate their native graphs onto
        # the fsaverage brain mesh and make it a dataset
        
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, k):
        return self.data[k].to(self.device)