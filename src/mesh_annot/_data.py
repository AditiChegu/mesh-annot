from pathlib import Path
import torch
from . import config as cfg

class HCPDataset(torch.utils.data.Dataset):
    @staticmethod
    def _prop_list(base_path):
        return [
            p.name
            for p in base_path.iter_dir()
            if p.is_dir() and not p.name.startswith('.')
        ]
    def _load_prop(self, name, sid, raters=None):
        prop_dirpath = os.path.join(self.base_path, name)
        if rater is None:
            impaths = prop_dirpath.glob(f"*_{sid}.pt")
            impath = next(impaths)
        else:
            impath = prop_dirpath / f"{rater}_{sid}.pt"
        prop = torch.load(impath, weights_only=True)
        
        return prop
    
    def __init__(
        self,
        base_path,
        sids,
        *,
        properties=Ellipsis,
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
        dset_dims = (nraters * nsids, nprops)

        self.data = torch.zeroes(dset_dims)
        ii = 0
        for rater in raters:
            for sid in sids:
                for (pii, ppropnames) in enumerate(properties):
                    prop = self._load_prop(propname, sid, rater)
                    self.data[ii, pii, ...] = prop
                ii += 1
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, k):
        return self.data[k].to(self.device)