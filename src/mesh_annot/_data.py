from pathlib import Path
import torch
from . import config as cfg

class GraphDataset(torch.utils.data.Dataset):
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

class ImageDataset(torch.utils.data.Dataset):
    @staticmethod
    def _prop_list(base_path):
        return [
            p.name
            for p in base_path.iter_dir()
            if p.is_dir() and not p.name.startswith('.')
        ]

    def _resize(self, im):
        from torch.nn.functional import interpolate
        h = self.image_height
        if h == im.shape[0]:
            return im
        w = h * im.shape[-1] // im.shape[-2]
        im = interpolate(
            im[None, None],
            size=(h, w),
            mode='bilinear',
            antialias=True
        )

        return im[0,0]

    def _load_prop(self, name, sid, rater=None):
        prop_dirpath = self.base_path / name
        if rater is None:
            impaths = prop_dirpath.glob(f"*_{sid}.pt")
            impath = next(impaths)
        else:
            impath = prop_dirpath / f"{rater}_{sid}.pt"

        # Load the property and reshape if needed
        prop = torch.load(impath, weights_only=True)
        prop = self._resize(prop)
        
        return prop
        
    
    @cfg.wrap_opts
    def __init__(
        self, 
        base_path, 
        sids, 
        *, 
        properties=Ellipsis, 
        image_size=Ellipsis, 
        device=Ellipsis, 
        raters=None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        self.device = device

        base_path = Path(base_path)
        self.base_path = base_path

        (h,w) = image_size
        self.image_height = h
        self.image_width = w

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
        dset_dims = (nraters * nsids, nprops, h, w)
        self.data = torch.zeros(dset_dims)
        ii = 0
        for rater in raters:
            for sid in sids:
                for (pii, propname) in enumerate(properties):
                    # TODO Implement the rest of this.
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, k):
        return self.data[k].to(self.device)