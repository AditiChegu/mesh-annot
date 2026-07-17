from pathlib import Path
import torch
import config as cfg

class HCPDataset(torch.utils.data.Dataset):
    # TODO Implement the rest of the helper functions.
    # TODO Add a function to help load the graph data in. 
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
        dset_dims = (nraters * ndims, nprops, h, w)
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