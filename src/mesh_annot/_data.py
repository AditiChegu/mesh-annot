import os
from pathlib import Path

import numpy as np
import neuropythy as ny
from . import config as cfg

import torch
import torch_geometric
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

class HCPDataset(torch_geometric.data.Dataset):
    def _graph_info(self, vertices, edges):
        if edges.shape[0] == 2 and edges.shape[1] != 2:
            edges = edges.T
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)

        x = torch.tensor(vertices.T, dtype=torch.long)
        src = x[edge_index[0]]
        trg = x[edge_index[1]]
        edge_attr = src - trg
        edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())

        return x, edge_index, edge_attr

    def _load_prop(self, prop, sid, hemisphere):
        prop_path = os.path.join(self.base_path, prop, f'{sid}.{hemisphere}.mgz')
        prop_mgz = ny.load(prop_path)
        prop_arr = prop_mgz.byteswap().view(prop_mgz.dtype.newbyteorder())
        prop = torch.tensor(prop_arr, dtype=torch.float).unsqueeze(1)
        return prop

    @cfg.wrap_opts
    def __init__(
        self,
        base_path,
        sids,
        *,
        hemisphere=Ellipsis,
        properties=Ellipsis,
        target=Ellipsis,
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
        self.hemisphere = hemisphere

        self.raters = raters
        if raters is None:
            raters = (None,)

        if isinstance(properties, str):
            properties = (properties,)

        # Allocate the dataset space
        nprops = len(properties)
        nsids = len(sids)
        nraters = len(raters)
        dset_dims = (nraters * nsids, nprops, 163842)

        self.graph_data = {}

        for rater in raters:
            for sid in sids:
                mesh_path = os.path.join(self.base_path, f'{sid}.{hemisphere}.mesh')
                mesh = ny.load(mesh_path, 'freesurfer_geometry')
                vertices = mesh.coordinates

                edge_path = os.path.join(self.base_path, f'edges/{sid}.{hemisphere}.pt')
                edges = torch.load(edge_path, weights_only=False)

                x_coords, edge_index, edge_attr = self._graph_info(vertices=vertices, edges=edges)

                y = self._load_prop(prop=target, sid=sid, hemisphere=hemisphere)
                y_mask = torch.isnan(y)
                y[y_mask] = 0.0

                # Setting all the NaN prf values to zero.
                prf_x = self._load_prop('prf_x', sid, hemisphere)
                prf_y = self._load_prop('prf_y', sid, hemisphere)
                prf_sigma = self._load_prop('prf_sigma', sid, hemisphere)
                prf_cod = self._load_prop('prf_cod', sid, hemisphere)
                
                mask = torch.isnan(prf_x) | torch.isnan(prf_y) | torch.isnan(prf_sigma) | torch.isnan(prf_cod)
                
                prf_x[mask] = 0.0
                prf_y[mask] = 0.0
                prf_sigma[mask] = 0.0
                prf_cod[mask] = 0.0
                
                # TODO Need to figure out how to load in other properties.
                # This feels very... slow. I am not a fan of the iteration.
                prop_list = []
                for prop_name in properties:
                    if prop_name == 'prf_x':
                        prop_list.append(prf_x)
                    elif prop_name == 'prf_y':
                        prop_list.append(prf_y)
                    elif prop_name == 'prf_sigma':
                        prop_list.append(prf_cod)
                    elif prop_name == 'prf_cod':
                        prop_list.append(prf_sigma)
                    else:
                        prop = self._load_prop(prop_name, sid, hemisphere)
                        prop_list.append(prop)

                all_props = torch.cat(prop_list, dim=1)
                x = torch.cat([x_coords, all_props], dim=1)

                self.graph_data[sid] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def __len__(self):
        # TODO Is this even right?
        return len(self.graph_data)

    def __getitem__(self, k):
        # TODO
        sid = self.sids[k]
        return self.graph_data[sid].to(self.device)