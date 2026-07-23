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
    def create_graph(self, vertices, edges):
        if edges.shape[0] == 2 and edges.shape[1] != 2:
            edges = edges.T

        edge_index = torch.tensor(edges.T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        x_coords = torch.tensor(vertices.T, dtype=torch.long)

        src = x_coords[edge_index[0]]
        trg = x_coords[edge_index[1]]
        edge_attr = src - trg
        edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())

        graph_data = Data(x=x_coords, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph_data
        
    @staticmethod
    def _prop_list(base_path):
        return [
            p.name
            for p in base_path.iter_dir()
            if p.is_dir() and not p.name.startswith('.')
        ]
    def _load_prop(self, name, sid, raters=None):
        prop_dirpath = self.base_path / name
        if raters is None:
            impaths = prop_dirpath.glob(f"{sid}.*.mgz")
            impath = str(next(impaths))
        else:
            impath = prop_dirpath / f"{sid}.*.mgz"
        prop_mgz = ny.load(impath)
        prop = np.asarray(prop_mgz).astype(dtype=np.float32)
        return prop
        
    @cfg.wrap_opts
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
        dset_dims = (nraters * nsids, nprops, 163842)

        self.lh_graphs = {}
        self.rh_graphs = {}
        for sid in sids:
            lh_mesh_path = os.path.join(self.base_path, f"{sid}.lh.mesh")
            rh_mesh_path = os.path.join(self.base_path, f"{sid}.rh.mesh")

            lh_mesh = ny.load(lh_mesh_path, 'freesurfer_geometry')
            rh_mesh = ny.load(rh_mesh_path, 'freesurfer_geometry')
            
            lh_vertices = lh_mesh.coordinates
            rh_vertices = rh_mesh.coordinates
            
            lh_edges = lh_mesh.tess.edges
            rh_edges = rh_mesh.tess.edges

            lh_graph = self.create_graph(lh_vertices, lh_edges)
            rh_graph = self.create_graph(rh_vertices, rh_edges)

            self.lh_graphs[sid] = lh_graph
            self.rh_graphs[sid] = rh_graph
            print("Saved:", sid)

        self.data = torch.zeros(dset_dims)
        ii = 0
        for rater in raters:
            for sid in sids:
                for (pii, propname) in enumerate(properties):
                    prop = self._load_prop(propname, sid, rater)
                    self.data[ii, pii, ...] = torch.from_numpy(prop).float()
                ii += 1
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, k):
        return self.data[k].to(self.device)