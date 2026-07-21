from ._model import SplineGCN
from ._data import HCPDataset

SplineGCN.__module__ = __name__
HCPDataset.__module__ = __name__

def reload():
    import sys, importlib
    importlib.reload(sys.modules['mesh_annot._model'])
    importlib.reload(sys.modules['mesh_annot._data'])