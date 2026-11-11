from .config import hyperparams
from ._model import (SplineGCN, SplineUNet)
from ._data import HCPDataset
from ._train import (train, loadrun)

SplineGCN.__module__ = __name__
SplineUNet.__module__ = __name__
HCPDataset.__module__ = __name__
train.__module__ = __name__
loadrun.__module__ = __name__

def reload():
    import sys, importlib
    importlib.reload(sys.modules['mesh_annot.config'])
    importlib.reload(sys.modules['mesh_annot._model'])
    importlib.reload(sys.modules['mesh_annot._data'])
    importlib.reload(sys.modules['mesh_annot._train'])
    return importlib.reload(sys.modules[__name__])