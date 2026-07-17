from ._model import SplineGCN

SplineGCN.__module__ = __name__

def reload():
    import sys, importlib
    importlib.reload(sys.modules['mesh_annot._model'])