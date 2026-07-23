# THIS MODULE CONTAINS THE CONFIGURATION ITEMS, PRIMARILY DEFAULT VALUES, FOR THE mesh-annot LIBRARY.

properties = ('convexity','myelin','thickness','curvature', 'prf_x', 'prf_y', 'prf_cod', 'prf_sigma')

image_size = (128, 256)
epochs = 200
batch_size = 20
kernel_size = 5
dropout = 0.1
lr = 1e-4
lr_min = 1e-5
lr_patience = 10
lr_decay = 0.8 # Set the lr_decay to None to prevent plateau decay from being used.
weight_decay = 1e-4
noise_std = None
outputs = 1

# NOT A HYPERPARAMETER: The default device is set here.
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
del torch

# Collects and returns the hyperparameters for training. All hyperparameters have default value of "Ellipsis",
# any such value is replaced with the hyperparameter in the "mesh_annot.config" namespace.
def hyperparams(
    properties=Ellipsis,
    image_size=Ellipsis,
    epochs=Ellipsis,
    batch_size=Ellipsis,
    kernel_size=Ellipsis,
    dropout=Ellipsis,
    device=Ellipsis,
    lr=Ellipsis,
    lr_min=Ellipsis,
    lr_patience=Ellipsis,
    lr_decay=Ellipsis,
    weight_decay=Ellipsis,
    noise_std=Ellipsis,
    outputs=Ellipsis
):
    from . import config as cfg
    properties = cfg.properties if properties is Ellipsis else properties
    image_size = cfg.image_size if image_size is Ellipsis else image_size
    epochs = cfg.epochs if epochs is Ellipsis else epochs
    batch_size = cfg.batch_size if batch_size is Ellipsis else batch_size
    kernel_size = cfg.kernel_size if kernel_size is Ellipsis else kernel_size
    dropout = cfg.dropout if dropout is Ellipsis else dropout
    device = cfg.device if device is Ellipsis else device
    lr = cfg.lr if lr is Ellipsis else lr
    lr_min = cfg.lr_min if lr_min is Ellipsis else lr_min
    lr_patience = cfg.lr_patience if lr_patience is Ellipsis else lr_patience
    lr_decay = cfg.lr_decay if lr_decay is Ellipsis else lr_decay
    weight_decay = cfg.weight_decay if weight_decay is Ellipsis else weight_decay
    noise_std = cfg.noise_std if noise_std is Ellipsis else noise_std
    outputs = cfg.outputs if outputs is Ellipsis else outputs
    
    return dict(
        properties=properties,
        image_size=image_size,
        epochs=epochs,
        batch_size=batch_size,
        kernel_size=kernel_size,
        dropout=dropout,
        device=device,
        lr=lr,
        lr_min=lr_min,
        lr_patience=lr_patience,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        noise_std=noise_std
    )

# Wraps a function such that any training hyperparameters it takes are
# automatically filled in with default values from the mesh_annot.config
# namespace if they are equal to "Ellipsis"
def wrap_opts(fn):
    import inspect
    from functools import wraps
    from . import config as cfg
    
    # Get the signature of the function
    sig = inspect.signature(fn)
    hpsig = inspect.signature(hyperparams)
    
    opts = [
        k
        for k in sig.parameters.keys()
        if hasattr(cfg, k) and 'hyperparam' not in k
    ]
    
    # Make the decorated function
    @wraps(fn)
    def decorated_fn(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for opt in opts:
            if bound.arguments.get(opt, Ellipsis) is Ellipsis:
                bound.arguments[opt] = getattr(cfg, opt)
        if 'hyperparams' in sig.parameters:
            bound.arguments['hyperparams'] = hyperparams(
                **{
                    k:bound.arguments[k]
                    for k in hpsig.parameters.keys()
                    if k in bound.arguments
                }
            )
            
        return fn(*bound.args, **bound.kwargs)
    return decorated_fn