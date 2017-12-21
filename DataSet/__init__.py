from .CUB200 import CUB200
from .Car196 import Car196
__factory = {
    'cub': CUB200,
    'car': Car196,	
}


def names():
    return sorted(__factory.keys())


def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
