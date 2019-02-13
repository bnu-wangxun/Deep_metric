from __future__ import print_function, absolute_import

from .NCA import NCALoss
from .Contrastive import ContrastiveLoss
from .Binomial import BinomialLoss
from .LiftedStructure import LiftedStructureLoss
from .Weight import WeightLoss
from .HardMining import HardMiningLoss
from .DistWeighted import DistWeightedLoss

__factory = {
    'NCA': NCALoss,
    'Contrastive': ContrastiveLoss,
    'Binomial': BinomialLoss,
    'LiftedStructure': LiftedStructureLoss,
    'Weight': WeightLoss,
    'HardMining': HardMiningLoss,
    'Margin': DistWeightedLoss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)



def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)
