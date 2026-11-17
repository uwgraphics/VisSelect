"""
A Python package for designing subset selection for data visualization by blending visualization goals as composable objective functions.
"""

from importlib.metadata import version

# import top-level classes
from .dataset import Dataset
from .subset import Subset
from .loss import Loss
from .problem import Problem
from .solver import Solver

# import submodules
from . import algorithm
from . import metric
from . import objective
from . import scalarization


__all__ = [
    "Dataset", 
    "Subset", 
    "Loss", 
    "Problem", 
    "Solver",
    "algorithm",
    "metric", 
    "objective", 
    "scalarization"
    ]

__version__ = version("visselect")