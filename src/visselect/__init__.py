"""
A Python package for designing subset selection for data visualization by blending visualization goals as composable objective functions.
"""

from importlib.metadata import version

# import top-level classes
from .dataset import Dataset
from .subset import Subset

# import submodules
from . import metric
from . import objective

__version__ = version("visselect")
__all__ = ["Dataset", "Subset", "metric", "objective"]