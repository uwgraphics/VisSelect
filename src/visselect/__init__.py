"""
A Python package for designing subset selection for data visualization by blending visualization goals as composable objective functions.
"""

from importlib.metadata import version

# Import top-level classes
from .dataset import Dataset

__version__ = version("visselect")
__all__ = ["Dataset"]
