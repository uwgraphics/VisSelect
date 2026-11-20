# --- Imports ------------------------------------------------------------------

# third party
import numpy as np
from scipy.spatial import ConvexHull


# --- Metric Functions ---------------------------------------------------------

def mean(array: np.ndarray) -> np.ndarray:
    """The means of each column feature of the array"""
    return np.mean(array, axis=0)

def variance(array: np.ndarray) -> np.ndarray:
    """The population variance of each column feature of the array"""
    return np.var(array, axis=0, ddof=0)

def hull(array: np.ndarray) -> float:
    """The volume (area at two features) of the convex hull of the array"""
    return ConvexHull(array).volume