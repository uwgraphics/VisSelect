# --- Imports ------------------------------------------------------------------

# third party
import numpy as np


# --- Metric Functions ---------------------------------------------------------

def mean(array: np.ndarray) -> np.ndarray:
    """The means of each column feature of the array"""
    return np.mean(array, axis=0)

def variance(array: np.ndarray) -> np.ndarray:
    """The population variance of each column feature of the array"""
    return np.var(array, axis=0, ddof=0)