# --- Imports ------------------------------------------------------------------

# Third party
import numpy as np

# Local files
from .dataset import Dataset


# --- Subset Class -------------------------------------------------------------

class Subset:
    """
    A class for storing a subset of a dataset by indicator vector selection
    """

    def __init__(
        self, 
        dataset: Dataset, 
        indicator: np.ndarray, 
    ) -> None:
        """
        Initialize a subset with a Dataset object and a boolean indicator vector

        Args:
            dataset: The dataset from which to select items for the subset
            indicator: The boolean vector indicating items in the subset

        Raises: 
            ValueError: If indicator is not boolean or if the length of 
                indicator does not match the length of the dataset
            TypeError: If the indicator vector is not a NumPy array
        """

        if not isinstance(indicator, np.ndarray):
            raise TypeError("Indicator vector is not a NumPy ndarray")
        if not np.issubdtype(indicator.dtype, np.bool_):
            raise ValueError("Indicator vector is not a boolean selector")
        if len(indicator) != dataset.size[0]:
            raise ValueError("Lengths of indicator vector and dataset differ")

        self._dataset = dataset
        self._indicator = indicator