# --- Imports ------------------------------------------------------------------

# third party
import numpy as np

# local files
from .dataset import Dataset


# --- Subset Class -------------------------------------------------------------

class Subset:
    """
    A subset of items in a dataset created by indicator vector selection
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
            ValueError: If indicator is not boolean or if the indicator selects 
                no items or its length does not match the length of the dataset
            TypeError: If the indicator vector is not a NumPy array
        """

        if not isinstance(indicator, np.ndarray):
            raise TypeError("Indicator vector is not a NumPy ndarray")
        if not np.issubdtype(indicator.dtype, np.bool_):
            raise ValueError("Indicator vector is not a boolean selector")
        if len(indicator) != dataset.size[0]:
            raise ValueError("Lengths of indicator vector and dataset differ")
        if not np.any(indicator):
            raise ValueError("Indicator vector selects no items")
        
        self._dataset = dataset
        self._indicator = indicator

    def __len__(self) -> int:
        """
        The number of items contained in the subset
        """
        return np.count_nonzero(self._indicator)

    def __getitem__(self, name: str) -> np.ndarray:
        """
        The items selected in the subset from a named representation of the
        dataset, or from the data itself by "data"

        Args:
            name: The name of the representation, or "data" for the data

        Raises:
            KeyError: If no representation is stored under the name
        """
        return np.compress(self._indicator, self._dataset[name], axis=0)

    def freeze(self) -> "Subset":
        """
        Create an independent write-protected snapshot of the subset
        """
        indicator = self._indicator.copy()
        indicator.flags.writeable = False
        return Subset(self._dataset, indicator)

    @property
    def indicator(self) -> np.ndarray:
        """
        The indicator array of the items selected in the subset
        """
        return self._indicator

    @property
    def dataset(self) -> Dataset:
        """
        The dataset the subset selects items from
        """
        return self._dataset

    @property
    def size(self) -> tuple[int, int]:
        """
        The tuple (rows, columns) of the size of the subset
        """
        return len(self), self._dataset.size[1]