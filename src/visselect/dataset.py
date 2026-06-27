# --- Imports ------------------------------------------------------------------

import numpy as np


# --- Dataset Class ------------------------------------------------------------

class Dataset:
    """
    A class for storing a dataset for subset selection.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize a dataset with a NumPy array.
        
        Arg: data: A 2D NumPy array containing the dataset in tabular form with column features and row items

        Raises: 
            TypeError: If data is not a NumPy array
            ValueError: If data is not 2D or has no rows/columns
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data is {type(data).__name__}, not a NumPy array")
        if data.ndim != 2:
            raise ValueError(f"Data is {data.ndim}D, not 2D")
        if data.shape[0] == 0:
            raise ValueError("Data has no items")
        if data.shape[1] == 0:
            raise ValueError("Data has no features")

        self._root = data

    @property
    def size(self) -> tuple[int, int]:
        """
        The tuple (rows, columns) of the size of the dataset
        """
        return self._root.shape

    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset
        """
        return self.size[0]