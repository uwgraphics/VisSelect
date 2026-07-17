# --- Imports ------------------------------------------------------------------

import numpy as np


# --- Dataset Class ------------------------------------------------------------

class Dataset:
    """
    A class for storing a dataset for subset selection.
    """

    def __init__(
            self, 
            data: np.ndarray, 
            features: list[str] | None = None
        ) -> None:
        """
        Initialize a dataset with a NumPy array.
        
        Args: 
            data: A 2D NumPy array containing the dataset in tabular form with column features and row items
            features: A list of feature names assigned to columns of the dataset to support named feature queries

        Raises: 
            TypeError: If data is not a NumPy array
            ValueError: If data is not 2D, data has no rows/columns, length of features and data columns differ, or features has duplicates
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data is {type(data).__name__}, not a NumPy array")
        if data.ndim != 2:
            raise ValueError(f"Data is {data.ndim}D, not 2D")
        if data.shape[0] == 0:
            raise ValueError("Data has no items")
        if data.shape[1] == 0:
            raise ValueError("Data has no features")
        
        if features is not None:
            if len(features) != data.shape[1]:
                raise ValueError("Length of features and data columns differs")
            if len(set(features)) != len(features): 
                raise ValueError("Features list contains duplicates")
            self._features = list(features)

        self._root = data

    @property
    def features(self) -> list[str]:
        """
        The feature names of the dataset columns
        """
        if not hasattr(self, "_features"):
            self._features = [str(i) for i in range(self.size[1])]
        return self._features

    @property
    def size(self) -> tuple[int, int]:
        """
        The tuple (rows, columns) of the size of the dataset
        """
        return self._root.shape

    def __len__(self) -> int:
        """
        Compute the length as the number of rows in the dataset
        """
        return self.size[0]