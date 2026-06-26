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
            ValueError: If data is not 2D
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data is {type(data).__name__}, not a NumPy array")
            
        if data.ndim != 2:
            raise ValueError(f"Data is {data.ndim}D, not 2D.")
        
        self._root = data