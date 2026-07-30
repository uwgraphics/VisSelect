# --- Imports ------------------------------------------------------------------

# standard library
from typing import Any, Callable, Sequence

# third party
import numpy as np
from numpy.typing import ArrayLike


# --- Dataset Class ------------------------------------------------------------

class Dataset:
    """
    A tabular dataset for subset selection
    """

    def __init__(
        self, 
        data: ArrayLike, 
        features: Sequence[str] | None = None
    ) -> None:
        """
        Initialize a dataset with a 2D tabular data array
        
        Args: 
            data: A 2D NumPy ArrayLike (ndarray, or to_numpy/asarray coercible) 
                of a dataset in tabular form with column features and row items
            features: A sequence of feature names assigned to columns of the 
                dataset to support named feature queries

        Raises: 
            TypeError: If data cannot be converted to a NumPy ndarray
            ValueError: If data is not 2D, data has no rows/columns, length of 
                features and data columns differ, or features has duplicates
        """
        self._characteristics = {}
        if hasattr(data, "to_numpy"): # if supported, attempt to_numpy convert
            if features is None and hasattr(data, "columns"):
                features = [str(c) for c in data.columns]
            try:
                data = data.to_numpy()
            except Exception as e:
                raise TypeError(
                    f"Failed to convert {type(data).__name__} with to_numpy"
                ) from e
        try: # if supported, attempt asarray convert
            data = np.asarray(data)
        except Exception as e:
            raise TypeError(
                f"Failed to convert {type(data).__name__} with asarray"
            ) from e

        # verify the dataset loaded is a 2D array with both items and features
        if data.ndim != 2:
            raise ValueError(f"Data is {data.ndim}D, not 2D")
        if data.shape[0] == 0:
            raise ValueError("Data has no items")
        if data.shape[1] == 0:
            raise ValueError("Data has no features")
        
        # load and verify the feature column names if provided
        if features is not None:
            if len(features) != data.shape[1]:
                raise ValueError("Lengths of features and data columns differ")
            if len(set(features)) != len(features): 
                raise ValueError("Features list contains duplicates")
            self._features = list(features)

        # load the root as a non-writeable view of the supplied dataset
        self._root = data.view()
        self._root.flags.writeable = False

    def __getattr__(self, name: str) -> Any:
        """
        Look up a cached characteristic of the dataset by name

        Args:
            name: The name of the characteristic of the dataset to look up

        Raises: 
            AttributeError: If the name starts with "_" or the characteristic 
                has not been evaluated on the dataset
        """
        if name.startswith("_"):
            raise AttributeError(name)
        
        for function in self._characteristics:
            if getattr(function, "__name__", None) == name:
                return self._characteristics[function]

        raise AttributeError(f"Dataset has no characteristic {name}")

    def __len__(self) -> int:
        """
        Compute the length as the number of rows in the dataset
        """
        return self.size[0]

    def evaluate(self, function: Callable) -> Any:
        """
        Evaluate a metric function on the dataset and cache it

        Args: 
            function: The function to evaluate on the dataset
        """
        if function not in self._characteristics:
            self._characteristics[function] = function(self.data)

        return self._characteristics[function]

    @property
    def data(self) -> np.ndarray:
        """
        The data array of the dataset with column features and row items
        """
        return self._root

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