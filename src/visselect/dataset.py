# --- Imports ------------------------------------------------------------------

# standard library
import csv
from contextlib import nullcontext
from os import PathLike
from typing import Any, Callable, Sequence, TextIO

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
        data: ArrayLike | str | PathLike | TextIO,
        features: Sequence[str] | None = None
    ) -> None:
        """
        Initialize a dataset with a 2D tabular data array

        Args:
            data: A 2D NumPy ArrayLike, a path to a csv file, or a text stream
                of a csv file containing a dataset in tabular form with column
                features and row items
            features: A sequence of feature names assigned to columns of the
                dataset to support named feature queries, which take precedence
                over features in csv headers or DataFrame columns

        Raises:
            TypeError: If data cannot be converted to a NumPy ndarray
            ValueError: If data is not 2D, data has no rows/columns, length of
                features and data columns differ, features has duplicates, or
                a csv file is empty or has non-numeric or ragged rows
            FileNotFoundError: If a csv file path does not exist
        """
        self._characteristics = {}
        dataFeatures = None # feature names supplied by the data itself, if any

        if isinstance(data, (str, PathLike)) or hasattr(data, "read"): # if file
            data, dataFeatures = self._read(data)
        elif hasattr(data, "to_numpy"): # if supported, attempt to_numpy convert
            if hasattr(data, "columns"):
                dataFeatures = [str(c) for c in data.columns]
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
        if features is None:
            features = dataFeatures
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

    def _read(
        self,
        source: str | PathLike | TextIO
    ) -> tuple[np.ndarray, list[str]]:
        """
        Reads a dataset from a csv file with a header row of feature names

        Args:
            source: A path to a csv file, or a text stream of a csv file which
                is read from its current position and left open

        Raises:
            FileNotFoundError: If a filepath is passed but does not exist
            ValueError: If the file is empty or has non-numeric or ragged rows
        """
        if isinstance(source, (str, PathLike)): # open paths, use streams as is
            context = open(source, encoding="utf-8")
        else:
            context = nullcontext(source)

        with context as file:
            line = file.readline()
            if not line.strip():
                raise ValueError("File is empty")
            header = next(csv.reader([line]))
            data = np.loadtxt(file, delimiter=",", ndmin=2)

        return data, header

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