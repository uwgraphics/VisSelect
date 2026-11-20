# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# third party
import numpy as np
from scipy.spatial.distance import cdist

# local files
from .subset import Subset


# --- Preserve Metric Class ----------------------------------------------------

class Preserve:
    """
    Minimize the difference between the dataset and subset in terms of a metric
    """

    def __init__(
        self,
        metric: Callable,
        p: int | str = 1
    ) -> None:
        """
        Initialize a metric preserving objective

        Args:
            metric: The metric function to apply to the data
            p: Specify the ord of the norm from linalg.norm NumPy options

        Raises:
            TypeError: If metric is not callable
        """
        if not callable(metric):
            raise TypeError("Metric is not a callable function")

        self._metric = metric
        self._p = p

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Evaluate the distance of the subset from the dataset

        Args:
            subset: The subset to evaluate the distance from the target
        """
        datasetMetric = subset.dataset.evaluate(self._metric)
        subsetMetric = self._metric(subset.data)

        # If the metric results are scalars, use the absolute difference
        if np.ndim(datasetMetric) == 0:
            return np.abs(datasetMetric - subsetMetric)

        # Otherwise, use np.linalg.norm for array-like metric results
        return np.linalg.norm(datasetMetric - subsetMetric, ord=self._p)


# --- Distinctness Class -------------------------------------------------------

class Distinctness:
    """
    Maximize the distance from each item in the subset to its nearest neighbor
    """

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Evaluate the negative sum of each item's distance to its nearest
        neighbor in the subset, computed from the subset's own distances

        Args:
            subset: The subset to evaluate the distinctness of
        """
        data = subset.data
        distances = cdist(data, data)
        np.fill_diagonal(distances, np.inf) # an item is not its own neighbor

        return -np.sum(np.min(distances, axis=1))
