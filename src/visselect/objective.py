# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# third party
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# local files
from .subset import Subset


# --- Preserve Metric Objective ------------------------------------------------

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
            metric: The metric applied to the data to preserve in the subset
            p: The order of the norm of the difference between metrics: 1, 2, ∞

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
        Compute the distance of the subset from the dataset

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


# --- Distinctness Objective ---------------------------------------------------

class Distinctness:
    """
    Maximize the distance from each item in the subset to its nearest neighbor
    """

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the negative sum of each item's distance to its nearest neighbor

        Args:
            subset: The subset to evaluate the distinctness of
        """
        data = subset.data
        distances = cdist(data, data)
        np.fill_diagonal(distances, np.inf) # an item is not its own neighbor

        return -np.sum(np.min(distances, axis=1))


# --- Energy Distance Objective ------------------------------------------------

class EnergyDistance:
    """
    Minimize the energy distance between the distributions of the subset and
    the dataset
    """

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Evaluate the energy distance between the distributions of the subset
        and the dataset (Székely and Rizzo, 2013), up to the dataset's own
        mean pairwise distance, a constant no selection changes

        Args:
            subset: The subset to evaluate the distribution distance of
        """
        data = subset.data
        between = cdist(data, subset.dataset.data).mean()
        within = cdist(data, data).mean()
        return 2.0 * between - within


# --- Outliers Objective -------------------------------------------------------

class Outliers:
    """
    Maximize how outlying the items of the subset are in the dataset
    """

    def __init__(self, neighbors: int = 20) -> None:
        """
        Initialize an outliers objective

        Args:
            neighbors: The number of nearest neighbors each item's local
                outlier factor considers

        Raises:
            ValueError: If neighbors is less than 1
        """
        if neighbors < 1:
            raise ValueError("Neighbors is less than 1")
        self._neighbors = neighbors

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the negative sum of the local outlier factors of the subset

        Args:
            subset: The subset to evaluate the outlierness of

        Raises:
            ValueError: If the dataset does not have more items than neighbors
        """
        factors = subset.dataset.evaluate(self._factors)
        return -np.sum(np.compress(subset.indicator, factors))

    def _factors(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the local outlier factor of every item (Breunig et al., 2000)

        Args:
            data: The data array of the dataset

        Raises:
            ValueError: If the number of neighbors is ≥ the number of items
        """
        k = self._neighbors
        if data.shape[0] <= k:
            raise ValueError("Neighbors is ≥ the number of items")

        distances, neighbors = KDTree(data).query(data, k=k + 1)
        distances = distances[:, 1:] # an item is not its own neighbor
        neighbors = neighbors[:, 1:]

        # how far an item reaches for each of its neighbors
        reachability = np.maximum(distances[:, -1][neighbors], distances)
        density = k / np.sum(reachability, axis=1) # local reachability
        return np.sum(density[neighbors], axis=1) / k / density
