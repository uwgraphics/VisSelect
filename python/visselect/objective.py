# --- Imports ------------------------------------------------------------------

# standard library
from typing import Any, Callable

# third party
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

try:
    import ot
except ImportError:
    ot = None

# local files
from . import metric
from .dataset import Dataset
from .subset import Subset


# --- Objective Classes --------------------------------------------------------

class Objective:
    """
    Base class of the objectives, holding the transform they evaluate on
    """

    def __init__(self, on: str = "data") -> None:
        """
        Initialize an objective

        Args:
            on: The name of the transform to run the objective on
        """
        self._on = on

    @property
    def on(self) -> str:
        """
        The name of the transform the objective is evaluated on
        """
        return self._on


class Preserve(Objective):
    """
    Minimize the difference between the dataset and subset in terms of a metric
    """

    def __init__(
        self,
        metric: Callable,
        p: int | str = 1,
        on: str = "data"
    ) -> None:
        """
        Initialize a metric preserving objective

        Args:
            metric: The metric applied to the data to preserve in the subset
            p: The order of the norm of the difference between metrics: 1, 2, ∞
            on: The name of the transform to run the objective on

        Raises:
            TypeError: If metric is not callable
        """
        if not callable(metric):
            raise TypeError("Metric is not a callable function")

        super().__init__(on)
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
        datasetMetric = subset.dataset.evaluate(self._metric, self._on)
        subsetMetric = self._metric(subset[self._on])

        # If the metric results are scalars, use the absolute difference
        if np.ndim(datasetMetric) == 0:
            return np.abs(datasetMetric - subsetMetric)

        # Otherwise, use np.linalg.norm for array-like metric results
        return np.linalg.norm(datasetMetric - subsetMetric, ord=self._p)

    def encode(
        self,
        model: Any,
        z: Any,
        dataset: Dataset,
        size: int
    ) -> Any:
        """
        Encode the objective as a MIP expression of z

        Args:
            model: The MIP model to encode the objective into
            z: The subset decision variables, one binary per dataset item
            dataset: The dataset to apply subset selection to
            size: The size of the subset to select

        Raises:
            ValueError: If the metric or norm order has no MIP encoding
        """
        if self._metric is not metric.mean or self._p != 1:
            raise ValueError(f"No encoding for metric {self._metric.__name__}")

        data = dataset[self._on]
        datasetMetric = dataset.evaluate(self._metric, self._on)
        subsetMetric = (data.T @ z) / size # constant size fixed

        # per-feature absolute differences linearized with nonnegative t
        t = model.addMVar(data.shape[1])
        model.addConstr(t >= subsetMetric - datasetMetric)
        model.addConstr(t >= datasetMetric - subsetMetric)
        return t.sum()


class Distinctness(Objective):
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
        data = subset[self._on]
        if data.shape[0] < 2: # a lone item has no neighbor to differ from
            return 0.0
        distances = cdist(data, data)
        np.fill_diagonal(distances, np.inf) # an item is not its own neighbor

        return -np.sum(np.min(distances, axis=1))


class Crossings(Objective):
    """
    Minimize the rate of line crossings the subset draws between the
    neighboring axes of a parallel coordinates chart
    """

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the share of the subset's line pairs crossing between each
        pair of neighboring axes: lines cross when the sign of their
        difference flips from one axis to the next, with ties not crossing

        Args:
            subset: The subset to evaluate the crossings of
        """
        data = subset[self._on]
        items, features = data.shape
        if items < 2 or features < 2: # nothing can cross
            return 0.0

        crossings = 0
        for axis in range(features - 1):
            left = data[:, axis]
            right = data[:, axis + 1]
            crossings += np.sum(
                np.sign(left[:, None] - left[None, :])
                * np.sign(right[:, None] - right[None, :]) < 0
            )
        # both orders of every pair were counted, and so does the divisor
        return float(crossings / (items * (items - 1) * (features - 1)))


class EnergyDistance(Objective):
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
        mean pairwise distance

        Args:
            subset: The subset to evaluate the distribution distance of
        """
        means = subset.dataset.evaluate(self._meanDistances, self._on)
        between = np.compress(subset.indicator, means).mean()
        data = subset[self._on]
        within = cdist(data, data).mean()
        return 2.0 * between - within

    def _meanDistances(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the mean distance from each item to the whole dataset,
        a per-item constant no selection changes

        Args:
            data: The data array of the dataset
        """
        return cdist(data, data).mean(axis=1)


class EarthMoversDistance(Objective):
    """
    Minimize the earth mover's distance between the distributions of the
    subset and the dataset
    """

    def __init__(self, on: str = "data") -> None:
        """
        Initialize an earth mover's distance objective

        Args:
            on: The name of the transform to run the objective on

        Raises:
            ImportError: if POT is not installed
        """
        if ot is None:
            raise ImportError("EarthMoversDistance requires POT")
        super().__init__(on)

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the least total cost of transporting the subset's uniform
        distribution onto the dataset's, over Euclidean ground distances,
        with the POT library

        Args:
            subset: The subset to evaluate the distribution distance of
        """
        data = subset[self._on]
        dataset = subset.dataset[self._on]
        return float(ot.emd2([], [], cdist(data, dataset)))


class Outliers(Objective):
    """
    Maximize how outlying the items of the subset are in the dataset
    """

    def __init__(
        self,
        neighbors: int = 20,
        on: str = "data"
    ) -> None:
        """
        Initialize an outliers objective

        Args:
            neighbors: The number of nearest neighbors each item considers
            on: The name of the transform to run the objective on

        Raises:
            ValueError: If neighbors is less than 1
        """
        if neighbors < 1:
            raise ValueError("Neighbors is less than 1")
        super().__init__(on)
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
        factors = subset.dataset.evaluate(self._factors, self._on)
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


class ClusterCenters(Objective):
    """
    Minimize the distance from each cluster center of the dataset to the
    item of the subset nearest to it
    """

    def __init__(
        self,
        k: int = 5,
        seed: int | None = None,
        on: str = "data"
    ) -> None:
        """
        Initialize a cluster centers objective

        Args:
            k: The number of cluster centers the subset represents
            seed: The random seed for reproducibility
            on: The name of the transform to run the objective on

        Raises:
            ValueError: If k is less than 1
        """
        if k < 1:
            raise ValueError("k is less than 1")
        super().__init__(on)
        self._k = k
        self._seed = seed

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the sum of the distances from each cluster center to the
        item of the subset nearest to it

        Args:
            subset: The subset to evaluate as cluster representatives

        Raises:
            ValueError: If the dataset does not have more items than k
        """
        centers = subset.dataset.evaluate(self._centers, self._on)
        data = subset[self._on]
        return cdist(centers, data).min(axis=1).sum()

    def _centers(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the k-means cluster centers of the items, from a k-means++
        start so no cluster starts empty

        Args:
            data: The data array of the dataset

        Raises:
            ValueError: If the number of items is not more than k
        """
        if data.shape[0] <= self._k:
            raise ValueError("k is ≥ the number of items")

        centers, _ = kmeans2(
            data.astype(float), self._k, minit="++", seed=self._seed
        )
        return centers


class DiscreteCoverage(Objective):
    """
    Minimize the share of the dataset's categories the subset leaves
    uncovered, over a one hot encoded transform
    """

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Compute the share of the categories of the dataset, one per column
        of the one hot encoding, that no item of the subset covers

        Args:
            subset: The subset to evaluate the coverage of

        Raises:
            ValueError: If the transform is not one hot encoded
        """
        categories = subset.dataset.evaluate(self._categories, self._on)
        data = subset[self._on]
        covered = np.count_nonzero(data.any(axis=0))
        return 1.0 - covered / categories

    def _categories(self, data: np.ndarray) -> int:
        """
        Count the categories of the one hot encoding, one per column

        Args:
            data: The data array of the dataset

        Raises:
            ValueError: If the array holds values other than 0 and 1
        """
        if np.any((data != 0) & (data != 1)):
            raise ValueError("Transform is not one hot encoded")
        return data.shape[1]
