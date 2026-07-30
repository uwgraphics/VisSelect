# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# third party
import numpy as np

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
