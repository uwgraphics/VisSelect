# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# third party 
import numpy as np

# local files
from .dataset import Dataset
from .subset import Subset


# --- Preserve Metric Class ----------------------------------------------------

class Preserve:
    """
    Minimize the difference between the dataset and subset in terms of a metric
    """

    def __init__(
        self,
        metric: Callable,
        dataset: Dataset,
        p: int | str = 1
    ) -> None:
        """
        Define the metric distance to be minimized between a dataset and subset 

        Args:
            metric: The metric function to apply to the data
            dataset: The dataset to evaluate the target for in metric space
            p: Specify the ord of the norm from linalg.norm NumPy options
        """
        self._metric = metric
        self._p = p
        self._target = metric(dataset.data)
        self._scalar = np.ndim(self._target) == 0

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Evaluate the distance of the subset from the dataset

        Args:
            subset: The subset to evaluate the distance from the target
        """
        subsetMetric = self._metric(subset.data)

        # If the metric results are scalars, use the absolute difference
        if self._scalar:
            return np.abs(self._target - subsetMetric)

        # Otherwise, use np.linalg.norm for array-like metric results
        return np.linalg.norm(self._target - subsetMetric, ord=self._p)
