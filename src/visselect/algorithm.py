# --- Imports ------------------------------------------------------------------

# standard library
from typing import Iterator

# third party
import numpy as np

# local files
from .problem import Problem
from .subset import Subset


# --- Algorithm Implementations ------------------------------------------------

class Random:
    """
    Selects subsets randomly
    """

    def __init__(
        self, 
        samples: int = 1, 
        seed: int | None = None
    ) -> None:
        """
        Initialize a random selection algorithm

        Args: 
            samples: The number of random samples to take and evaluate
            seed: The random seed for reproducibility

        Raises: 
            ValueError: If the number of samples < 1
        """
        if samples < 1:
            raise ValueError("Samples is less than 1")
        self._samples = samples

        self._seed = seed


    def __call__(
        self,
        problem: Problem,
    ) -> Iterator[tuple[Subset, float]]:
        """
        Runs random selection, yielding a random subset on each call

        Args:
            problem: The subset selection problem definition to sample from
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        indicator = np.zeros(length, dtype=bool)
        indices = np.arange(problem.size)
        indicator[indices] = True
        subset = Subset(problem.dataset, indicator)

        for _ in range(self._samples):
            indicator[indices] = False
            indices = rng.choice(length, size=problem.size, replace=False)
            indicator[indices] = True
            yield subset, float(problem.loss(subset))

    @property
    def seed(self) -> int:
        """
        The random seed for reproducibility
        """
        return self._seed