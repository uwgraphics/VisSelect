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
    def evaluations(self) -> int:
        """
        The number of evaluations a run takes: one per random sample
        """
        return self._samples

    @property
    def seed(self) -> int:
        """
        The random seed for reproducibility
        """
        return self._seed


class GreedySwap:
    """
    Selects a subset by swapping out items and keeping those that lower the loss
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize a greedy swap algorithm

        Args:
            seed: The random seed for reproducibility
        """
        self._seed = seed

    def __call__(
        self,
        problem: Problem,
    ) -> Iterator[tuple[Subset, float]]:
        """
        Starts from a random subset and swaps an item in if it improves the loss

        Args:
            problem: The subset selection problem definition to solve
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        indicator = np.zeros(length, dtype=bool)
        members = rng.choice(length, size=problem.size, replace=False)
        indicator[members] = True
        subset = Subset(problem.dataset, indicator)

        loss = float(problem.loss(subset))
        yield subset, loss

        for item in rng.permutation(np.flatnonzero(~indicator)):
            bestLoss, bestPosition = loss, None

            # try swapping the item with each member, reverting after each try
            for position, member in enumerate(members):
                indicator[member] = False
                indicator[item] = True
                swapLoss = float(problem.loss(subset))
                yield subset, swapLoss
                indicator[item] = False
                indicator[member] = True

                if swapLoss < bestLoss:
                    bestLoss, bestPosition = swapLoss, position

            # keep the best swap if it improved on the current subset
            if bestPosition is not None:
                indicator[members[bestPosition]] = False
                indicator[item] = True
                members[bestPosition] = item
                loss = bestLoss

    def evaluations(self, problem: Problem) -> int:
        """
        The number of evaluations a run takes: a random starting subset plus 
        one per swap of each item outside the subset

        Args:
            problem: The problem the run would solve
        """
        size = problem.size
        return 1 + size * (len(problem.dataset) - size)

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed