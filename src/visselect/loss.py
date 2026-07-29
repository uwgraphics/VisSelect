# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable, Sequence, SupportsFloat

# local files
from .subset import Subset

# --- Loss Function Class ------------------------------------------------------

class Loss:
    """
    A loss function defined by one or more objectives and corresponding weights
    """

    def __init__(
        self,
        objectives: Sequence[Callable],
        weights: Sequence[SupportsFloat] | None = None,
    ) -> None:
        """
        Initialize a loss function with one or more weighted objectives

        Args:
            objectives: A sequence of objective functions
            weights: Corresponding sequence of weights to apply to objectives

        Raises: 
            ValueError: if objectives list is empty or weights and objectives 
                do not have the same length
        """
        if len(objectives) < 1:
            raise ValueError("Objectives list is empty")
        if weights is None:
            weights = [1.0] * len(objectives)
        if len(weights) != len(objectives):
            raise ValueError("Lengths of weights and objectives differ")

        weights = [float(w) for w in weights]
        self._terms = list(zip(weights, objectives))

    def __call__(
        self,
        subset: Subset
    ) -> float:
        """
        Evaluate the weighted sum of the objectives on a subset

        Args: 
            subset: The subset to evaluate the loss on
        """
        loss = 0.0
        for weight, objective in self._terms:
            loss += weight * objective(subset)

        return loss