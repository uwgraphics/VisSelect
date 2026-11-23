# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable, Sequence

# third party
import numpy as np

# local files
from .subset import Subset
from .scalarization import WeightedSum


# --- Loss Function Class ------------------------------------------------------

class Loss:
    """
    A loss function defined by a set of objectives and a scalarization
    """

    def __init__(
        self,
        objectives: Sequence[Callable],
        scalarization: Callable | None = None
    ) -> None:
        """
        Initialize a loss function with a set of objectives and a scalarization

        Args:
            objectives: A sequence of objective functions
            scalarization: A method for scalarizing the objective-space vector

        Raises: 
            ValueError: if objectives list is empty
        """
        if len(objectives) < 1:
            raise ValueError("Objectives list is empty")
        if scalarization is None:
            scalarization = WeightedSum()

        self._objectives = list(objectives)
        self._scalarization = scalarization

    def vector(self, subset: Subset) -> np.ndarray:
        """
        The objective space vector from evaluating each objective on the subset

        Args: 
            subset: The subset to evaluate the objective-space vector on
        """
        return np.array([objective(subset) for objective in self._objectives])

    def __call__(self, subset: Subset) -> float:
        """
        Evaluate the scalarization of the objective-space vector

        Args: 
            subset: The subset to evaluate the loss on
        """
        return self._scalarization(self.vector(subset))
    
    def __len__(self) -> int:
        """
        Compute the length as the number of objectives in the set
        """
        return len(self._objectives)

    @property
    def objectives(self) -> list[Callable]:
        """
        A sequence of objective functions
        """
        return self._objectives

    @property
    def scalarization(self) -> Callable:
        """
        A method for scalarizing the objective-space vector
        """
        return self._scalarization
