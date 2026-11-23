# --- Imports ------------------------------------------------------------------

# standard library
from typing import Sequence, SupportsFloat

# third party
import numpy as np


# --- Weighted Sum Scalarization Class -----------------------------------------

class WeightedSum:
    """
    Compute the linear scalarization of a vector by the weighted sum method
    """

    def __init__(
        self,
        weights: Sequence[SupportsFloat] | None = None
    ) -> None:
        """
        Initialize a linear scalarization with a weights vector

        Args:
            weights: Corresponding sequence of weights to apply to objectives

        Raises: 
            ValueError: if weights are not positive
        """
        if weights is not None:
            weights = [float(w) for w in weights]
            if any(w <= 0 for w in weights):
                raise ValueError("Weights are not strictly positive")
            weights = np.asarray(weights)

        self._weights = weights

    def __call__(
        self, 
        vector: np.ndarray,
    ) -> float:
        """
        Evaluate the linear scalarization of the vector

        Args:
            vector: The vector to evaluate the weighted sum on

        Raises: 
            ValueError: if objectives and weights vectors lengths differ
        """
        if self._weights is None:
            return sum(vector)
        if len(self._weights) != len(vector):
            raise ValueError("Lengths of weights and objectives vectors differ")
        
        return self._weights @ vector
    
    @property
    def weights(self) -> np.ndarray | None:
        """
        The weights vector to apply in the linear scalarization
        """
        return self._weights