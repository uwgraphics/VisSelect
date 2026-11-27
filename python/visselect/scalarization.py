# --- Imports ------------------------------------------------------------------

# standard library
from typing import Any, Sequence, SupportsFloat

# third party
import numpy as np


# --- Chebyshev Scalarization Class --------------------------------------------

class Chebyshev:
    """
    Compute the scalarization of a vector as its largest weighted objective
    """

    def __init__(
        self,
        weights: Sequence[SupportsFloat] | None = None
    ) -> None:
        """
        Initialize a Chebyshev scalarization with a weights vector

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
        Evaluate the largest weighted objective of the vector

        Args:
            vector: The vector to evaluate the weighted maximum on

        Raises:
            ValueError: if objectives and weights vectors lengths differ
        """
        if self._weights is None:
            return max(vector)
        if len(self._weights) != len(vector):
            raise ValueError("Lengths of weights and objectives vectors differ")

        return np.max(self._weights * vector)

    def encode(
        self,
        model: Any,
        expressions: Sequence[Any],
    ) -> Any:
        """
        Encode the weighted maximum of the objectives' MIP expressions as a
        bound variable above every weighted term, which minimizing presses
        down onto the largest of them

        Args:
            model: The MIP model to encode the scalarization into
            expressions: The objectives' MIP expressions to combine

        Raises:
            ValueError: If objectives and weights vectors lengths differ
        """
        weights = self._weights
        if weights is None:
            weights = np.ones(len(expressions))
        if len(weights) != len(expressions):
            raise ValueError("Lengths of weights and objectives vectors differ")

        bound = model.addVar(lb=-float("inf")) # losses can be negative
        for weight, expression in zip(weights, expressions):
            model.addConstr(bound >= weight * expression)
        return bound

    @property
    def weights(self) -> np.ndarray | None:
        """
        The weights vector to apply in the Chebyshev scalarization
        """
        return self._weights


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

    def encode(
        self,
        model: Any,
        expressions: Sequence[Any],
    ) -> Any:
        """
        Encode the linear scalarization of the objectives' MIP expressions

        Args:
            model: The MIP model the loss is encoded into, not needed here
            expressions: The objectives' MIP expressions to combine

        Raises:
            ValueError: If objectives and weights vectors lengths differ
        """
        if self._weights is None:
            return sum(expressions)
        if len(self._weights) != len(expressions):
            raise ValueError("Lengths of weights and objectives vectors differ")

        return sum(weight * expression for weight, expression
                   in zip(self._weights, expressions))

    @property
    def weights(self) -> np.ndarray | None:
        """
        The weights vector to apply in the linear scalarization
        """
        return self._weights