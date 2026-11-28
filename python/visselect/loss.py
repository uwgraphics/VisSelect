# --- Imports ------------------------------------------------------------------

# standard library
from typing import Any, Callable, Sequence

# third party
import numpy as np

# local files
from .dataset import Dataset
from .subset import Subset
from .scalarization import Chebyshev


# --- Loss Function Class ------------------------------------------------------

class Loss:
    """
    A loss function defined by a set of objectives and a scalarization
    """

    def __init__(
        self,
        objectives: Sequence[Callable],
        scalarization: Callable | None = None,
        utopia: Sequence[float] | None = None,
        nadir: Sequence[float] | None = None
    ) -> None:
        """
        Initialize a loss function with a set of objectives, a scalarization,
        and the utopia-nadir normalization bounds of each objective

        Args:
            objectives: A sequence of objective functions
            scalarization: A method for scalarizing the objective-space vector
            utopia: The best achievable value of each objective
            nadir: The worst value of each objective across the
                single-objective optima

        Raises:
            ValueError: if the objectives list is empty, only one of 
                utopia-nadir is given, their lengths differ from the 
                objectives, or a nadir is not above its utopia
        """
        if len(objectives) < 1:
            raise ValueError("Objectives list is empty")
        if scalarization is None:
            scalarization = Chebyshev()
        if (utopia is None) != (nadir is None):
            raise ValueError("Utopia and nadir come together")
        if utopia is not None:
            utopia = np.asarray(utopia, dtype=float)
            nadir = np.asarray(nadir, dtype=float)
            if len(utopia) != len(objectives) or len(nadir) != len(objectives):
                raise ValueError("Lengths of bounds and objectives differ")
            if np.any(nadir <= utopia):
                raise ValueError("Nadir is not above utopia")

        self._objectives = list(objectives)
        self._scalarization = scalarization
        self._utopia = utopia
        self._nadir = nadir

    def vector(self, subset: Subset) -> np.ndarray:
        """
        The objective space vector from evaluating each objective on the subset

        Args: 
            subset: The subset to evaluate the objective-space vector on
        """
        return np.array([objective(subset) for objective in self._objectives])

    def __call__(self, subset: Subset) -> float:
        """
        Evaluate the scalarization of the objective-space vector, normalized
        to each objective's utopia-nadir bounds

        Args:
            subset: The subset to evaluate the loss on
        """
        vector = self.vector(subset)
        if self._utopia is not None:
            vector = (vector - self._utopia) / (self._nadir - self._utopia)
        return self._scalarization(vector)

    def encode(
        self,
        model: Any,
        z: Any,
        dataset: Dataset,
        size: int
    ) -> Any:
        """
        Encode the loss as the scalarization of the objectives' MIP expressions

        Args:
            model: The MIP model to encode the loss into
            z: The subset decision variables, one binary per dataset item
            dataset: The dataset to apply subset selection to
            size: The size of the subset to select

        Raises:
            ValueError: If any objective has no encoding
        """
        noEncoding = []
        encoding = []
        for objective in self._objectives:
            if not hasattr(objective, "encode"):
                noEncoding.append(getattr(objective, "__name__", type(objective).__name__))
            else:
                encoding.append(objective.encode(model, z, dataset, size))
        if noEncoding:
            raise ValueError(f"No encoding for {', '.join(noEncoding)}")

        if self._utopia is not None: # the same affine map evaluation applies
            encoding = [
                (expression - utopia) / (nadir - utopia)
                for expression, utopia, nadir
                in zip(encoding, self._utopia, self._nadir)
            ]
        return self._scalarization.encode(model, encoding)

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
        The method for scalarizing the objective-space vector
        """
        return self._scalarization

    @property
    def utopia(self) -> np.ndarray | None:
        """
        The best achievable value of each objective, or None unnormalized
        """
        return self._utopia

    @property
    def nadir(self) -> np.ndarray | None:
        """
        The worst value of each objective across the single-objective
        optima, or None unnormalized
        """
        return self._nadir
