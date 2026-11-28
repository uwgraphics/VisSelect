# --- Imports ------------------------------------------------------------------

# local files
from .dataset import Dataset
from .loss import Loss
from .subset import Subset


# --- Problem Class ------------------------------------------------------------

class Problem:
    """
    The problem definition based on a dataset, a loss function, and one way
    of setting the subset size
    """

    def __init__(
        self,
        dataset: Dataset,
        loss: Loss,
        size: int | None = None,
        epsilon: float | None = None,
        sizeWeight: float | None = None
    ) -> None:
        """
        Define a problem with a dataset, a loss function, and one way of
        setting the subset size: fixed at a size, the smallest subset whose
        loss is within epsilon, or the size minimized with the loss as a
        weighted term of it

        Args:
            dataset: The dataset to apply subset selection to
            loss: The loss function to evaluate candidate subsets against
            size: The size of the subset to select
            epsilon: The loss within which the smallest subset is selected,
                in the loss's units: a range fraction when it has bounds
            sizeWeight: The weight balancing the subset size against the
                loss, in the loss's units

        Raises:
            ValueError: if none or several of size, epsilon and sizeWeight
                are given, size is < 1 or ≥ dataset length, epsilon
                is negative, sizeWeight is not positive, or an objective's
                transform is not numeric
            KeyError: if an objective names a transform the dataset does
                not have
        """
        given = [value for value in (size, epsilon, sizeWeight)
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                "One of size, epsilon or sizeWeight is required"
            )
        if size is not None:
            if size < 1:
                raise ValueError("Size < 1")
            if size >= len(dataset):
                raise ValueError("Size ≥ dataset length")
            size = int(size)
        if epsilon is not None and epsilon < 0:
            raise ValueError("Epsilon is negative")
        if sizeWeight is not None and sizeWeight <= 0:
            raise ValueError("Size weight is not positive")

        # objectives compute on numbers
        for objective in loss.objectives:
            name = getattr(objective, "on", None)
            if name is not None and dataset[name].dtype.kind in "OUS":
                raise ValueError(f"Transform {name} is not numeric")

        self._dataset = dataset
        self._loss = loss
        self._size = size
        self._epsilon = epsilon
        self._sizeWeight = sizeWeight

    def evaluate(self, subset: Subset, loss: float | None = None) -> float:
        """
        Evaluate the total a solve minimizes on a candidate subset: the
        loss itself, the loss plus the weighted size when the size is
        minimized with it, or, bounding the loss within epsilon, the size
        once the subset is feasible — and before that the dataset length
        plus the excess loss, which mixes units but stays finite, above
        every feasible total, and falls as the loss falls

        Args:
            subset: The subset to evaluate the total on
            loss: The subset's loss, when it is already evaluated
        """
        if loss is None:
            loss = float(self._loss(subset))
        if self._sizeWeight is not None:
            return loss + self._sizeWeight * len(subset)
        if self._epsilon is not None:
            if loss <= self._epsilon:
                return float(len(subset))
            return len(self._dataset) + loss - self._epsilon
        return loss

    def feasible(self, subset: Subset) -> bool:
        """
        Whether a candidate subset's loss is within epsilon, which every
        subset is when no epsilon bounds the loss

        Args:
            subset: The subset to evaluate the loss bound on
        """
        if self._epsilon is None:
            return True
        return float(self._loss(subset)) <= self._epsilon

    @property
    def dataset(self) -> Dataset:
        """
        The dataset to apply subset selection to
        """
        return self._dataset

    @property
    def loss(self) -> Loss:
        """
        The loss function to evaluate candidate subsets against
        """
        return self._loss

    @property
    def size(self) -> int | None:
        """
        The size of the subset to select, or None when the size is variable
        """
        return self._size

    @property
    def epsilon(self) -> float | None:
        """
        The loss within which the smallest subset is selected, if set
        """
        return self._epsilon

    @property
    def sizeWeight(self) -> float | None:
        """
        The weight balancing the subset size against the loss, if set
        """
        return self._sizeWeight
