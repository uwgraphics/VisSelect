# --- Imports ------------------------------------------------------------------

# local files
from .dataset import Dataset
from .loss import Loss


# --- Problem Class ------------------------------------------------------------

class Problem:
    """
    The problem definition based on a dataset loss function and subset size
    """

    def __init__(
        self,
        dataset: Dataset,
        loss: Loss,
        size: int
    ) -> None:
        """
        Define a problem with a loss function, dataset and subset size

        Args:
            dataset: The dataset to apply subset selection to
            loss: The loss function to evaluate candidate subsets against
            size: The size of the subset to select

        Raises: 
            ValueError: if size is less than 1 or greater than dataset length
        """
        if size < 1:
            raise ValueError("Size < 1")
        if size >= len(dataset):
            raise ValueError("Size ≥ dataset length")
        
        self._dataset = dataset
        self._loss = loss
        self._size = int(size)

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
    def size(self) -> int:
        """
        The size of the subset to select
        """
        return self._size
    