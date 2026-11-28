# --- Imports ------------------------------------------------------------------

# third party
import numpy as np

# local files
from .problem import Problem
from .subset import Subset

# --- Results Class ------------------------------------------------


class Results:    
    """
    A store of the results of a solve of an algorithm on a problem
    """

    def __init__(
        self, 
        problem: Problem, 
        seed: int | None = None, 
        log: bool = True,
        total: int | None = None
    ) -> None:
        """
        Initialize a results record for a solve of a problem

        Args:
            problem: The problem being solved
            seed: The seed used in the solve, for reproducibility
            log: Whether to log the loss and time of every evaluation
            total: The total number of evaluations for the algorithm and problem
        """
        self._problem = problem
        self._log = log
        self._seed = seed
        self._total = total

         # Initialize the losses for this solve
        self._losses = []
        self._times = []

        # Initialize the loss to ∞, time to 0, and the status and subset to None
        self._evaluations = 0
        self._time = 0.0
        self._loss = np.inf
        self._subset = None
        self._status = None


    def _record(
        self, 
        subset: Subset, 
        loss: float, 
        time: float
    ) -> bool:
        """
        Records an iteration step of the solve in the results

        Args: 
            subset: The subset evaluated at this step
            loss: The loss of the subset evaluated for the problem
            time: The amount of time elapsed to this step in the solve
        
        Raises:
            ValueError: if the loss or time are NaN
        """
        if np.isnan(loss): 
            raise ValueError("Loss is NaN")
        if np.isnan(time): 
            raise ValueError("Time is NaN")

        self._evaluations += 1
        if self._log:
            self._losses.append(loss)
            self._times.append(time)

        # Update the best loss and subset if the loss is lower
        improved = loss < self._loss
        if improved:
            self._loss = loss
            self._subset = subset.freeze()

        self._time = time
        return improved

    def _complete(self, status: str, time: float) -> None:
        """
        Logs the completion of the solve with an updated status

        Args: 
            status: the status of the solve at completion
            time: the time taken for the full completion of the solve
        """
        self._status = status
        self._time = time

    @property
    def subset(self) -> Subset | None:
        """
        The minimum loss subset found up to this point
        """
        return self._subset

    @property
    def losses(self) -> list[float]:
        """
        The loss of every evaluation logged in sequential order
        """
        return list(self._losses)

    @property
    def evaluations(self) -> int:
        """
        The number of evaluations recorded so far, counted whether or not
        they are logged, and cheap enough to read on every callback
        """
        return self._evaluations

    @property
    def times(self) -> list[float]:
        """
        The elapsed time of every evaluation logged in sequential order
        """
        return list(self._times)

    @property
    def loss(self) -> float:
        """
        The best loss evaluated for the best subset found for the problem
        """
        return self._loss

    @property
    def time(self) -> float:
        """
        The amount of time elapsed searching for the best subset so far
        """
        return self._time

    @property
    def status(self) -> str | None:
        """
        The status at completion: finished, stopped, or None while running
        """
        return self._status

    @property
    def total(self) -> int | None:
        """
        The total number of evaluations for the algorithm and problem
        """
        return self._total
