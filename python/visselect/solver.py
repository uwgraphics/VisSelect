# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# local files
from .problem import Problem
from .results import Results
from .timer import Timer

# --- Solver Class -------------------------------------------------------------

class Solver:
    """
    An optimization solver that solves for a subset with minimized loss
    """

    def __init__(self, algorithm: Callable):
        """
        Initialize a solver defined by a solve algorithm

        Args:
            algorithm: The function to call to run the solve algorithm

        Raises:
            TypeError: If algorithm is not callable
        """
        if not callable(algorithm):
            raise TypeError("Algorithm is not callable")

        self._algorithm = algorithm

    def __call__(
        self,
        problem: Problem,
        log: bool = True,
        callback: Callable | None = None
    ) -> Results:
        """
        Runs its solve algorithm on a problem

        Args:
            problem: The subset selection problem definition to solve
            log: Whether to log the loss and time of every evaluation
            callback: Called with the results and returns false to stop early
        """
        evaluations = getattr(self._algorithm, "evaluations", None)
        total = evaluations(problem) if callable(evaluations) else evaluations
        seed = getattr(self._algorithm, "seed", None)
        results = Results(problem, log=log, seed=seed, total=total)

        timer = Timer()
        stopped = False

        def emit(subset, loss):
            nonlocal stopped
            timer.stop() # time spent in the callback is not solve time
            results._record(subset, loss, timer.elapsed)
            if callback is not None and callback(results) is False:
                stopped = True
            timer.start()
            return not stopped

        with timer:
            self._algorithm(problem, emit)

        results._complete("stopped" if stopped else "finished", timer.elapsed)
        return results
