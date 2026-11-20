# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable, Iterator

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
        log: bool = True
    ) -> Results:
        """
        Runs its solve algorithm on a problem

        Args: 
            problem: The subset selection problem definition to solve
            log: Whether to log the loss and time of every evaluation
        """
        for results in self.steps(problem, log):
            pass
        return results

    def steps(
        self, 
        problem: Problem, 
        log: bool = True
    ) -> Iterator[Results]:
        """
        Runs the algorithm on a problem, yielding results after each evaluation

        Args:
            problem: The subset selection problem definition to solve
            log: Whether to log the loss and time of every evaluation
        """
        seed = getattr(self._algorithm, "seed", None)
        results = Results(problem, log=log, seed=seed)
        run = self._algorithm(problem)
        timer = Timer()

        try:
            while True:
                with timer:
                    candidate = next(run, None)
                if candidate is None:
                    break
                subset, loss = candidate
                results._record(subset, loss, timer.elapsed)

                yield results
        except GeneratorExit:  # the caller closed the generator early
            results._complete("stopped", timer.elapsed)
            return
        
        results._complete("finished", timer.elapsed)
