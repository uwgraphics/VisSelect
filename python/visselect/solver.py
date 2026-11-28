# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable, Sequence

# third party
import numpy as np

# local files
from .dataset import Dataset
from .loss import Loss
from .problem import Problem
from .results import Results
from .subset import Subset
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
        self._utopias = {}
        self._bounds = {}

    def __call__(
        self,
        problem: Problem,
        log: bool = True,
        callback: Callable | None = None,
        calibrate: bool = True
    ) -> Results:
        """
        Runs its solve algorithm on a problem, calibrating normalization
        bounds first when the loss mixes objectives without any

        Args:
            problem: The subset selection problem definition to solve
            log: Whether to log the loss and time of every evaluation
            callback: Called with the results and returns false to stop early
            calibrate: Whether to calibrate an unbounded loss before solving

        Raises:
            ValueError: If the problem's size is variable and the algorithm
                does not declare variableSize support, or calibration finds
                an objective with no range
        """
        if problem.size is None and not getattr(
            self._algorithm, "variableSize", False
        ):
            name = type(self._algorithm).__name__
            raise ValueError(f"{name} requires a problem with a fixed size")

        loss = problem.loss
        cheap = (
            len(loss) > 1 # a lone solve is only normalized for free
            or getattr(loss.objectives[0], "utopia", None) is not None
            or (loss.objectives[0], problem.dataset, problem.size)
            in self._utopias
        )
        if (calibrate and problem.size is not None and cheap
                and loss.utopia is None):
            utopia, nadir, _ = self.calibrate(
                problem.dataset,
                loss.objectives,
                problem.size,
                callback
            )
            loss = Loss(loss.objectives, loss.scalarization, utopia, nadir)
            problem = Problem(problem.dataset, loss, size=problem.size)

        # with every bound at a declared optimum, a loss of 0 is every
        # goal provably at its best: the solve ends the moment it appears
        optimum = None
        if problem.size is not None and loss.utopia is not None:
            declared = [
                self._declared(objective, problem.dataset, problem.size)
                for objective in loss.objectives
            ]
            if all(value is not None and bound <= value
                   for value, bound in zip(declared, loss.utopia)):
                optimum = 0.0

        evaluations = getattr(self._algorithm, "evaluations", None)
        total = evaluations(problem) if callable(evaluations) else evaluations
        seed = getattr(self._algorithm, "seed", None)
        results = Results(problem, log=log, seed=seed, total=total)

        timer = Timer()
        stopped = False
        optimal = False

        def emit(subset, loss):
            nonlocal stopped, optimal
            timer.stop() # time spent in the callback is not solve time
            results._record(subset, loss, timer.elapsed)
            if optimum is not None and loss <= optimum:
                optimal = True # the declared optimum: the solve is done
            if callback is not None and callback(results) is False:
                stopped = True
            timer.start()
            return not (stopped or optimal)

        with timer:
            self._algorithm(problem, emit)

        results._complete("stopped" if stopped else "finished", timer.elapsed)
        return results

    def calibrate(
        self,
        dataset: Dataset,
        objectives: Sequence[Callable],
        size: int,
        callback: Callable | None = None
    ) -> tuple[list[float], list[float], list[Results]]:
        """
        Measure utopia-nadir bounds for the objectives

        Args:
            dataset: The dataset to apply subset selection to
            objectives: The objectives to bound, in the order their
                bounds are returned
            size: The fixed subset size the bounds are measured at
            callback: Called with the running results of each solve and
                returns false to stop that solve early

        Raises:
            ValueError: If an objective's payoff is constant
        """
        if len(objectives) == 1: # no payoff table: random subsets instead
            objective = objectives[0]
            declared = self._declared(objective, dataset, size)
            key = (objective, dataset, size)
            if declared is not None:
                results = []
                utopia = np.array([declared])
            elif key in self._utopias: # measured by an earlier calibration
                results = []
                utopia = np.array([self._utopias[key]])
            else:
                results = [self(
                    Problem(dataset, Loss([objective]), size=size),
                    log=False,
                    callback=callback,
                    calibrate=False
                )]
                utopia = np.array([objective(results[0].subset)])
                if results[0].status == "finished": # a stop cuts it short
                    self._utopias[key] = float(utopia[0])
            rng = np.random.default_rng(
                getattr(self._algorithm, "seed", None))
            nadir = np.array([np.mean([
                objective(self._sample(dataset, size, rng))
                for _ in range(100)
            ])])
        else:
            key = (tuple(objectives), dataset, size)
            if key in self._bounds: # calibrated before: nothing to solve
                utopia, nadir = self._bounds[key]
                return list(utopia), list(nadir), []

            probe = Loss(list(objectives)) # evaluates every objective at once
            results = [
                self(Problem(dataset, Loss([objective]), size=size),
                     log=False, callback=callback, calibrate=False)
                for objective in objectives
            ]
            payoff = np.array(
                [probe.vector(result.subset) for result in results]
            )

            utopia = payoff.min(axis=0)
            nadir = payoff.max(axis=0)
            finished = all(result.status == "finished" for result in results)
            for index, objective in enumerate(objectives):
                declared = self._declared(objective, dataset, size)
                if declared is not None: # the exact optimum beats a measure
                    utopia[index] = declared
                    continue
                key = (objective, dataset, size)
                if key in self._utopias: # the best measurement wins
                    utopia[index] = min(utopia[index], self._utopias[key])
                if finished: # a stopped solve is not worth remembering
                    self._utopias[key] = float(utopia[index])
        for index, objective in enumerate(objectives):
            if nadir[index] > utopia[index]:
                continue
            if self._declared(objective, dataset, size) is not None:
                # a declared goal met everywhere measured is satisfied:
                # a unit range keeps violations repelling, without error
                nadir[index] = utopia[index] + 1
                continue
            name = getattr(
                objective,
                "__name__",
                type(objective).__name__
            )
            raise ValueError(f"{name} has no range over the payoff table")
        if len(objectives) > 1 and finished: # the whole table is reusable
            self._bounds[tuple(objectives), dataset, size] = (
                utopia.tolist(), nadir.tolist())
        return utopia.tolist(), nadir.tolist(), results

    @property
    def algorithm(self) -> Callable:
        """
        The solve algorithm the solver runs
        """
        return self._algorithm

    def _declared(self, objective: Callable, dataset: Dataset, size: int):
        """
        The utopia an objective declares, as a value or computed, or None

        Args:
            objective: The objective asked for its utopia
            dataset: The dataset selected from
            size: The fixed subset size
        """
        declared = getattr(objective, "utopia", None)
        if callable(declared):
            return declared(dataset, size)
        return declared

    def _sample(self, dataset: Dataset, size: int, rng) -> Subset:
        """
        A uniform random subset of the size, for the random baseline

        Args:
            dataset: The dataset to draw the subset from
            size: The number of items to draw
            rng: The random generator to draw with
        """
        indicator = np.zeros(len(dataset), dtype=bool)
        indicator[rng.choice(len(dataset), size, replace=False)] = True
        return Subset(dataset, indicator)
