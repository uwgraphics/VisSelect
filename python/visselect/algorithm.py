# --- Imports ------------------------------------------------------------------

# standard library
from typing import Callable

# third party
import numpy as np

try:
    import gurobipy as gp
except ImportError:
    gp = None

# local files
from .problem import Problem
from .subset import Subset


# --- Algorithm Implementations ------------------------------------------------

class Random:
    """
    Selects subsets randomly
    """

    def __init__(
        self, 
        samples: int = 1, 
        seed: int | None = None
    ) -> None:
        """
        Initialize a random selection algorithm

        Args: 
            samples: The number of random samples to take and evaluate
            seed: The random seed for reproducibility

        Raises: 
            ValueError: If the number of samples < 1
        """
        if samples < 1:
            raise ValueError("Samples is less than 1")
        self._samples = samples

        self._seed = seed


    def __call__(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Runs random selection, reporting each random subset as an evaluation

        Args:
            problem: The subset selection problem definition to sample from
            emit: The callback reporting each evaluation, emit(subset, loss);
                stops the run when it returns False
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        indicator = np.zeros(length, dtype=bool)
        indices = np.arange(problem.size)
        indicator[indices] = True
        subset = Subset(problem.dataset, indicator)

        for _ in range(self._samples):
            indicator[indices] = False
            indices = rng.choice(length, size=problem.size, replace=False)
            indicator[indices] = True
            if not emit(subset, float(problem.loss(subset))):
                return

    @property
    def evaluations(self) -> int:
        """
        The number of evaluations a run takes: one per random sample
        """
        return self._samples

    @property
    def seed(self) -> int:
        """
        The random seed for reproducibility
        """
        return self._seed


class GreedySwap:
    """
    Selects a subset by swapping out items and keeping those that lower the loss
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize a greedy swap algorithm

        Args:
            seed: The random seed for reproducibility
        """
        self._seed = seed

    def __call__(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Starts from a random subset and swaps an item in if it improves the loss

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback reporting each evaluation, emit(subset, loss);
                stops the run when it returns False
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        indicator = np.zeros(length, dtype=bool)
        members = rng.choice(length, size=problem.size, replace=False)
        indicator[members] = True
        subset = Subset(problem.dataset, indicator)

        loss = float(problem.loss(subset))
        if not emit(subset, loss):
            return

        for item in rng.permutation(np.flatnonzero(~indicator)):
            bestLoss, bestPosition = loss, None

            # try swapping the item with each member, reverting after each try
            for position, member in enumerate(members):
                indicator[member] = False
                indicator[item] = True
                swapLoss = float(problem.loss(subset))
                keepGoing = emit(subset, swapLoss) # report before reverting
                indicator[item] = False
                indicator[member] = True
                if not keepGoing:
                    return

                if swapLoss < bestLoss:
                    bestLoss, bestPosition = swapLoss, position

            # keep the best swap if it improved on the current subset
            if bestPosition is not None:
                indicator[members[bestPosition]] = False
                indicator[item] = True
                members[bestPosition] = item
                loss = bestLoss

    def evaluations(self, problem: Problem) -> int:
        """
        The number of evaluations a run takes: a random starting subset plus 
        one per swap of each item outside the subset

        Args:
            problem: The problem the run would solve
        """
        size = problem.size
        return 1 + size * (len(problem.dataset) - size)

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed

class Gurobi:
    """
    Selects subsets by applying the Gurobi optimization branch and cut algorithm
    """
    def __init__(
        self,
        tolerance: float = 1e-4,
        seed: int | None = None
    ) -> None:
        """
        Initialize the Gurobi optimization branch and cut algorithm

        Args:
            tolerance: The loss slack: a subset whose loss is within this of
                the best provable loss is accepted as optimal
            seed: The random seed for reproducibility

        Raises:
            ImportError: if gurobipy is not installed
            ValueError: If the tolerance is negative
        """
        if gp is None:
            raise ImportError("Gurobi requires gurobipy")
        if tolerance < 0:
            raise ValueError("Tolerance is negative")

        self._environment = gp.Env(params={"OutputFlag": 0})
        self._tolerance = tolerance
        self._seed = seed

    def __call__(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Runs the Gurobi optimization branch and cut algorithm on the problem,
        reporting each improved incumbent subset found during the solve

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run 
                when it returns False

        Raises:
            RuntimeError: If the algorithm fails to find the subset
        """

        with gp.Model(env=self._environment) as model:
            if self._seed is not None:
                model.Params.Seed = self._seed
            model.Params.MIPGapAbs = self._tolerance

            z = model.addMVar(len(problem.dataset), vtype=gp.GRB.BINARY)
            model.addConstr(z.sum() == problem.size)

            encodedLoss = problem.loss.encode(
                model = model, 
                z = z, 
                dataset = problem.dataset, 
                size = problem.size
            )
            model.setObjective(encodedLoss, gp.GRB.MINIMIZE)

            stopped = False
            def callback(model, where):
                nonlocal stopped
                if where == gp.GRB.Callback.MIPSOL:
                    values = np.asarray(model.cbGetSolution(z))
                    subset = Subset(problem.dataset, values > 0.5)
                    if not emit(subset, float(problem.loss(subset))):
                        stopped = True
                        model.terminate()

            model.optimize(callback)
            if stopped:
                return

            indicator = z.X > 0.5 # thresholding for tolerance

        # report the final solution: presolve solves may skip the callback
        subset = Subset(problem.dataset, indicator)
        emit(subset, float(problem.loss(subset)))

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed