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
            emit: The callback which reports each evaluation and stops the run 
                when it returns False
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


class Greedy:
    """
    Selects a subset by swapping out items and keeping those that lower the
    loss, also growing and shrinking the subset when its size is variable
    """

    variableSize = True # also solves problems that leave the size variable

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
        Starts from a random subset and improves it greedily: by swaps
        alone at a fixed size, or by adds, drops and swaps when variable

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run
                when it returns False
        """
        if problem.size is None:
            self._resize(problem, emit)
        else:
            self._sweep(problem, emit)

    def _sweep(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Swaps an item in if it improves the loss, over one pass that tries
        every item outside the starting subset against every member

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run
                when it returns False
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

    def _resize(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Grows, shrinks and swaps a subset from one random starting item,
        keeping the changes that improve the problem's total, with ties
        broken toward lower loss, until a full pass improves nothing

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run
                when it returns False
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        indicator = np.zeros(length, dtype=bool)
        indicator[rng.integers(length)] = True
        subset = Subset(problem.dataset, indicator)

        loss = float(problem.loss(subset))
        key = (problem.evaluate(subset, loss), loss)
        if not emit(subset, key[0]):
            return

        improved = True
        while improved:
            improved = False
            for item in rng.permutation(length):
                if indicator[item]: # a member auditions to leave
                    if np.count_nonzero(indicator) == 1:
                        continue # the last member stays
                    indicator[item] = False
                    dropLoss = float(problem.loss(subset))
                    dropKey = (problem.evaluate(subset, dropLoss), dropLoss)
                    keepGoing = emit(subset, dropKey[0]) # report, then revert
                    indicator[item] = True
                    if not keepGoing:
                        return
                    if dropKey < key:
                        indicator[item] = False
                        loss, key = dropLoss, dropKey
                        improved = True
                    continue

                # an outside item auditions to join alone or by a swap, the
                # best improving try of either kind kept
                bestKey, bestLoss, leaving = key, None, None
                indicator[item] = True
                addLoss = float(problem.loss(subset))
                addKey = (problem.evaluate(subset, addLoss), addLoss)
                keepGoing = emit(subset, addKey[0]) # report, then revert
                indicator[item] = False
                if not keepGoing:
                    return
                if addKey < bestKey:
                    bestKey, bestLoss, leaving = addKey, addLoss, None

                for member in np.flatnonzero(indicator):
                    indicator[member] = False
                    indicator[item] = True
                    swapLoss = float(problem.loss(subset))
                    swapKey = (problem.evaluate(subset, swapLoss), swapLoss)
                    keepGoing = emit(subset, swapKey[0]) # report, then revert
                    indicator[item] = False
                    indicator[member] = True
                    if not keepGoing:
                        return
                    if swapKey < bestKey:
                        bestKey, bestLoss, leaving = swapKey, swapLoss, member

                if bestLoss is not None:
                    indicator[item] = True
                    if leaving is not None:
                        indicator[leaving] = False
                    loss, key = bestLoss, bestKey
                    improved = True

    def evaluations(self, problem: Problem) -> int | None:
        """
        The number of evaluations a fixed size run takes: a random starting
        subset plus one per swap of each item outside the subset, with no
        count for a variable size run, which ends when a pass stops helping

        Args:
            problem: The problem the run would solve
        """
        if problem.size is None:
            return None
        size = problem.size
        return 1 + size * (len(problem.dataset) - size)

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed

class Annealing:
    """
    Selects a subset by simulated annealing: random swaps that may be
    accepted with a chance that shrinks as the temperature runs down
    """

    def __init__(
        self,
        cooling: float = 0.5,
        seed: int | None = None
    ) -> None:
        """
        Initialize a simulated annealing algorithm

        Args:
            cooling: The share of the temperature kept after one neighborhood's 
                worth of swaps
            seed: The random seed for reproducibility

        Raises:
            ValueError: If cooling is not between 0 and 1, exclusive
        """
        if not 0 < cooling < 1:
            raise ValueError("Cooling is not between 0 and 1")
        self._cooling = cooling
        self._seed = seed

    def __call__(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Starts from a random subset and tries random swaps, keeping a worse 
        subset with probability exp(-delta / T) where temperature T cools with 
        every swap by the cooling share over each neighborhood's worth

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run
                when it returns False
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

        # the temperature has run out at a thousandth of where it started
        neighborhood = problem.size * (length - problem.size)
        warmup = 100 # measuring swaps that set the starting temperature
        swaps = warmup + int(np.ceil(
            neighborhood * np.log(1e-3) / np.log(self._cooling)))
        deltas = []

        for swap in range(swaps):
            # try one random swap: a member out and an outside item in
            position = rng.integers(problem.size)
            member = members[position]
            item = rng.choice(np.flatnonzero(~indicator))
            indicator[member] = False
            indicator[item] = True
            swapLoss = float(problem.loss(subset))
            keepGoing = emit(subset, swapLoss) # report before deciding
            delta = swapLoss - loss

            if swap < warmup: # still measuring, never keeping
                deltas.append(abs(delta))
                accept = False
            else:
                if swap == warmup:
                    # start where a typical worse swap is 80% acceptable
                    start = np.mean(deltas) / np.log(1 / 0.8) or 1.0
                temperature = start * self._cooling ** (
                    (swap - warmup) / neighborhood)
                accept = (delta <= 0
                          or rng.random() < np.exp(-delta / temperature))

            if accept: # the swap becomes the new current subset
                members[position] = item
                loss = swapLoss
            else:
                indicator[item] = False
                indicator[member] = True
            if not keepGoing:
                return

    def evaluations(self, problem: Problem) -> int:
        """
        The number of evaluations a run takes: a random starting subset,
        the measuring swaps, and the swaps until the temperature runs out

        Args:
            problem: The problem the run would solve
        """
        neighborhood = problem.size * (len(problem.dataset) - problem.size)
        warmup = 100 # measuring swaps that set the starting temperature
        return 1 + warmup + int(np.ceil(
            neighborhood * np.log(1e-3) / np.log(self._cooling)))

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed

class Genetic:
    """
    Selects a subset by a genetic algorithm: a population of subsets evolves
    by crossing over tournament-selected parents, with children replacing
    the worst subsets as the generations pass
    """

    def __init__(
        self,
        population: int = 30,
        generations: int = 1000,
        seed: int | None = None
    ) -> None:
        """
        Initialize a genetic algorithm

        Args:
            population: The number of subsets evolving together
            generations: The number of generations bred, each one breeding
                a population's worth of children
            seed: The random seed for reproducibility

        Raises:
            ValueError: If population < 2 or generations < 1
        """
        if population < 2:
            raise ValueError("Population is less than 2")
        if generations < 1:
            raise ValueError("Generations is less than 1")
        self._population = population
        self._generations = generations
        self._seed = seed

    def __call__(
        self,
        problem: Problem,
        emit: Callable
    ) -> None:
        """
        Starts from a random population and breeds one child per birth from
        two tournament-chosen parents, keeping the members the parents share
        and filling from the rest of either, with a small chance of one
        member mutating to any item; a child that outranks the worst of the
        population replaces it

        Args:
            problem: The subset selection problem definition to solve
            emit: The callback which reports each evaluation and stops the run
                when it returns False
        """
        rng = np.random.default_rng(self._seed)
        length = len(problem.dataset)
        population = np.stack([
            rng.choice(length, size=problem.size, replace=False)
            for _ in range(self._population)
        ])
        losses = np.empty(self._population)

        indicator = np.zeros(length, dtype=bool)
        indicator[population[0]] = True
        subset = Subset(problem.dataset, indicator)
        for individual, members in enumerate(population):
            indicator[:] = False
            indicator[members] = True
            losses[individual] = float(problem.loss(subset))
            if not emit(subset, losses[individual]):
                return

        for _ in range(self._generations * self._population):
            # each parent wins a tournament between two random individuals
            parents = []
            for _ in range(2):
                a, b = rng.integers(self._population, size=2)
                parents.append(population[a if losses[a] <= losses[b] else b])

            shared = np.intersect1d(parents[0], parents[1])
            pool = np.setxor1d(parents[0], parents[1])
            fill = problem.size - len(shared)
            child = shared if fill == 0 else np.concatenate(
                (shared, rng.choice(pool, size=fill, replace=False)))
            if rng.random() < 0.1:
                outside = np.setdiff1d(np.arange(length), child)
                child[rng.integers(problem.size)] = rng.choice(outside)

            indicator[:] = False
            indicator[child] = True
            childLoss = float(problem.loss(subset))
            keepGoing = emit(subset, childLoss)

            worst = np.argmax(losses)
            if childLoss < losses[worst]: # insert the child by rank,
                population[worst] = child # dropping the worst subset
                losses[worst] = childLoss
            if not keepGoing:
                return

    @property
    def evaluations(self) -> int:
        """
        The number of evaluations a run takes: the starting population plus
        a population's worth of children bred in each generation
        """
        return self._population * (1 + self._generations)

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
            tolerance: The tolerance within which a loss is accepted as optimal
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
        self._model = None
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
        """

        with gp.Model(env=self._environment) as model:
            self._model = model # so stop() can reach the running solve
            try:
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
                if stopped or model.SolCount == 0:
                    return # stopped early: any incumbent is already reported

                indicator = z.X > 0.5 # thresholding for tolerance
            finally:
                self._model = None

        # report the final solution: presolve solves may skip the callback
        subset = Subset(problem.dataset, indicator)
        emit(subset, float(problem.loss(subset)))

    def stop(self) -> None:
        """
        Stop a solve running on another thread at its next opportunity,
        keeping the subsets already reported
        """
        model = self._model
        if model is not None:
            model.terminate() # safe across threads, and after the solve ends

    @property
    def seed(self) -> int | None:
        """
        The random seed for reproducibility
        """
        return self._seed