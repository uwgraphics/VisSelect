# --- Imports ------------------------------------------------------------------

# standard library
import importlib
import itertools
import sys
import threading
import unittest
from unittest import mock

# third party
import numpy as np

# local files
from visselect import (
    Dataset, 
    Loss, 
    Problem, 
    Solver, 
    Subset, 
    algorithm,
    metric, 
    objective, 
    scalarization
)
from visselect.algorithm import GreedySwap, Gurobi, Random

# optional dependencies
try:
    import gurobipy
    GUROBI = True
except ImportError:
    GUROBI = False


# --- Test Classes -------------------------------------------------------------

class AlgorithmTestCase(unittest.TestCase):
    """
    Common base class for algorithm test classes
    """

    def setUp(self):
        """Create a test problem large enough for selected subsets to differ"""
        data = np.arange(60.0).reshape(20, 3)
        self.dataset = Dataset(data)
        self.loss = Loss([objective.Preserve(metric.mean)])
        self.problem = Problem(self.dataset, self.loss, size=5)

class TestRandomSelection(AlgorithmTestCase):
    """
    Test the random selection algorithm
    """

    def testSamples(self):
        """Test that a run reports samples number of subsets of problem size"""
        losses = []
        def check(subset, loss):
            self.assertEqual(len(subset), self.problem.size)
            self.assertEqual(loss, float(self.problem.loss(subset)))
            losses.append(loss)
            return True
        Random(samples=7, seed=0)(self.problem, check)
        self.assertEqual(len(losses), 7)

    def testSamplesBoundary(self):
        """Test that setting samples < 1 raises a ValueError"""
        with self.assertRaises(ValueError):
            Random(samples=0)

    def testSeed(self):
        """Test that a seed reproduces a run and different seeds differ"""
        algorithm = Random(samples=5, seed=3)
        first, repeat, other = [], [], []
        algorithm(self.problem, lambda _, loss: first.append(loss) or True)
        algorithm(self.problem, lambda _, loss: repeat.append(loss) or True)
        Random(samples=5, seed=4)(
            self.problem, lambda _, loss: other.append(loss) or True
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)

    def testDeclaredEvaluations(self):
        """Test that the declared evaluations match a run's actual count"""
        algorithm = Random(samples=7, seed=0)
        losses = []
        algorithm(self.problem, lambda _, loss: losses.append(loss) or True)
        self.assertEqual(algorithm.evaluations, len(losses))

    def testStop(self):
        """Test that the run ends as soon as the callback returns False"""
        losses = []
        Random(samples=20, seed=0)(
            self.problem, lambda _, loss: losses.append(loss) or len(losses) < 3
        )
        self.assertEqual(len(losses), 3)

    def testFreeze(self):
        """Test that the live subset changes and a frozen copy does not"""
        seen = []
        def freezeFirst(subset, loss):
            if not seen:
                seen.append((subset, subset.freeze()))
            return True
        Random(samples=2, seed=0)(self.problem, freezeFirst)
        subset, frozen = seen[0]
        self.assertFalse(np.array_equal(subset._indicator, frozen._indicator))

class TestGreedySwap(AlgorithmTestCase):
    """
    Test the greedy swap algorithm
    """

    def testEvaluations(self):
        """Test that a run tries every swap once, keeping the problem size"""
        losses = []
        def check(subset, loss):
            self.assertEqual(len(subset), self.problem.size)
            self.assertEqual(loss, float(self.problem.loss(subset)))
            losses.append(loss)
            return True
        GreedySwap(seed=0)(self.problem, check)
        self.assertEqual(len(losses), 1 + (20 - 5) * 5)

    def testDeclaredEvaluations(self):
        """Test that the declared evaluations match a run's actual count"""
        algorithm = GreedySwap(seed=0)
        losses = []
        algorithm(self.problem, lambda _, loss: losses.append(loss) or True)
        self.assertEqual(algorithm.evaluations(self.problem), len(losses))

    def testBestSwap(self):
        """Test that trying every swap finds the best size one subset exactly"""
        dataset = Dataset(np.array([[0.0], [1.0], [10.0]]))
        problem = Problem(dataset, self.loss, size=1)
        results = Solver(GreedySwap(seed=0))(problem)
        # the item closest to the mean of 11/3 is item 1, with loss 8/3
        np.testing.assert_array_equal(results.subset.indicator,
                                      [False, True, False])
        self.assertAlmostEqual(results.loss, 8.0 / 3.0)

    def testImproves(self):
        """Test that the result is no worse than the random starting subset"""
        start = []
        Random(seed=0)(self.problem, lambda _, loss: start.append(loss) or True)
        greedy = Solver(GreedySwap(seed=0))(self.problem)
        self.assertLessEqual(greedy._loss, start[0])

    def testSeed(self):
        """Test that a seed reproduces a run"""
        algorithm = GreedySwap(seed=3)
        first, repeat = [], []
        algorithm(self.problem, lambda _, loss: first.append(loss) or True)
        algorithm(self.problem, lambda _, loss: repeat.append(loss) or True)
        self.assertEqual(first, repeat)

    def testStop(self):
        """Test that the run ends as soon as the callback returns False"""
        for limit in (1, 3): # stop on the start subset, then inside a swap
            losses = []
            GreedySwap(seed=0)(
                self.problem,
                lambda _, loss: losses.append(loss) or len(losses) < limit
            )
            self.assertEqual(len(losses), limit)

@unittest.skipUnless(GUROBI, "gurobipy is not installed")
class TestGurobi(unittest.TestCase):
    """
    Test the Gurobi optimization algorithm
    """

    def setUp(self):
        """Create a test problem small enough to brute force every subset"""
        rng = np.random.default_rng(0)
        self.dataset = Dataset(rng.normal(size=(10, 2)))
        self.loss = Loss([objective.Preserve(metric.mean)])
        self.problem = Problem(self.dataset, self.loss, size=3)

    def bruteForce(self, problem):
        """The minimum loss over every possible subset, by exhaustive search"""
        length = len(problem.dataset)
        best = np.inf
        for members in itertools.combinations(range(length), problem.size):
            indicator = np.isin(np.arange(length), members)
            loss = float(problem.loss(Subset(problem.dataset, indicator)))
            best = min(best, loss)
        return best

    def testOptimal(self):
        """Test that the solve finds the exact brute force minimum loss"""
        results = Solver(Gurobi(tolerance=0, seed=0))(self.problem)
        self.assertAlmostEqual(results.loss, self.bruteForce(self.problem))

    def testWeighted(self):
        """Test that weighted objectives solve to the brute force minimum"""
        loss = Loss(
            [objective.Preserve(metric.mean), objective.Preserve(metric.mean)],
            scalarization.WeightedSum([2.0, 0.5])
        )
        problem = Problem(self.dataset, loss, size=3)
        results = Solver(Gurobi(tolerance=0))(problem)
        self.assertAlmostEqual(results.loss, self.bruteForce(problem))

    def testNoEncoding(self):
        """Test that every objective without an encoding is named in the error"""
        loss = Loss([objective.Distinctness(), objective.EnergyDistance()])
        problem = Problem(self.dataset, loss, 3)
        with self.assertRaisesRegex(ValueError, "Distinctness, EnergyDistance"):
            Solver(Gurobi())(problem)

    def testToleranceBoundary(self):
        """Test that a negative tolerance raises a ValueError"""
        with self.assertRaises(ValueError):
            Gurobi(tolerance=-1.0)

    def testStop(self):
        """Test that the solve terminates when the callback returns False"""
        results = Solver(Gurobi(seed=0))(self.problem, callback=lambda _: False)
        self.assertEqual(results._status, "stopped")
        self.assertGreaterEqual(len(results.losses), 1)

    def testStopMidSolve(self):
        """Test that stop interrupts a grinding solve from another thread"""
        rng = np.random.default_rng(0)
        problem = Problem(Dataset(rng.normal(size=(1000, 2))),
                          Loss([objective.Preserve(metric.mean)]), 10)
        gurobi = Gurobi(tolerance=0, seed=0) # grinds its zero proof bound

        started = threading.Event()
        def onStep(results):
            started.set()
            return True # the stop arrives from the side, not the callback
        worker = threading.Thread(target=Solver(gurobi), args=(problem,),
                                  kwargs={"callback": onStep}, daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=10)) # the first incumbent
        gurobi.stop()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        gurobi.stop() # after the solve, with no model, stop is a no-op

    def testStopBeforeSolve(self):
        """Test that a stop before optimization begins is safely ignored:
        gurobi drops a terminate request that arrives before optimize"""
        gurobi = Gurobi(tolerance=0, seed=0)
        mean = objective.Preserve(metric.mean)
        class StopsDuringEncode:
            """Stops the solve while encoding, before optimization begins"""
            def __call__(self, subset):
                return mean(subset)
            def encode(self, model, z, dataset, size):
                gurobi.stop()
                return mean.encode(model, z, dataset, size)
        problem = Problem(self.dataset, Loss([StopsDuringEncode()]), 3)
        results = Solver(gurobi)(problem)
        self.assertEqual(results._status, "finished")
        self.assertAlmostEqual(results.loss, self.bruteForce(problem))

class TestGurobiMissing(unittest.TestCase):
    """
    Test the Gurobi algorithm when gurobipy is not installed
    """

    def testRequiresGurobipy(self):
        """Test that the module and constructor degrade without gurobipy"""
        state = dict(vars(algorithm)) # the original classes, restored so that
        self.addCleanup(vars(algorithm).update, state) # identities also hold
        with mock.patch.dict(sys.modules, {"gurobipy": None}):
            importlib.reload(algorithm) # import falls back to gp = None
            with self.assertRaises(ImportError):
                algorithm.Gurobi()


if __name__ == "__main__":
    unittest.main()
