# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Loss, Problem, Solver, metric, objective
from visselect.algorithm import GreedySwap, Random


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
        """Test that a run yields samples number of subsets of problem size"""
        count = 0
        for subset, loss in Random(samples=7, seed=0)(self.problem):
            self.assertEqual(len(subset), self.problem.size)
            self.assertEqual(loss, float(self.problem.loss(subset)))
            count += 1
        self.assertEqual(count, 7)

    def testSamplesBoundary(self):
        """Test that setting samples < 1 raises a ValueError"""
        with self.assertRaises(ValueError):
            Random(samples=0)

    def testSeed(self):
        """Test that a seed reproduces a run and different seeds differ"""
        algorithm = Random(samples=5, seed=3)
        first = [loss for _, loss in algorithm(self.problem)]
        repeat = [loss for _, loss in algorithm(self.problem)]
        other = [loss for _, loss in Random(samples=5, seed=4)(self.problem)]
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)

    def testFreeze(self):
        """Test that the live subset changes and a frozen copy does not"""
        randomSelectionRun = Random(samples=2, seed=0)(self.problem)
        subset, _ = next(randomSelectionRun)
        frozen = subset.freeze()
        next(randomSelectionRun)
        self.assertFalse(np.array_equal(subset._indicator, frozen._indicator))

class TestGreedySwap(AlgorithmTestCase):
    """
    Test the greedy swap algorithm
    """

    def testEvaluations(self):
        """Test that a run tries every swap once, keeping the problem size"""
        count = 0
        for subset, loss in GreedySwap(seed=0)(self.problem):
            self.assertEqual(len(subset), self.problem.size)
            self.assertEqual(loss, float(self.problem.loss(subset)))
            count += 1
        self.assertEqual(count, 1 + (20 - 5) * 5)

    def testImproves(self):
        """Test that the result is no worse than the random starting subset"""
        start = next(iter(Random(seed=0)(self.problem)))[1]
        greedy = Solver(GreedySwap(seed=0))(self.problem)
        self.assertLessEqual(greedy._loss, start)

    def testSeed(self):
        """Test that a seed reproduces a run"""
        algorithm = GreedySwap(seed=3)
        first = [loss for _, loss in algorithm(self.problem)]
        repeat = [loss for _, loss in algorithm(self.problem)]
        self.assertEqual(first, repeat)


if __name__ == "__main__":
    unittest.main()
