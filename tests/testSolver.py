# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Loss, Problem, metric, objective
from visselect.algorithm import Random
from visselect.solver import Solver


# --- Test Classes -------------------------------------------------------------

class TestSolver(unittest.TestCase):
    """
    Test the Solver class
    """

    def setUp(self):
        """Create a test problem large enough for selected subsets to differ"""
        data = np.arange(60.0).reshape(20, 3)
        self.dataset = Dataset(data)
        self.loss = Loss([objective.Preserve(metric.mean)])
        self.problem = Problem(self.dataset, self.loss, size=5)

    def testNotCallable(self):
        """Test that a non-callable algorithm raises a TypeError"""
        with self.assertRaises(TypeError):
            Solver(5)

    def testBest(self):
        """Test that the results keep the lowest loss and its frozen subset"""
        results = Solver(Random(samples=20, seed=0))(self.problem)
        self.assertEqual(len(results.losses), 20)
        self.assertEqual(results._loss, min(results.losses))
        self.assertEqual(results._loss, float(self.problem.loss(results.subset)))

    def testNoLog(self):
        """Test that turning off logging keeps the best loss but no log"""
        solver = Solver(Random(samples=20, seed=0))
        logged = solver(self.problem)
        unlogged = solver(self.problem, log=False)
        self.assertEqual(unlogged.losses, [])
        self.assertEqual(unlogged._loss, logged._loss)


if __name__ == "__main__":
    unittest.main()
