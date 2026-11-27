# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Loss, Problem, Subset, metric, objective


# --- Test Classes -------------------------------------------------------------

class ProblemTestCase(unittest.TestCase):
    """
    Common base class for Problem test classes
    """

    def setUp(self):
        """Create test dataset with example data and a test indicator"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        self.dataset = Dataset(data)
        self.indicator = np.array([True, False, True])
        self.loss = Loss([objective.Preserve(metric.mean)])

class TestProblemConstructor(ProblemTestCase):
    """
    Test the Problem class constructor
    """

    def testSizeBoundaries(self):
        """Test that sizes < 1 and size ≥ dataset length raise a ValueError"""
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, size=0)
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, size=len(self.dataset))
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, size=-1)
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, size=len(self.dataset) + 1)

    def testInitProblem(self):
        """Test that a problem can be created from a dataset, loss, and size"""
        problem = Problem(self.dataset, self.loss, size=1)
        self.assertIs(problem.dataset, self.dataset)
        self.assertIs(problem.loss, self.loss)
        self.assertEqual(problem.size, 1)

    def testOneWay(self):
        """Test that none or several ways of setting the size raise"""
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss)
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, size=1, epsilon=0.5)

    def testEpsilonBoundary(self):
        """Test that a negative epsilon raises a ValueError"""
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, epsilon=-0.1)

    def testSizeWeightBoundary(self):
        """Test that a size weight of zero or below raises a ValueError"""
        with self.assertRaises(ValueError):
            Problem(self.dataset, self.loss, sizeWeight=0.0)

    def testVariableSize(self):
        """Test that size is None when epsilon or sizeWeight sets it"""
        problem = Problem(self.dataset, self.loss, epsilon=0.5)
        self.assertIsNone(problem.size)
        self.assertEqual(problem.epsilon, 0.5)
        self.assertIsNone(problem.sizeWeight)


class TestProblemEvaluate(ProblemTestCase):
    """
    Test the problem's total evaluation and loss bound on subsets
    """

    def testEvaluate(self):
        """Test that the total is the loss when the size is fixed"""
        problem = Problem(self.dataset, self.loss, size=2)
        subset = Subset(self.dataset, self.indicator)
        self.assertEqual(problem.evaluate(subset),
                         float(self.loss(subset)))

    def testEvaluateSizeWeight(self):
        """Test that the total adds the subset size, weighted"""
        problem = Problem(self.dataset, self.loss, sizeWeight=0.25)
        subset = Subset(self.dataset, self.indicator)
        self.assertEqual(problem.evaluate(subset),
                         float(self.loss(subset)) + 2 * 0.25)

    def testEvaluateEpsilon(self):
        """Test that the total is the size once the loss is within epsilon
        and beyond the dataset length before"""
        subset = Subset(self.dataset, self.indicator)
        loss = float(self.loss(subset))
        feasible = Problem(self.dataset, self.loss, epsilon=loss + 1.0)
        self.assertEqual(feasible.evaluate(subset), 2.0)
        infeasible = Problem(self.dataset, self.loss, epsilon=loss / 2)
        self.assertEqual(infeasible.evaluate(subset),
                         len(self.dataset) + loss - loss / 2)

    def testEvaluateGivenLoss(self):
        """Test that a loss already evaluated is used, not reevaluated"""
        problem = Problem(self.dataset, self.loss, size=2)
        subset = Subset(self.dataset, self.indicator)
        self.assertEqual(problem.evaluate(subset, 7.0), 7.0)

    def testFeasible(self):
        """Test that feasibility follows the epsilon bound on the loss"""
        subset = Subset(self.dataset, self.indicator)
        loss = float(self.loss(subset))
        within = Problem(self.dataset, self.loss, epsilon=loss + 1.0)
        beyond = Problem(self.dataset, self.loss, epsilon=loss / 2)
        self.assertTrue(within.feasible(subset))
        self.assertFalse(beyond.feasible(subset))

    def testFeasibleUnbounded(self):
        """Test that every subset is feasible without an epsilon"""
        problem = Problem(self.dataset, self.loss, size=2)
        subset = Subset(self.dataset, self.indicator)
        self.assertTrue(problem.feasible(subset))


if __name__ == "__main__":
    unittest.main()
