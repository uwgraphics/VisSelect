# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Loss, Problem, metric, objective


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


if __name__ == "__main__":
    unittest.main()
