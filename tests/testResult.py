# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset, Loss, Problem, Subset, metric, objective
from visselect.results import Results


# --- Test Classes -------------------------------------------------------------

class TestResults(unittest.TestCase):
    """
    Test the Results class
    """

    def setUp(self):
        """Create a test problem and a live subset to record"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        self.dataset = Dataset(data)
        loss = Loss([objective.Preserve(metric.mean)])
        self.problem = Problem(self.dataset, loss, size=2)
        self.indicator = np.array([True, True, False])
        self.subset = Subset(self.dataset, self.indicator)

    def testRecord(self):
        """Test that recording keeps only a strictly lower loss and its subset"""
        results = Results(self.problem)
        self.assertTrue(results._record(self.subset, 2.0, 0.1))
        self.assertFalse(results._record(self.subset, 2.0, 0.2))
        self.indicator[:] = [False, True, True]
        self.assertTrue(results._record(self.subset, 1.0, 0.3))
        self.indicator[:] = [True, False, True]
        self.assertFalse(results._record(self.subset, 3.0, 0.4))
        self.assertEqual(results.losses, [2.0, 2.0, 1.0, 3.0])
        assert_array_equal(results.subset._indicator, [False, True, True])

    def testNoLog(self):
        """Test that turning off logging keeps the best subset but no log"""
        results = Results(self.problem, log=False)
        results._record(self.subset, 2.0, 0.1)
        self.assertEqual(results.losses, [])
        assert_array_equal(results.subset._indicator, [True, True, False])

    def testNaN(self):
        """Test that recording a NaN loss or time raises a ValueError"""
        with self.assertRaises(ValueError):
            Results(self.problem)._record(self.subset, np.nan, 0.1)
        with self.assertRaises(ValueError):
            Results(self.problem)._record(self.subset, 2.0, np.nan)

    def testComplete(self):
        """Test that completing a solve sets its status and final time"""
        results = Results(self.problem)
        results._record(self.subset, 2.0, 0.1)
        results._complete("finished", 0.5)
        self.assertEqual(results._status, "finished")
        self.assertEqual(results._time, 0.5)


if __name__ == "__main__":
    unittest.main()
