# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import metric


# --- Test Classes -------------------------------------------------------------

class TestRange(unittest.TestCase):
    """
    Test the per feature range metric
    """

    def testRanges(self):
        """Test that each feature's range is its maximum minus its minimum"""
        data = np.array([[0.0, 5.0], [2.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_equal(metric.range(data), [4.0, 4.0])

class TestExtremes(unittest.TestCase):
    """
    Test the per feature maximum and minimum metrics
    """

    def testMax(self):
        """Test that each feature's maximum is found"""
        data = np.array([[0.0, 5.0], [2.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_equal(metric.max(data), [4.0, 5.0])

    def testMin(self):
        """Test that each feature's minimum is found"""
        data = np.array([[0.0, 5.0], [2.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_equal(metric.min(data), [0.0, 1.0])

class TestSum(unittest.TestCase):
    """
    Test the per feature sum metric
    """

    def testSums(self):
        """Test that each feature's values are added up"""
        data = np.array([[0.0, 5.0], [2.0, 1.0], [4.0, 3.0]])
        np.testing.assert_array_equal(metric.sum(data), [6.0, 9.0])

class TestHull(unittest.TestCase):
    """
    Test the convex hull metric
    """

    def testArea(self):
        """Test that the hull of a unit square with an inside point is area 1"""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        self.assertAlmostEqual(metric.hull(square), 1.0)


if __name__ == "__main__":
    unittest.main()
