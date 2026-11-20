# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import metric


# --- Test Classes -------------------------------------------------------------

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
