# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset, Subset, metric, objective


# --- Test Classes -------------------------------------------------------------

class ObjectiveTestCase(unittest.TestCase):
    """
    Common base class for Objective test classes
    """

    def setUp(self):
        """Create test dataset with example data and a test indicator"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.dataset = Dataset(data)
        self.subset = Subset(self.dataset, np.array([False, True, True]))

class TestPreserve(ObjectiveTestCase):
    """
    Test the preserve metric objective class
    """

    def testInitPreserve(self):
        """Test that the objective can be created and dataset values cached"""
        preserve = objective.Preserve(metric.mean, self.dataset)
        assert_array_equal(preserve._target, np.array([4.0, 5.0, 6.0]))

    def testFullSelection(self):
        """Test that a subset of every item has no loss"""
        subset = Subset(self.dataset, np.array([True, True, True]))
        preserve = objective.Preserve(metric.mean, self.dataset)
        self.assertEqual(preserve(subset), 0.0)

    def testManhattan(self):
        """Test that a subset is correctly evaluated with the default L1 norm"""
        preserve = objective.Preserve(metric.mean, self.dataset)
        self.assertEqual(preserve(self.subset), 3*1.5)

    def testEuclidean(self):
        """Test that a subset is correctly evaluated with the L2 norm"""
        preserve = objective.Preserve(metric.mean, self.dataset, p=2)
        self.assertEqual(preserve(self.subset), np.sqrt(3*1.5**2))

    def testScalarMetric(self):
        """Test that a scalar metric is evaluated by absolute difference"""
        preserve = objective.Preserve(np.sum, self.dataset)
        self.assertEqual(preserve(self.subset), 6.0)

if __name__ == "__main__":
    unittest.main()
