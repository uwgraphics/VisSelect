# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Subset, metric, objective


# --- Test Classes -------------------------------------------------------------

class ObjectiveTestCase(unittest.TestCase):
    """
    Common base class for Objective test classes
    """

    def setUp(self):
        """Create test dataset with example data and a test subset"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.dataset = Dataset(data)
        self.subset = Subset(self.dataset, np.array([False, True, True]))

class TestPreserve(ObjectiveTestCase):
    """
    Test the preserve metric objective class
    """

    def testFullSelection(self):
        """Test that a subset of every item has no loss"""
        subset = Subset(self.dataset, np.array([True, True, True]))
        preserve = objective.Preserve(metric.mean)
        self.assertEqual(preserve(subset), 0.0)

    def testManhattan(self):
        """Test that a subset is correctly evaluated with the default L1 norm"""
        preserve = objective.Preserve(metric.mean)
        self.assertEqual(preserve(self.subset), 3*1.5)

    def testEuclidean(self):
        """Test that a subset is correctly evaluated with the L2 norm"""
        preserve = objective.Preserve(metric.mean, p=2)
        self.assertEqual(preserve(self.subset), np.sqrt(3*1.5**2))

    def testScalarMetric(self):
        """Test that a scalar metric is evaluated by absolute difference"""
        preserve = objective.Preserve(np.sum)
        self.assertEqual(preserve(self.subset), 6.0)

    def testReuseAcrossDatasets(self):
        """Test that one objective evaluates against each subset's dataset"""
        preserve = objective.Preserve(metric.mean)
        dataset2 = Dataset(np.zeros((3, 3)))
        subset2 = Subset(dataset2, np.array([False, True, True]))
        self.assertEqual(preserve(self.subset), 4.5)
        self.assertEqual(preserve(subset2), 0.0)

    def testInitNonCallable(self):
        """Test that a non-callable metric raises a TypeError"""
        with self.assertRaises(TypeError):
            objective.Preserve("mean")


if __name__ == "__main__":
    unittest.main()
