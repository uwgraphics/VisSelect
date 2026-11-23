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

    def testRepresentation(self):
        """Test that a named representation is preserved, not the data"""
        self.dataset["doubled"] = self.dataset["data"] * 2
        preserve = objective.Preserve(metric.mean, representation="doubled")
        self.assertEqual(preserve(self.subset), 2 * 3 * 1.5)

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

class TestDistinctness(ObjectiveTestCase):
    """
    Test the distinctness objective class
    """

    def testNearestNeighbors(self):
        """Test that each item's distance to its nearest neighbor is summed"""
        dataset = Dataset(np.array([[0.0], [1.0], [3.0]]))
        subset = Subset(dataset, np.array([True, True, True]))
        self.assertEqual(objective.Distinctness()(subset), -(1 + 1 + 2))



class TestEnergyDistance(unittest.TestCase):
    """
    Test the energy distance objective class
    """

    def testValue(self):
        """Test the value on a small case computed by hand"""
        dataset = Dataset(np.array([[0.0], [1.0]]))
        subset = Subset(dataset, np.array([True, False]))
        # twice the mean distance to the dataset less the mean within
        self.assertAlmostEqual(objective.EnergyDistance()(subset), 2 * 0.5 - 0.0)

    def testPrefersTheDistribution(self):
        """Test that a subset spanning both clusters scores lower than one
        drawn from a single cluster"""
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 0.5, size=(100, 2))
        data[50:] += 10.0 # two clusters, half the items in each
        dataset = Dataset(data)
        indices = np.arange(100)
        both = (indices % 25) == 0 # two items in each cluster
        one = indices < 4 # four items in the first cluster
        energy = objective.EnergyDistance()
        self.assertLess(energy(Subset(dataset, both)),
                        energy(Subset(dataset, one)))


class TestOutliers(unittest.TestCase):
    """
    Test the outliers objective class
    """

    def setUp(self):
        """Create a tight cluster of items with one far outlier"""
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 0.5, size=(50, 2))
        data[0] = [10.0, 10.0]
        self.dataset = Dataset(data)
        self.outliers = objective.Outliers(neighbors=10)

    def testPrefersOutliers(self):
        """Test that a subset holding the outlier has the lower loss"""
        withOutlier = np.arange(50) < 5
        without = (np.arange(50) >= 1) & (np.arange(50) < 6)
        self.assertLess(self.outliers(Subset(self.dataset, withOutlier)),
                        self.outliers(Subset(self.dataset, without)))

    def testFactors(self):
        """Test the factors are near 1 for inliers and large for the outlier"""
        factors = self.outliers._factors(self.dataset["data"])
        self.assertGreater(factors[0], 2.0)
        self.assertLess(np.median(factors[1:]), 1.2)
        self.assertGreater(np.median(factors[1:]), 0.8)

    def testCached(self):
        """Test that the factors are computed once and cached on the dataset"""
        subset = Subset(self.dataset, np.arange(50) < 5)
        self.outliers(subset)
        self.assertIn((self.outliers._factors, "data"),
                      self.dataset._characteristics)

    def testRejectNeighbors(self):
        """Test that too few or too many neighbors raise a ValueError"""
        with self.assertRaises(ValueError):
            objective.Outliers(neighbors=0)
        with self.assertRaises(ValueError):
            objective.Outliers(neighbors=50)(
                Subset(self.dataset, np.arange(50) < 5))

class TestClusterCenters(unittest.TestCase):
    """
    Test the cluster centers objective class
    """

    def setUp(self):
        """Create three tight clusters of five items each"""
        rng = np.random.default_rng(0)
        means = [[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]]
        data = np.vstack([rng.normal(mean, 0.1, (5, 2)) for mean in means])
        self.dataset = Dataset(data)
        self.clusters = objective.ClusterCenters(k=3, seed=0)

    def testRepresentatives(self):
        """Test that one item per cluster beats items from one cluster"""
        spread = np.isin(np.arange(15), [0, 5, 10])
        bunched = np.arange(15) < 3
        self.assertLess(self.clusters(Subset(self.dataset, spread)),
                        self.clusters(Subset(self.dataset, bunched)))

    def testSeed(self):
        """Test that a seed reproduces the centers across instances"""
        subset = Subset(self.dataset, np.arange(15) < 5)
        other = objective.ClusterCenters(k=3, seed=0)
        self.assertEqual(self.clusters(subset), other(subset))

    def testRejectK(self):
        """Test that a k below 1 raises a ValueError"""
        with self.assertRaises(ValueError):
            objective.ClusterCenters(k=0)

    def testRejectFew(self):
        """Test that k at or above the number of items raises a ValueError"""
        clusters = objective.ClusterCenters(k=15)
        with self.assertRaises(ValueError):
            clusters(Subset(self.dataset, np.arange(15) < 5))


if __name__ == "__main__":
    unittest.main()
