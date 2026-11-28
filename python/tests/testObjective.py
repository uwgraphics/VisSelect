# --- Imports ------------------------------------------------------------------

# standard library
import importlib
import sys
import unittest
from unittest import mock

# third party
import numpy as np
from scipy.stats import wasserstein_distance_nd

# local files
from visselect import Dataset, Subset, metric, objective, transform

# optional dependencies
try:
    import ot
    POT = True
except ImportError:
    POT = False


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

    def testOn(self):
        """Test that a named transform is preserved, not the data"""
        self.dataset["doubled"] = self.dataset["data"] * 2
        preserve = objective.Preserve(metric.mean, on="doubled")
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

    def testEncodeUnsupported(self):
        """Test that encoding a metric without an encoding raises the name"""
        preserve = objective.Preserve(metric.variance)
        with self.assertRaisesRegex(ValueError, "variance"):
            preserve.encode(None, None, self.dataset, 2)

class TestDistinctness(ObjectiveTestCase):
    """
    Test the distinctness objective class
    """

    def testNearestNeighbors(self):
        """Test that each item's distance to its nearest neighbor is summed"""
        dataset = Dataset(np.array([[0.0], [1.0], [3.0]]))
        subset = Subset(dataset, np.array([True, True, True]))
        self.assertEqual(objective.Distinctness()(subset), -(1 + 1 + 2))

    def testLoneItem(self):
        """Test that a single item, with no neighbor, has no distinctness"""
        dataset = Dataset(np.array([[0.0], [1.0], [3.0]]))
        subset = Subset(dataset, np.array([True, False, False]))
        self.assertEqual(objective.Distinctness()(subset), 0.0)



class TestCrossings(unittest.TestCase):
    """
    Test the crossings objective class
    """

    def testCrossing(self):
        """Test that a crossing pair of lines counts at the full rate"""
        dataset = Dataset(np.array([[0.0, 1.0], [1.0, 0.0]]))
        subset = Subset(dataset, np.array([True, True]))
        self.assertEqual(objective.Crossings()(subset), 1.0)

    def testParallel(self):
        """Test that parallel lines and ties do not cross"""
        dataset = Dataset(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]))
        subset = Subset(dataset, np.array([True, True, True]))
        self.assertEqual(objective.Crossings()(subset), 0.0)

    def testShare(self):
        """Test that the rate is the crossing share of all the line pairs"""
        dataset = Dataset(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]))
        subset = Subset(dataset, np.array([True, True, True]))
        self.assertAlmostEqual(objective.Crossings()(subset), 1 / 3)

    def testDegenerate(self):
        """Test that one item or one feature rates no crossings, not NaN"""
        dataset = Dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(
            objective.Crossings()(Subset(dataset, np.array([True, False]))),
            0.0)
        narrow = Dataset(np.array([[1.0], [2.0]]))
        self.assertEqual(
            objective.Crossings()(Subset(narrow, np.array([True, True]))),
            0.0)


class TestOverdraw(unittest.TestCase):
    """
    Test the scatterplot overdraw objective class
    """

    def testValue(self):
        """Test stacked, partly overlapping and separate pairs of marks"""
        data = np.array([[0.0, 0.0], [0.0, 0.0], # stacked: one mark's area
                         [5.0, 0.0], [6.0, 0.0], # distance 1 at radius 1
                         [20.0, 0.0]]) # far from everything: no overlap
        subset = Subset(Dataset(data), np.full(5, True))
        lens = 2 * np.arccos(0.5) - 0.5 * np.sqrt(3.0) # r = 1, d = 1
        self.assertAlmostEqual(objective.Overdraw(radius=1.0)(subset),
                               np.pi + lens)

    def testLone(self):
        """Test that a lone mark has no overdraw"""
        dataset = Dataset(np.array([[0.0, 0.0], [1.0, 1.0]]))
        subset = Subset(dataset, np.array([True, False]))
        self.assertEqual(objective.Overdraw(radius=1.0)(subset), 0.0)

    def testReject(self):
        """Test that a bad radius or a non-2D transform raises a ValueError"""
        with self.assertRaises(ValueError):
            objective.Overdraw(radius=0.0)
        with self.assertRaisesRegex(ValueError, "required"):
            objective.Overdraw(radius=None) # an empty setting from the page
        dataset = Dataset(np.array([[0.0], [1.0]]))
        subset = Subset(dataset, np.array([True, True]))
        with self.assertRaisesRegex(ValueError, "not 2D"):
            objective.Overdraw(radius=1.0)(subset)


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

    def testMatchesDirectFormula(self):
        """Test that the cached between term matches the direct evaluation"""
        from scipy.spatial.distance import cdist
        rng = np.random.default_rng(1)
        data = rng.normal(size=(30, 3))
        dataset = Dataset(data)
        indicator = rng.random(30) < 0.3
        chosen = data[indicator]
        direct = (2 * cdist(chosen, data).mean()
                  - cdist(chosen, chosen).mean())
        energy = objective.EnergyDistance()
        subset = Subset(dataset, indicator)
        self.assertAlmostEqual(energy(subset), direct, places=12)
        self.assertAlmostEqual(energy(subset), direct, places=12) # cached


@unittest.skipUnless(POT, "POT is not installed")
class TestEarthMoversDistance(unittest.TestCase):
    """
    Test the earth mover's distance objective class
    """

    def testValue(self):
        """Test the value on a small case computed by hand"""
        dataset = Dataset(np.array([[0.0], [1.0]]))
        subset = Subset(dataset, np.array([True, False]))
        # half the subset's mass moves a distance of 1 to reach the dataset
        self.assertAlmostEqual(objective.EarthMoversDistance()(subset), 0.5)

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
        earthMovers = objective.EarthMoversDistance()
        self.assertLess(earthMovers(Subset(dataset, both)),
                        earthMovers(Subset(dataset, one)))

    def testGroundMetric(self):
        """Test the Euclidean ground metric against SciPy's exact solver"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(20, 2)))
        subset = Subset(dataset, np.arange(20) < 5)
        self.assertAlmostEqual(
            objective.EarthMoversDistance()(subset),
            wasserstein_distance_nd(subset["data"], dataset["data"]))


class TestPotMissing(unittest.TestCase):
    """
    Test the earth mover's distance objective when POT is not installed
    """

    def testRequiresPot(self):
        """Test that the module and constructor degrade without POT"""
        state = dict(vars(objective)) # the original classes, restored so that
        self.addCleanup(vars(objective).update, state) # identities also hold
        with mock.patch.dict(sys.modules, {"ot": None}):
            importlib.reload(objective) # import falls back to ot = None
            with self.assertRaises(ImportError):
                objective.EarthMoversDistance()


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


class TestDiscreteCoverage(unittest.TestCase):
    """
    Test the discrete coverage objective class
    """

    def testShare(self):
        """Test that the loss is the share of the categories left uncovered"""
        dataset = Dataset(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        coverage = objective.DiscreteCoverage()
        self.assertEqual(
            coverage(Subset(dataset, np.array([True, True, False]))), 0.5)
        self.assertEqual(
            coverage(Subset(dataset, np.array([True, False, True]))), 0.0)

    def testEncoded(self):
        """Test coverage over a discretized and encoded transform"""
        dataset = Dataset(np.array([[0.0], [5.0], [10.0]]))
        root = transform.Transform(dataset["data"])
        dataset["encoded"] = root.discretize(bins=2).encode().array
        coverage = objective.DiscreteCoverage(on="encoded")
        self.assertEqual(
            coverage(Subset(dataset, np.array([True, False, False]))), 0.5)
        self.assertEqual(
            coverage(Subset(dataset, np.array([True, True, False]))), 0.0)

    def testUtopia(self):
        """Test the declared optima: full coverage, or one item per
        category when the size runs short, and zero for the geometry"""
        dataset = Dataset(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        coverage = objective.DiscreteCoverage()
        self.assertEqual(coverage.utopia(dataset, 2), 0.0)
        self.assertEqual(coverage.utopia(dataset, 1), 0.5)
        self.assertEqual(objective.Crossings.utopia, 0.0)
        self.assertEqual(objective.Overdraw.utopia, 0.0)

    def testRejectOrdinal(self):
        """Test that bin numbers, discretized but not encoded, raise"""
        dataset = Dataset(np.array([[0.0], [5.0], [10.0]]))
        root = transform.Transform(dataset["data"])
        dataset["discretized"] = root.discretize(bins=3).array
        coverage = objective.DiscreteCoverage(on="discretized")
        with self.assertRaisesRegex(ValueError, "one hot"):
            coverage(Subset(dataset, np.array([True, True, False])))


if __name__ == "__main__":
    unittest.main()
