# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# local files
from visselect import transform


# --- Test Classes -------------------------------------------------------------

class TestScale(unittest.TestCase):
    """
    Test the feature scaling transform
    """

    def testInterval(self):
        """Test that features scale to the interval, others left unchanged"""
        data = np.array([[0.0, 5.0, 7.0], [5.0, 10.0, 8.0], [10.0, 0.0, 9.0]])
        scaled = transform.scale(data, (0.0, 1.0), [0, 1])
        assert_allclose(scaled[:, 0], [0.0, 0.5, 1.0])
        assert_allclose(scaled[:, 1], [0.5, 1.0, 0.0])
        assert_allclose(scaled[:, 2], data[:, 2])

    def testEveryFeature(self):
        """Test that leaving indices out transforms every feature"""
        data = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        assert_allclose(transform.scale(data, (0.0, 1.0)),
                        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert_array_equal(transform.discretize(data, 2),
                           [[0, 0], [1, 1], [1, 1]])
        assert_allclose(transform.encode(np.array([[0, 1], [1, 0]])),
                        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])

    def testConstant(self):
        """Test that a constant feature scales to the interval's low end"""
        data = np.full((4, 1), 3.0)
        assert_allclose(transform.scale(data, (2.0, 8.0), [0]), 2.0)

    def testWholeNumbers(self):
        """Test that a whole number array scales to decimals, not truncated"""
        data = np.array([[0], [1], [2]])
        assert_allclose(transform.scale(data, (0.0, 1.0), [0]),
                        [[0.0], [0.5], [1.0]])


class TestDiscretize(unittest.TestCase):
    """
    Test the discretizing transform
    """

    def testUniform(self):
        """Test that uniform bins divide a feature's range evenly"""
        data = np.array([[0.0], [2.4], [2.6], [7.4], [7.6], [10.0]])
        binned = transform.discretize(data, 4, [0])
        assert_array_equal(binned[:, 0], [0, 0, 1, 2, 3, 3])

    def testQuantile(self):
        """Test that quantile bins hold even shares of the items"""
        data = np.arange(12.0).reshape(-1, 1) ** 2 # a skewed feature
        binned = transform.discretize(data, 3, [0], "quantile")
        assert_array_equal(np.bincount(binned[:, 0]), [4, 4, 4])

    def testTies(self):
        """Test that tied values merge quantile bins instead of emptying one"""
        data = np.concatenate((np.zeros(50), np.arange(1.0, 51))).reshape(-1, 1)
        binned = transform.discretize(data, 4, [0], "quantile")
        assert_array_equal(np.bincount(binned[:, 0]), [50, 25, 25])

    def testClusters(self):
        """Test that cluster bins are ordered along the feature"""
        rng = np.random.default_rng(0)
        data = (rng.normal([0, 10, 20], 0.5, (50, 3))).reshape(-1, 1)
        binned = transform.discretize(data, 3, [0], "kmeans")
        order = np.argsort(data[:, 0])
        self.assertTrue(np.all(np.diff(binned[order, 0]) >= 0))
        assert_array_equal(np.unique(binned), [0, 1, 2])

    def testRejectStrategy(self):
        """Test that an unknown strategy raises a ValueError"""
        with self.assertRaises(ValueError):
            transform.discretize(np.ones((3, 1)), 2, [0], "spectral")


class TestEncode(unittest.TestCase):
    """
    Test the one hot encoding transform
    """

    def testCategories(self):
        """Test that each category becomes a column after the unencoded"""
        data = np.array([[7.0, 0], [8.0, 2], [9.0, 0]])
        encoded = transform.encode(data, [1])
        assert_allclose(encoded, [[7.0, 1, 0], [8.0, 0, 1], [9.0, 1, 0]])

    def testJoint(self):
        """Test that the full order makes one category per combination"""
        data = np.array([[0, 0], [0, 1], [1, 0], [0, 1]])
        encoded = transform.encode(data, [0, 1], order=2)
        assert_allclose(encoded, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

    def testOrder(self):
        """Test that a middle order encodes every pair of features"""
        data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
        encoded = transform.encode(data, order=2)
        # three pairs, each with the combinations that appear as categories
        pairs = [np.unique(data[:, [a, b]], axis=0).shape[0]
                 for a, b in ((0, 1), (0, 2), (1, 2))]
        self.assertEqual(encoded.shape[1], sum(pairs))
        assert_allclose(encoded.sum(axis=1), 3.0) # one category per pair

    def testRejectOrder(self):
        """Test that an order beyond the features raises a ValueError"""
        with self.assertRaises(ValueError):
            transform.encode(np.zeros((2, 2)), order=3)


if __name__ == "__main__":
    unittest.main()
