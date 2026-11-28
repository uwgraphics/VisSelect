# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# local files
from visselect.transform import Transform


# --- Test Classes -------------------------------------------------------------

class TestTransform(unittest.TestCase):
    """
    Test the transform node class
    """

    def testTree(self):
        """Test that a derived child is wired to its parent with provenance"""
        root = Transform(np.zeros((3, 2)))
        child = root.discretize(bins=2)
        self.assertIs(child.parent, root)
        self.assertEqual(root.children, [child])
        self.assertEqual(child.name, "discretized")
        self.assertEqual(child.transform, "discretize")
        self.assertEqual(child.settings["bins"], 2)

    def testRejectRows(self):
        """Test that a child with a different row count raises a ValueError"""
        root = Transform(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            Transform(np.zeros((2, 2)), parent=root)

    def testRejectDuplicateName(self):
        """Test that a name already anywhere in the tree raises a ValueError"""
        root = Transform(np.zeros((3, 2)))
        root.scale().discretize(bins=2) # two levels down from the root
        with self.assertRaisesRegex(ValueError, "discretized"):
            root.discretize(bins=2) # in another branch of the same tree

    def testContiguous(self):
        """Test that a column-major array, as pandas hands tables over, is
        stored with its rows together, which evaluations gather"""
        table = np.asfortranarray(np.zeros((3, 2)))
        self.assertTrue(Transform(table).array.flags["C_CONTIGUOUS"])
        root = Transform(np.asfortranarray(np.arange(6.0).reshape(3, 2)))
        self.assertTrue(root.select([1, 0]).array.flags["C_CONTIGUOUS"])

    def testFeatures(self):
        """Test that features are numbered by default and must fit the array"""
        self.assertEqual(Transform(np.zeros((3, 2))).features, ["0", "1"])
        with self.assertRaises(ValueError):
            Transform(np.zeros((3, 2)), features=["only"])

    def testFeaturesCarried(self):
        """Test that scaling and discretizing keep the names of the features"""
        root = Transform(np.arange(6.0).reshape(3, 2), features=["a", "b"])
        self.assertEqual(root.scale().features, ["a", "b"])
        self.assertEqual(root.discretize(bins=2).features, ["a", "b"])


class TestSelect(unittest.TestCase):
    """
    Test the feature selecting transform
    """

    def setUp(self):
        """Create a root of three named features"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.root = Transform(data, features=["a", "b", "c"])

    def testByName(self):
        """Test that one feature is kept by name, and names its node"""
        chosen = self.root.select("b")
        assert_array_equal(chosen.array, [[2.0], [5.0]])
        self.assertEqual(chosen.name, "b")
        self.assertEqual(chosen.features, ["b"])

    def testByIndex(self):
        """Test that features are kept by index, in the order given"""
        chosen = self.root.select([2, 0])
        assert_array_equal(chosen.array, [[3.0, 1.0], [6.0, 4.0]])
        self.assertEqual(chosen.name, "selected")
        self.assertEqual(chosen.features, ["c", "a"])
        self.assertEqual(chosen.settings, {"features": ["c", "a"]})

    def testName(self):
        """Test that a name given replaces the default"""
        self.assertEqual(self.root.select(["a", "c"], name="ends").name, "ends")

    def testBelowRoot(self):
        """Test that a feature is selected by name from a derived node"""
        chosen = self.root.scale().select("c")
        assert_array_equal(chosen.array, [[0.0], [1.0]])

    def testMixed(self):
        """Test that numeric features selected out of a mixed table become
        a numeric array, while identity features stay as they are"""
        table = np.array([["MSN-ORD", 1.0], ["EWR-MSN", 2.0]], dtype=object)
        root = Transform(table, features=["route", "flights"])
        self.assertEqual(root.select("flights").array.dtype, float)
        self.assertEqual(root.select("route").array.dtype, object)
        self.assertEqual(root.select(["route", "flights"]).array.dtype,
                         object)

    def testRejectNone(self):
        """Test that choosing no features at all raises a ValueError"""
        with self.assertRaisesRegex(ValueError, "No features"):
            self.root.select([])

    def testRejectUnknown(self):
        """Test that an unknown name or index raises a ValueError"""
        with self.assertRaises(ValueError):
            self.root.select("d")
        with self.assertRaises(ValueError):
            self.root.select(3)


class TestScale(unittest.TestCase):
    """
    Test the feature scaling transform
    """

    def testInterval(self):
        """Test that features scale to the interval, others left unchanged"""
        data = np.array([[0.0, 5.0, 7.0], [5.0, 10.0, 8.0], [10.0, 0.0, 9.0]])
        scaled = Transform(data).scale((0.0, 1.0), [0, 1]).array
        assert_allclose(scaled[:, 0], [0.0, 0.5, 1.0])
        assert_allclose(scaled[:, 1], [0.5, 1.0, 0.0])
        assert_allclose(scaled[:, 2], data[:, 2])

    def testEveryFeature(self):
        """Test that leaving indices out transforms every feature"""
        data = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        assert_allclose(Transform(data).scale((0.0, 1.0)).array,
                        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        assert_array_equal(Transform(data).discretize(2).array,
                           [[0, 0], [1, 1], [1, 1]])
        assert_allclose(Transform(np.array([[0, 1], [1, 0]])).encode().array,
                        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])

    def testConstant(self):
        """Test that a constant feature scales to the interval's low end"""
        data = np.full((4, 1), 3.0)
        assert_allclose(Transform(data).scale((2.0, 8.0), [0]).array, 2.0)

    def testWholeNumbers(self):
        """Test that a whole number array scales to decimals, not truncated"""
        data = np.array([[0], [1], [2]])
        assert_allclose(Transform(data).scale((0.0, 1.0), [0]).array,
                        [[0.0], [0.5], [1.0]])


class TestDiscretize(unittest.TestCase):
    """
    Test the discretizing transform
    """

    def testUniform(self):
        """Test that uniform bins divide a feature's range evenly"""
        data = np.array([[0.0], [2.4], [2.6], [7.4], [7.6], [10.0]])
        binned = Transform(data).discretize(4, [0]).array
        assert_array_equal(binned[:, 0], [0, 0, 1, 2, 3, 3])

    def testQuantile(self):
        """Test that quantile bins hold even shares of the items"""
        data = np.arange(12.0).reshape(-1, 1) ** 2 # a skewed feature
        binned = Transform(data).discretize(3, [0], "quantile").array
        assert_array_equal(np.bincount(binned[:, 0].astype(int)), [4, 4, 4])

    def testTies(self):
        """Test that tied values merge quantile bins instead of emptying one"""
        data = np.concatenate((np.zeros(50), np.arange(1.0, 51))).reshape(-1, 1)
        binned = Transform(data).discretize(4, [0], "quantile").array
        assert_array_equal(np.bincount(binned[:, 0].astype(int)),
                           [50, 25, 25])

    def testClusters(self):
        """Test that cluster bins are ordered along the feature"""
        rng = np.random.default_rng(0)
        data = (rng.normal([0, 10, 20], 0.5, (50, 3))).reshape(-1, 1)
        binned = Transform(data).discretize(3, [0], "kmeans").array
        order = np.argsort(data[:, 0])
        self.assertTrue(np.all(np.diff(binned[order, 0]) >= 0))
        assert_array_equal(np.unique(binned), [0, 1, 2])

    def testKeepsOthers(self):
        """Test that features not discretized are left unchanged in place"""
        data = np.array([[0.0, 7.5], [5.0, 8.5], [10.0, 9.5]])
        binned = Transform(data).discretize(2, [0]).array
        assert_array_equal(binned, [[0, 7.5], [1, 8.5], [1, 9.5]])

    def testRejectStrategy(self):
        """Test that an unknown strategy raises a ValueError"""
        with self.assertRaises(ValueError):
            Transform(np.ones((3, 1))).discretize(2, [0], "spectral")


class TestEncode(unittest.TestCase):
    """
    Test the one hot encoding transform
    """

    def testCategories(self):
        """Test that each category becomes a column after the unencoded"""
        data = np.array([[7.0, 0], [8.0, 2], [9.0, 0]])
        encoded = Transform(data).encode([1]).array
        assert_allclose(encoded, [[7.0, 1, 0], [8.0, 0, 1], [9.0, 1, 0]])

    def testJoint(self):
        """Test that the full order makes one category per combination"""
        data = np.array([[0, 0], [0, 1], [1, 0], [0, 1]])
        encoded = Transform(data).encode([0, 1], order=2).array
        assert_allclose(encoded, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])

    def testOrder(self):
        """Test that a middle order encodes every pair of features"""
        data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
        encoded = Transform(data).encode(order=2).array
        # three pairs, each with the combinations that appear as categories
        pairs = [np.unique(data[:, [a, b]], axis=0).shape[0]
                 for a, b in ((0, 1), (0, 2), (1, 2))]
        self.assertEqual(encoded.shape[1], sum(pairs))
        assert_allclose(encoded.sum(axis=1), 3.0) # one category per pair

    def testFeatures(self):
        """Test that each category's column is named for its feature"""
        root = Transform(np.array([[7.0, 0], [8.0, 2], [9.0, 0]]),
                         features=["kept", "bin"])
        self.assertEqual(root.encode([1]).features,
                         ["kept", "bin=0", "bin=2"])
        joint = Transform(np.array([[0, 0], [0, 1]]), features=["x", "y"])
        self.assertEqual(joint.encode(order=2).features, ["x+y=0", "x+y=1"])

    def testRejectText(self):
        """Test that scaling or discretizing text raises a ValueError"""
        node = Transform(np.array([["a"], ["b"]], dtype=object))
        with self.assertRaisesRegex(ValueError, "not numeric"):
            node.scale()
        with self.assertRaisesRegex(ValueError, "not numeric"):
            node.discretize()

    def testStrings(self):
        """Test that text categories encode with their values as names"""
        data = np.array([["white"], ["black"], ["white"]], dtype=object)
        encoded = Transform(data, features=["colour"]).encode()
        self.assertEqual(encoded.features, ["colour=black", "colour=white"])
        self.assertEqual(encoded.array.dtype, np.float64)
        assert_allclose(encoded.array, [[0, 1], [1, 0], [0, 1]])

        # a text feature kept beside the encoding keeps the table mixed
        table = np.array([["k", "u"], ["l", "v"]], dtype=object)
        kept = Transform(table, features=["keep", "cat"]).encode([1])
        self.assertEqual(kept.array.dtype, object)

    def testJointStrings(self):
        """Test that text features encode jointly through their codes"""
        data = np.array([["a", "x"], ["a", "y"], ["b", "x"]], dtype=object)
        encoded = Transform(data, features=["s", "t"]).encode(order=2)
        self.assertEqual(encoded.array.shape, (3, 3))
        assert_allclose(encoded.array.sum(axis=1), 1.0)

    def testRejectOrder(self):
        """Test that an order beyond the features raises a ValueError"""
        with self.assertRaises(ValueError):
            Transform(np.zeros((2, 2))).encode(order=3)


if __name__ == "__main__":
    unittest.main()
