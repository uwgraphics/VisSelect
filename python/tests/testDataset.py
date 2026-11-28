# --- Imports ------------------------------------------------------------------

# standard library
import io
import tempfile
import unittest
from pathlib import Path

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset, metric

# optional dependencies
try:
    import pandas as pd
    PANDAS = True
except ImportError:
    PANDAS = False


# --- Test Classes -------------------------------------------------------------

class TestDatasetConstructor(unittest.TestCase):
    """
    Test the Dataset class constructor
    """

    def testInitNumeric2DArray(self):
        """Test that a numeric 2D Numpy array can be loaded"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        assert_array_equal(dataset["data"], data)

    def testInitNonNumeric2DArray(self):
        """Test that a non-numeric 2D Numpy array can be loaded"""
        data = np.array([["Plane", "Bus"], ["Train", "Car"], ["Bike", "Boat"]])
        dataset = Dataset(data)
        assert_array_equal(dataset["data"], data)

    def testInitListOfLists(self):
        """Test that a numeric list of lists can be loaded"""
        dataset = Dataset([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        array = np.asarray([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        assert_array_equal(dataset["data"], array)

    def testInitStream(self):
        """Test that a csv text stream can be loaded with header features"""
        dataset = Dataset(io.StringIO("a,b\n1,2\n3,4\n"))
        assert_array_equal(dataset["data"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(dataset.features, ["a", "b"])

    def testInitPath(self):
        """Test that a csv file can be loaded from a string or Path filepath"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("a,b\n1,2\n3,4\n")
            for source in (str(path), path):
                dataset = Dataset(source)
                assert_array_equal(dataset["data"], [[1.0, 2.0], [3.0, 4.0]])
                self.assertEqual(dataset.features, ["a", "b"])

    def testInitMixed(self):
        """Test that text columns load as text beside numeric columns"""
        dataset = Dataset(io.StringIO(
            'name,games\n"Defense, Balogh",692\nBrooklyn,228\n'))
        self.assertEqual(dataset.features, ["name", "games"])
        self.assertEqual(dataset["data"].dtype, object)
        self.assertEqual(list(dataset["data"][:, 0]),
                         ["Defense, Balogh", "Brooklyn"])
        self.assertEqual(dataset["data"][0, 1], 692.0)

    def testRejectEmptyFile(self):
        """Test that an empty csv file raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(io.StringIO(""))

    def testRejectRaggedFile(self):
        """Test that a csv file with ragged rows raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(io.StringIO("a,b\n1,2\n3\n"))

    def testMixedColumnStaysText(self):
        """Test that a column of text and numbers loads whole as text"""
        dataset = Dataset(io.StringIO("a,b\n1,x\n2,3\n"))
        self.assertEqual(list(dataset["data"][:, 1]), ["x", "3"])

    def testReject1DArray(self):
        """Test that a 1D Numpy array raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.array([2.0, 7.0, 6.0]))

    def testReject3DArray(self):
        """Test that a 3D Numpy array raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((2, 2, 2)))
    
    def testRejectNoRows(self):
        """Test that an array with no rows raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((0, 10)))
    
    def testRejectNoColumns(self):
        """Test that an array with no columns raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 0)))
    
    def testRejectRaggedList(self):
        """Test that a ragged list of lists raises a TypeError"""
        with self.assertRaises(TypeError):
            Dataset([[1.0, 2.0], [3.0]])

    def testSourceWriteable(self):
        """Test that freezing the dataset does not freeze the supplied array"""
        data = np.zeros((5, 15))
        Dataset(data)
        data[0, 0] = 15
        self.assertEqual(data[0, 0], 15)

class TestDatasetProperties(unittest.TestCase):
    """
    Test the Dataset class properties
    """
    
    def testDataSharesMemory(self):
        """Test the Dataset data accesses shared memory"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        dataset = Dataset(data)
        self.assertTrue(np.shares_memory(dataset["data"], data))

    def testDataImmutable(self):
        """Test writing to the Dataset data raises a ValueError"""
        dataset = Dataset(np.zeros((5, 15)))
        with self.assertRaises(ValueError):
            dataset["data"][0, 0] = 15

    def testSize(self):
        """Test that the Dataset size property returns the correct size pair"""
        dataset = Dataset(np.zeros((5, 15)))
        self.assertEqual(dataset.size, (5, 15))

    def testLength(self):
        """Test that the Dataset len dunder returns the number of rows"""
        dataset = Dataset(np.zeros((20, 5)))
        self.assertEqual(len(dataset), 20)

    def testFeatures(self):
        """Test that the Dataset loads the column features correctly"""
        features = ["a", "b", "c"]
        dataset = Dataset(np.zeros((10, 3)), features)
        self.assertEqual(dataset.features, features)
    
    def testFeaturesWrongLength(self):
        """Test that a Dataset with wrong feature length raises a ValueError"""
        features = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 10)), features)
    
    def testFeaturesDuplicates(self):
        """Test that a Dataset with duplicate features raises a ValueError"""
        features = ["a", "a", "b"]
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 3)), features)

    def testFeaturesDefault(self):
        """Test that Dataset loads a default features list correctly"""
        dataset = Dataset(np.zeros((10, 3)))
        self.assertEqual(dataset.features, ["0", "1", "2"])

    def testFeaturesMutation(self):
        """Test that mutating the list does not change dataset features"""
        features = ["a", "b", "c"]
        dataset = Dataset(np.zeros((10, 3)), features)
        features.append("d")
        self.assertEqual(dataset.features, ["a", "b", "c"])

class TestDatasetCharacteristics(unittest.TestCase):
    """
    Test the Dataset characteristic evaluation and caching
    """

    def setUp(self):
        """Create test dataset with example data"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.dataset = Dataset(data)

    def testEvaluate(self):
        """Test that evaluate returns the metric value for a function"""
        value = self.dataset.evaluate(metric.mean)
        assert_array_equal(value, np.array([4.0, 5.0, 6.0]))

    def testEvaluateCached(self):
        """Test that repeated evaluation returns the same cached value"""
        first = self.dataset.evaluate(metric.mean)
        self.assertIs(self.dataset.evaluate(metric.mean), first)

    def testCharacteristicName(self):
        """Test that an evaluated characteristic is available by name"""
        value = self.dataset.evaluate(metric.mean)
        self.assertIs(self.dataset.mean, value)

    def testCharacteristicData(self):
        """Test that the name skips characteristics of other transforms"""
        self.dataset["doubled"] = self.dataset["data"] * 2
        self.dataset.evaluate(metric.mean, "doubled")
        value = self.dataset.evaluate(metric.mean)
        self.assertIs(self.dataset.mean, value)

    def testCharacteristicCustom(self):
        """Test that a custom metric function is available by name"""
        def total(array):
            return np.sum(array)
        self.dataset.evaluate(total)
        self.assertEqual(self.dataset.total, 45.0)

    def testCharacteristicEmpty(self):
        """Test that an empty characteristic cache raises an AttributeError"""
        with self.assertRaises(AttributeError):
            self.dataset.mean

    def testCharacteristicUnevaluated(self):
        """Test that an unevaluated characteristic raises an AttributeError"""
        self.dataset.evaluate(metric.mean)
        with self.assertRaises(AttributeError):
            self.dataset.variance

    def testUnderscoreGuard(self):
        """Test that an underscore property raises an AttributeError"""
        with self.assertRaises(AttributeError):
            self.dataset._help

class TestDatasetTransforms(unittest.TestCase):
    """
    Test the Dataset transform tree: branches, replacing, and evaluation
    """

    def setUp(self):
        """Create a test dataset with example data"""
        self.data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.dataset = Dataset(self.data)

    def testStore(self):
        """Test that a stored transform is returned by its name"""
        array = np.zeros((3, 5))
        self.dataset["wide"] = array
        self.assertTrue(np.shares_memory(self.dataset["wide"], array))

    def testDefault(self):
        """Test that indexing by data returns the dataset's own data"""
        self.assertTrue(np.shares_memory(self.dataset["data"], self.data))

    def testImmutable(self):
        """Test that a transform is read only but its source stays open"""
        array = np.zeros((3, 2))
        self.dataset["rep"] = array
        with self.assertRaises(ValueError):
            self.dataset["rep"][0, 0] = 1
        array[0, 0] = 1
        self.assertEqual(array[0, 0], 1)

    def testMixedTable(self):
        """Test that a table with identity columns loads, selectable apart"""
        table = np.array([["a", 1.0], ["b", 2.0], ["c", 3.0]], dtype=object)
        dataset = Dataset(table, features=["name", "x"])
        self.assertEqual(dataset.root.select("x").array.dtype, float)
        dataset.root.select("name")
        self.assertEqual(dataset["name"].tolist(), [["a"], ["b"], ["c"]])

    def testRejectRows(self):
        """Test that a row count mismatch raises a ValueError"""
        with self.assertRaises(ValueError):
            self.dataset["bad"] = np.zeros((4, 2))

    def testRejectUnknown(self):
        """Test that an unknown name raises a KeyError"""
        with self.assertRaises(KeyError):
            self.dataset["missing"]

    def testContains(self):
        """Test that stored names are in the dataset and missing are not"""
        self.dataset["rep"] = np.ones((3, 2))
        self.assertIn("rep", self.dataset)
        self.assertIn("data", self.dataset)
        self.assertNotIn("missing", self.dataset)

    def testEvaluateName(self):
        """Test that a characteristic is cached per transform name"""
        self.dataset["doubled"] = self.data * 2
        assert_array_equal(self.dataset.evaluate(metric.mean), [3.0, 4.0])
        assert_array_equal(self.dataset.evaluate(metric.mean, "doubled"),
                           [6.0, 8.0])

    def testStoreDropsLeaves(self):
        """Test that restoring a branch drops only its own characteristics"""
        self.dataset["rep"] = np.ones((3, 2))
        rootMean = self.dataset.evaluate(metric.mean)
        self.dataset.evaluate(metric.mean, "rep")
        self.dataset["rep"] = np.full((3, 2), 2.0)
        assert_array_equal(self.dataset.evaluate(metric.mean, "rep"),
                           [2.0, 2.0])
        self.assertIs(self.dataset.evaluate(metric.mean), rootMean)

    def testDelete(self):
        """Test that deleting a branch drops it and only its characteristics"""
        self.dataset["rep"] = np.ones((3, 2))
        self.dataset.evaluate(metric.mean, "rep")
        rootMean = self.dataset.evaluate(metric.mean)
        del self.dataset["rep"]
        with self.assertRaises(KeyError):
            self.dataset["rep"]
        self.assertIs(self.dataset.evaluate(metric.mean), rootMean)
        self.dataset["rep"] = np.full((3, 2), 2.0) # the old leaves are gone
        assert_array_equal(self.dataset.evaluate(metric.mean, "rep"),
                           [2.0, 2.0])

    def testStoreFells(self):
        """Test that replacing a node fells the branch derived from it"""
        self.dataset.root.discretize(bins=2).encode()
        self.dataset.evaluate(metric.mean, "encoded")
        self.dataset["discretized"] = np.ones((3, 2))
        with self.assertRaises(KeyError):
            self.dataset["encoded"]
        self.assertNotIn((metric.mean, "encoded"),
                         self.dataset._characteristics)

    def testDeleteFells(self):
        """Test that deleting a node also fells everything derived from it"""
        self.dataset.root.discretize(bins=2).encode()
        del self.dataset["discretized"]
        with self.assertRaises(KeyError):
            self.dataset["encoded"]

    def testDeleteData(self):
        """Test that deleting the data name raises a ValueError"""
        with self.assertRaises(ValueError):
            del self.dataset["data"]

    def testDeleteUnknown(self):
        """Test that deleting an unknown name raises a KeyError"""
        with self.assertRaises(KeyError):
            del self.dataset["missing"]

    def testReplace(self):
        """Test that assigning data swaps it and drops the derived tree"""
        self.dataset["rep"] = np.zeros((3, 2))
        self.dataset.evaluate(metric.mean)
        self.dataset["data"] = self.data * 2
        assert_array_equal(self.dataset["data"], self.data * 2)
        with self.assertRaises(KeyError):
            self.dataset["rep"]
        assert_array_equal(self.dataset.evaluate(metric.mean), [6.0, 8.0])

    def testReplaceReject(self):
        """Test that assigning data with different rows raises a ValueError"""
        with self.assertRaises(ValueError):
            self.dataset["data"] = np.zeros((4, 2))

    def testReplaceFeatures(self):
        """Test that feature names survive same width and fell on a change"""
        dataset = Dataset(self.data, features=["a", "b"])
        dataset["data"] = self.data * 2
        self.assertEqual(dataset.features, ["a", "b"])
        dataset["data"] = np.zeros((3, 3))
        self.assertEqual(dataset.features, ["0", "1", "2"])

    def testReplaceNoFeatures(self):
        """Test that a width change works before features ever materialize"""
        dataset = Dataset(self.data)
        dataset["data"] = np.zeros((3, 3))
        self.assertEqual(dataset.features, ["0", "1", "2"])

class Table:
    """
    A minimal table object for testing to_numpy ingestion
    """
    def __init__(self, array, columns=None):
        self._array = np.asarray(array)
        if columns is not None:
            self.columns = columns

    def to_numpy(self):
        return self._array

class TestDatasetTable(unittest.TestCase):
    """
    Test ingestion of table objects using to_numpy for conversion
    """

    def testInitTableNoColumns(self):
        """Test that a table converts and can be loaded with default features"""
        dataset = Dataset(Table([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(dataset.features, ["0", "1"])

    def testInitTableColumns(self):
        """Test that a table converts and columns load as feature names"""
        dataset = Dataset(Table([[1.0, 2.0], [3.0, 4.0]], columns=["x", "y"]))
        assert_array_equal(dataset["data"], np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(dataset.features, ["x", "y"])

    def testRejectTableConversionFailure(self):
        """Test that a failing to_numpy function raises a TypeError"""
        class Broken:
            def to_numpy(self):
                raise RuntimeError("conversion failed")
        with self.assertRaises(TypeError):
            Dataset(Broken())

    def testRejectTable1DResult(self):
        """Test that a table converting to a 1D array raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(Table([1.0, 2.0, 3.0], columns=["col"]))

@unittest.skipUnless(PANDAS, "pandas is not installed")
class TestDatasetDataFrame(unittest.TestCase):
    """
    Test ingestion of pandas DataFrames into Dataset
    """

    def testInitDataFrame(self):
        """Test that a DataFrame converts and columns load as feature names"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        dataset = Dataset(df)
        assert_array_equal(dataset["data"], df.to_numpy())
        self.assertEqual(dataset.features, ["a", "b"])

    def testDataFrameExplicitFeatures(self):
        """Test that a DataFrame loads custom feature names"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        dataset = Dataset(df, features=["x", "y"])
        self.assertEqual(dataset.features, ["x", "y"])

    def testInitDataFrameMixedType(self):
        """Test that a DataFrame with mixed types loads an object array"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        dataset = Dataset(df)
        self.assertEqual(dataset["data"].dtype, np.dtype(object))


if __name__ == "__main__":
    unittest.main()
