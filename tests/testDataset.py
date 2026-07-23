# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset

# optional dependencies
try:
    import pandas as pd
    PANDAS = True
except ImportError:
    PANDAS = False


# --- Test Classes -------------------------------------------------------------

class TestDatasetConstructor(unittest.TestCase):
    """Test the Dataset class constructor"""

    def testInitNumeric2DArray(self):
        """Test that a numeric 2D Numpy array can be loaded"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        assert_array_equal(dataset.data, data)

    def testInitNonNumeric2DArray(self):
        """Test that a non-numeric 2D Numpy array can be loaded"""
        data = np.array([["Plane", "Bus"], ["Train", "Car"], ["Bike", "Boat"]])
        dataset = Dataset(data)
        assert_array_equal(dataset.data, data)

    def testInitListOfLists(self):
        """Test that a numeric list of lists can be loaded"""
        dataset = Dataset([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        array = np.asarray([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        assert_array_equal(dataset.data, array)

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
    """Test the Dataset class properties"""
    
    def testDataSharesMemory(self):
        """Test the Dataset data accesses shared memory"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
        dataset = Dataset(data)
        self.assertTrue(np.shares_memory(dataset.data, data))

    def testDataImmutable(self):
        """Test writing to the Dataset data raises a ValueError"""
        dataset = Dataset(np.zeros((5, 15)))
        with self.assertRaises(ValueError):
            dataset.data[0, 0] = 15

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

class Table:
    """A minimal table object for testing to_numpy ingestion"""
    def __init__(self, array, columns=None):
        self._array = np.asarray(array)
        if columns is not None:
            self.columns = columns

    def to_numpy(self):
        return self._array

class TestDatasetTable(unittest.TestCase):
    """Test ingestion of table objects using to_numpy for conversion"""

    def testInitTableNoColumns(self):
        """Test that a table converts and can be loaded with default features"""
        dataset = Dataset(Table([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(dataset.features, ["0", "1"])

    def testInitTableColumns(self):
        """Test that a table converts and columns load as feature names"""
        dataset = Dataset(Table([[1.0, 2.0], [3.0, 4.0]], columns=["x", "y"]))
        assert_array_equal(dataset.data, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
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
    """Test ingestion of pandas DataFrames into Dataset"""

    def testInitDataFrame(self):
        """Test that a DataFrame converts and columns load as feature names"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        dataset = Dataset(df)
        assert_array_equal(dataset.data, df.to_numpy())
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
        self.assertEqual(dataset.data.dtype, np.dtype(object))

if __name__ == "__main__":
    unittest.main()
