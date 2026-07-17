# --- Imports ------------------------------------------------------------------

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from visselect import Dataset


# --- Test Class ---------------------------------------------------------------

class TestDatasetConstructor(unittest.TestCase):
    """Test the Dataset class constructor"""

    def testInitNumeric2DArray(self):
        """Test that a numeric 2D Numpy array can be loaded"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        assert_array_equal(dataset._root, data)

    def testInitNonNumeric2DArray(self):
        """Test that a non-numeric 2D Numpy array can be loaded"""
        data = np.array([["Plane", "Bus"], ["Train", "Car"], ["Bike", "Boat"]])
        dataset = Dataset(data)
        assert_array_equal(dataset._root, data)

    def testReject1DArray(self):
        """Test that a 1D Numpy array raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.array([2.0, 7.0, 6.0]))

    def testReject3DArray(self):
        """Test that a 3D Numpy array raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((2, 2, 2)))

    def testRejectNonArray(self):
        """Test that a list raises a TypeError"""
        with self.assertRaises(TypeError):
            Dataset([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0]])
    
    def testRejectNoRows(self):
        """Test that an array with no rows raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((0, 10)))
    
    def testRejectNoColumns(self):
        """Test that an array with no columns raises a ValueError"""
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 0)))

class TestDatasetProperties(unittest.TestCase):
    """Test the Dataset class properties"""

    def testDatasetSize(self):
        """Test the Dataset size property"""
        dataset = Dataset(np.zeros((5, 15)))
        self.assertEqual(dataset.size, (5, 15))

    def testDatasetLength(self):
        """Test the Dataset len dunder property"""
        dataset = Dataset(np.zeros((20, 5)))
        self.assertEqual(len(dataset), 20)

    def testDatasetFeatures(self):
        """Test the Dataset loads features correctly"""
        features = ["a", "b", "c"]
        dataset = Dataset(np.zeros((10, 3)), features)
        self.assertEqual(dataset.features, features)
    
    def testDatasetFeaturesWrongLength(self):
        """Test that a Dataset with wrong feature length raises a ValueError"""
        features = ["a", "b", "c"]
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 10)), features)
    
    def testDatasetFeaturesDuplicates(self):
        """Test that a Dataset with duplicate features raises a ValueError"""
        features = ["a", "a", "b"]
        with self.assertRaises(ValueError):
            Dataset(np.zeros((10, 3)), features)

    def testDatasetFeaturesDefault(self):
        """Test the Dataset loads a default features list correctly"""
        dataset = Dataset(np.zeros((10, 3)))
        self.assertEqual(dataset.features, ["0", "1", "2"])

    def testDatasetFeaturesMutation(self):
        """Test that mutating the list does not change dataset features"""
        features = ["a", "b", "c"]
        dataset = Dataset(np.zeros((10, 3)), features)
        features.append("d")
        self.assertEqual(dataset.features, ["a", "b", "c"])

if __name__ == "__main__":
    unittest.main()
