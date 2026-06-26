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

if __name__ == "__main__":
    unittest.main()
