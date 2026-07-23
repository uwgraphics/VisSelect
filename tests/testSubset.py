# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset
from visselect import Subset


# --- Test Classes -------------------------------------------------------------

class TestSubsetConstructor(unittest.TestCase):
    """Test the Subset class constructor"""

    def testInitSubset(self):
        """Test that a subset can be created from a dataset and an indicator"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        subset = Subset(dataset, np.array([True, False, True]))
        assert_array_equal(subset._indicator, np.array([True, False, True]))

    def testInitSubsetView(self):
        """Test that the subset holds the indicator by reference, not by copy"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        indicator = np.array([True, False, True])
        subset = Subset(dataset, indicator)
        indicator[1] = True
        assert_array_equal(subset._indicator, np.array([True, True, True]))
    
    def testInitSubsetEmpty(self):
        """Test that an all-False indicator constructs an empty subset"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        subset = Subset(dataset, np.array([False, False, False]))
        assert_array_equal(subset._indicator, np.array([False, False, False]))

    def testInitSubsetNonBoolean(self):
        """Test that a non-boolean indicator raises a ValueError"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        with self.assertRaises(ValueError):
            Subset(dataset, np.array([1, 0, 1]))
    
    def testInitSubsetList(self):
        """Test that an list indicator raises a TypeError"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        with self.assertRaises(TypeError):
            Subset(dataset, [True, False, True])

    def testInitSubsetWrongSize(self):
        """Test that a length-mismatched indicator raises a ValueError"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        dataset = Dataset(data)
        with self.assertRaises(ValueError):
            Subset(dataset, np.array([True, False, True, True, False]))

if __name__ == "__main__":
    unittest.main()
