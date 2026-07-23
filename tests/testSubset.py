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

class SubsetTestCase(unittest.TestCase):
    """Common base class for Subset test classes"""

    def setUp(self):
        """Create test dataset with example data and a test indicator"""
        data = np.array([[9.0, 3.0, 5.0], [4.0, 8.0, 7.0], [1.0, 2.0, 6.0]])
        self.dataset = Dataset(data)
        self.indicator = np.array([True, False, True])

class TestSubsetConstructor(SubsetTestCase):
    """Test the Subset class constructor"""

    def testInitSubset(self):
        """Test that a subset can be created from a dataset and an indicator"""
        subset = Subset(self.dataset, self.indicator)
        assert_array_equal(subset._indicator, np.array([True, False, True]))

    def testInitSubsetView(self):
        """Test that the subset holds the indicator by reference, not by copy"""
        subset = Subset(self.dataset, self.indicator)
        self.indicator[1] = True
        assert_array_equal(subset._indicator, np.array([True, True, True]))
    
    def testInitSubsetEmpty(self):
        """Test that an all-False indicator constructs an empty subset"""
        subset = Subset(self.dataset, np.array([False, False, False]))
        assert_array_equal(subset._indicator, np.array([False, False, False]))

    def testInitSubsetNonBoolean(self):
        """Test that a non-boolean indicator raises a ValueError"""
        with self.assertRaises(ValueError):
            Subset(self.dataset, np.array([1, 0, 1]))
    
    def testInitSubsetList(self):
        """Test that a list indicator raises a TypeError"""
        with self.assertRaises(TypeError):
            Subset(self.dataset, [True, False, True])

    def testInitSubsetWrongSize(self):
        """Test that a length-mismatched indicator raises a ValueError"""
        with self.assertRaises(ValueError):
            Subset(self.dataset, np.array([True, False, True, True, False]))

class TestSubsetProperties(SubsetTestCase):
    """Test the Subset class properties"""

    def testSize(self):
        """Test that the Subset size property returns the correct size pair"""
        subset = Subset(self.dataset, self.indicator)
        self.assertEqual(subset.size, (2, 3))

    def testLength(self):
        """Test that the Subset len dunder returns the number of rows"""
        subset = Subset(self.dataset, self.indicator)
        self.assertEqual(len(subset), 2)

    def testData(self):
        """Test that the Subset data returns the selected items"""
        subset = Subset(self.dataset, np.array([True, False, False]))
        assert_array_equal(subset.data, np.array([[9.0, 3.0, 5.0]]))

    def testFreeze(self):
        """Test that a frozen snapshot does not change with the subset"""
        subset = Subset(self.dataset, self.indicator)
        frozen = subset.freeze()
        subset._indicator[1] = True
        assert_array_equal(subset._indicator, np.array([True, True, True]))
        assert_array_equal(frozen._indicator, np.array([True, False, True]))

    def testFreezeImmutable(self):
        """Test that writing to a frozen subset indicator raises a ValueError"""
        frozen = Subset(self.dataset, self.indicator).freeze()
        with self.assertRaises(ValueError):
            frozen._indicator[1] = True

if __name__ == "__main__":
    unittest.main()
