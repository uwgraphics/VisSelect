# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import scalarization


# --- Test Classes -------------------------------------------------------------

class ScalarizationTestCase(unittest.TestCase):
    """
    Common base class for Scalarization test classes
    """

    def setUp(self):
        """Create test dataset with example data and a test indicator"""
        self.vector = np.array([1.0, 2.0, 3.0])

class TestWeightedSum(ScalarizationTestCase):
    """
    Test the weighted sum scalarization class 
    """

    def testNegativeWeight(self):
        """Test that a negative weight produces a ValueError"""
        with self.assertRaises(ValueError):
            scalarization.WeightedSum([-2.0])

    def testZeroWeight(self):
        """Test that a 0 weight produces a ValueError"""
        with self.assertRaises(ValueError):
            scalarization.WeightedSum([0])

    def testDefaultWeights(self):
        """Test that calling scalarization with no weight produces the sum"""
        linear = scalarization.WeightedSum()
        self.assertEqual(linear(self.vector), 6.0)

    def testWeights(self):
        """Test the weighted sum scalarization weights property"""
        linear = scalarization.WeightedSum([3.0, 2.0, 1.0])
        assert_array_equal(linear.weights, [3.0, 2.0, 1.0])

    def testCall(self):
        """Test that calling scalarization produces the correct value"""
        linear = scalarization.WeightedSum([3.0, 2.0, 1.0])
        self.assertEqual(linear(self.vector), 10.0)

    def testDifferentLengths(self):
        """Test weights and vectors of differing lengths produce a ValueError"""
        linear = scalarization.WeightedSum([1.0])
        with self.assertRaises(ValueError):
            linear(self.vector)

    def testEncodeDefaultWeights(self):
        """Test that encoding with no weights sums the expressions"""
        linear = scalarization.WeightedSum()
        self.assertEqual(linear.encode(None, [2.0, 3.0]), 5.0)

    def testEncodeDifferentLengths(self):
        """Test weights and expressions of differing lengths raise encoding"""
        linear = scalarization.WeightedSum([1.0])
        with self.assertRaises(ValueError):
            linear.encode(None, [2.0, 3.0])

class TestChebyshev(ScalarizationTestCase):
    """
    Test the Chebyshev scalarization class
    """

    def testNegativeWeight(self):
        """Test that a negative weight produces a ValueError"""
        with self.assertRaises(ValueError):
            scalarization.Chebyshev([-2.0])

    def testDefaultWeights(self):
        """Test that calling with no weights produces the largest objective"""
        largest = scalarization.Chebyshev()
        self.assertEqual(largest(self.vector), 3.0)

    def testCall(self):
        """Test that the largest weighted objective is the value"""
        largest = scalarization.Chebyshev([3.0, 2.0, 1.0])
        self.assertEqual(largest(self.vector), 4.0)

    def testWeights(self):
        """Test the Chebyshev scalarization weights property"""
        largest = scalarization.Chebyshev([3.0, 2.0, 1.0])
        assert_array_equal(largest.weights, [3.0, 2.0, 1.0])

    def testDifferentLengths(self):
        """Test weights and vectors of differing lengths produce a ValueError"""
        largest = scalarization.Chebyshev([1.0])
        with self.assertRaises(ValueError):
            largest(self.vector)

    def testEncodeDifferentLengths(self):
        """Test weights and expressions of differing lengths raise encoding"""
        largest = scalarization.Chebyshev([1.0])
        with self.assertRaises(ValueError):
            largest.encode(None, [2.0, 3.0])

if __name__ == "__main__":
    unittest.main()
