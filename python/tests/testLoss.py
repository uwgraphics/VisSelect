# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import Dataset, Subset, Loss, metric, objective, scalarization


# --- Test Classes -------------------------------------------------------------

class LossTestCase(unittest.TestCase):
    """
    Common base class for Loss test classes
    """

    def setUp(self):
        """Create test dataset with example data and a test indicator"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.dataset = Dataset(data)
        self.subset = Subset(self.dataset, np.array([False, True, True]))
        self.mean = objective.Preserve(metric.mean)
        self.variance = objective.Preserve(metric.variance)

class TestLossConstructor(LossTestCase):
    """
    Test the Loss class constructor
    """

    def testNoObjectives(self):
        """Test that an empty objectives list produces a ValueError"""
        with self.assertRaises(ValueError):
            Loss(objectives=[])

class TestLossCall(LossTestCase):
    """
    Test the loss object call function
    """

    def testSingleObjective(self):
        """Test that a single objective loss returns the objective value"""
        loss = Loss([self.mean])
        self.assertEqual(loss(self.subset), 4.5)

    def testSingleObjectiveWeighted(self):
        """Test that a weight scales the objective value"""
        linear = scalarization.WeightedSum([2.0])
        loss = Loss([self.mean], linear)
        self.assertEqual(loss(self.subset), 9.0)

    def testTwoObjectives(self):
        """Test that the largest objective is the default blended loss"""
        loss = Loss([self.mean, self.variance])
        self.assertEqual(loss(self.subset), 11.25)

    def testTwoObjectivesWeighted(self):
        """Test that each weight applies to its own objective"""
        linear = scalarization.WeightedSum([2.0, 1.0])
        loss = Loss([self.mean, self.variance], linear)
        self.assertEqual(loss(self.subset), 20.25)

class TestLossNormalized(LossTestCase):
    """
    Test utopia-nadir normalization of the objective terms
    """

    def testNormalizedCall(self):
        """Test that terms scalarize as fractions of their bound ranges"""
        # raw terms are 4.5 and 11.25: both sit halfway along these ranges
        utopia = [0.5, 1.25]
        nadir = [8.5, 21.25]
        loss = Loss([self.mean, self.variance], utopia=utopia, nadir=nadir)
        self.assertEqual(loss(self.subset), 0.5) # the worst fraction
        linear = scalarization.WeightedSum([1.0, 1.0])
        loss = Loss([self.mean, self.variance], linear, utopia, nadir)
        self.assertEqual(loss(self.subset), 1.0) # the sum of fractions

    def testVectorStaysRaw(self):
        """Test that the vector reports raw terms even with bounds"""
        loss = Loss([self.mean, self.variance],
                    utopia=[0.0, 0.0], nadir=[1.0, 1.0])
        assert_array_equal(loss.vector(self.subset), [4.5, 11.25])
        assert_array_equal(loss.utopia, [0.0, 0.0])
        assert_array_equal(loss.nadir, [1.0, 1.0])

    def testDefaultUnbounded(self):
        """Test that a loss without bounds has none and stays raw"""
        loss = Loss([self.mean])
        self.assertIsNone(loss.utopia)
        self.assertIsNone(loss.nadir)
        self.assertEqual(loss(self.subset), 4.5)

    def testRejectBounds(self):
        """Test that lone, misaligned or inverted bounds raise ValueErrors"""
        with self.assertRaisesRegex(ValueError, "come together"):
            Loss([self.mean], utopia=[0.0])
        with self.assertRaisesRegex(ValueError, "Lengths"):
            Loss([self.mean], utopia=[0.0, 1.0], nadir=[1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "not above"):
            Loss([self.mean], utopia=[1.0], nadir=[1.0])

    def testNormalizedEncode(self):
        """Test that encoding applies the same affine map to each term"""
        class Stub: # an objective whose encoding is a plain number
            def __init__(self, value):
                self.value = value
            def __call__(self, subset):
                return self.value
            def encode(self, model, z, dataset, size):
                return self.value
        linear = scalarization.WeightedSum([1.0, 1.0])
        loss = Loss([Stub(4.0), Stub(30.0)], linear,
                    utopia=[0.0, 10.0], nadir=[8.0, 50.0])
        encoded = loss.encode(None, None, self.dataset, 2)
        self.assertEqual(encoded, 0.5 + 0.5) # each term halfway, summed


class TestLossProperties(LossTestCase):
    """
    Test the loss object properties
    """

    def testSingleObjectiveLength(self):
        """Test that a single objective loss has a length of 1"""
        loss = Loss([self.mean])
        self.assertEqual(len(loss), 1)

    def testTwoObjectiveLength(self):
        """Test that the length of loss with two objectives is 2"""
        loss = Loss([self.mean, self.variance])
        self.assertEqual(len(loss), 2)

    def testDefaultScalarization(self):
        """Test that the default scalarization method is the Chebyshev"""
        loss = Loss([self.mean])
        self.assertIsInstance(loss.scalarization, scalarization.Chebyshev)

    def testObjectivesProperty(self):
        """Test that the objectives property returns the correct objectives"""
        loss = Loss([self.mean, self.variance])
        self.assertEqual(loss.objectives, [self.mean, self.variance])

    def testTwoObjectivesVectorMethod(self):
        """Test that the vector method returns the vector with correct values"""
        loss = Loss([self.mean, self.variance])
        assert_array_equal(loss.vector(self.subset), np.asarray([4.5, 11.25]))

if __name__ == "__main__":
    unittest.main()
