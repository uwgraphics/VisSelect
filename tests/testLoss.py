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
