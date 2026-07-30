# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Subset, Loss, metric, objective


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

    def testNoWeights(self):
        """Test that no weights list produces a default weight of 1.0"""
        loss = Loss(objectives=[self.mean])
        self.assertEqual(loss._terms[0][0], 1.0)

    def testDifferentLengths(self):
        """Test weights and objectives lengths differ produces a ValueError"""
        with self.assertRaises(ValueError): 
            Loss([self.mean], [0.2, 0.7, 0.1])

class TestLossCall(LossTestCase):
    """
    Test the loss object call function
    """

    def testSingleObjective(self):
        """Test that a single objective loss returns the objective value"""
        loss = Loss([self.mean])
        self.assertEqual(loss(self.subset), 4.5)

    def testWeighted(self):
        """Test that a weight scales the objective value"""
        loss = Loss([self.mean], [2.0])
        self.assertEqual(loss(self.subset), 9.0)

    def testTwoObjectives(self):
        """Test that two objectives sum with default weights"""
        loss = Loss([self.mean, self.variance])
        self.assertEqual(loss(self.subset), 15.75)

    def testTwoWeights(self):
        """Test that each weight applies to its own objective"""
        loss = Loss([self.mean, self.variance], [2.0, 1.0])
        self.assertEqual(loss(self.subset), 20.25)

if __name__ == "__main__":
    unittest.main()
