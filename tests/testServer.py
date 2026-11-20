# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# local files
from visselect import metric, objective
from visselect.algorithm import GreedySwap, Random
from visselect.server import buildCatalog


# --- Test Classes -------------------------------------------------------------

class TestBuildCatalog(unittest.TestCase):
    """
    Test building the catalog of what the package offers
    """

    def setUp(self):
        """Build the catalog of the package"""
        self.catalog = buildCatalog()

    def testObjectivesPerMetric(self):
        """Test that an objective taking a metric is expanded per metric"""
        objectives = self.catalog["objectives"]
        for name in ("mean", "variance", "hull"):
            self.assertIn(f"Preserve {name}", objectives)
        built = objectives["Preserve variance"]["constructor"]()
        self.assertIsInstance(built, objective.Preserve)
        self.assertIs(built._metric, metric.variance)

    def testObjectiveSettings(self):
        """Test that a fixed metric is left out and a norm has its choices"""
        settings = self.catalog["objectives"]["Preserve mean"]["settings"]
        norm = {"default": 1, "choices": [1, 2, "inf"]}
        self.assertEqual(settings, {"p": norm})

    def testAlgorithms(self):
        """Test that algorithms are found with their settings and defaults"""
        algorithms = self.catalog["algorithms"]
        self.assertIs(algorithms["GreedySwap"]["constructor"], GreedySwap)
        self.assertIs(algorithms["Random"]["constructor"], Random)
        self.assertEqual(algorithms["Random"]["settings"], {
            "samples": {"default": 1, "choices": None},
            "seed": {"default": None, "choices": None},
        })


if __name__ == "__main__":
    unittest.main()
