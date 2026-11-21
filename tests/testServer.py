# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# local files
from visselect import metric, objective, scalarization
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

    def testObjectiveMetric(self):
        """Test that a metric is a setting chosen from the metrics by name"""
        settings = self.catalog["objectives"]["Preserve"]["settings"]
        self.assertEqual(settings["metric"]["choices"],
                         ["mean", "variance", "hull"])
        self.assertEqual(settings["metric"]["default"], "mean")
        self.assertEqual(settings["metric"]["scalar"], ["hull"])
        function = self.catalog["metrics"]["variance"]["constructor"]
        built = self.catalog["objectives"]["Preserve"]["constructor"](
            metric=function
        )
        self.assertIsInstance(built, objective.Preserve)
        self.assertIs(built._metric, metric.variance)

    def testObjectiveSettings(self):
        """Test that a norm has its choices"""
        settings = self.catalog["objectives"]["Preserve"]["settings"]
        norm = {
            "default": 1, "choices": [1, 2, "inf"],
            "integer": True, "pair": False,
            "description": "The order of the norm of the difference "
                "between metrics: 1, 2, ∞"
        }
        self.assertEqual(settings["p"], norm)

    def testAlgorithms(self):
        """Test that algorithms are found with their settings and defaults"""
        algorithms = self.catalog["algorithms"]
        self.assertIs(algorithms["GreedySwap"]["constructor"], GreedySwap)
        self.assertIs(algorithms["Random"]["constructor"], Random)
        whole = {"choices": None, "integer": True, "pair": False}
        self.assertEqual(algorithms["Random"]["settings"], {
            "samples": {"default": 1, "description":
                "The number of random samples to take and evaluate"} | whole,
            "seed": {"default": None, "description":
                "The random seed for reproducibility"} | whole,
        })

    def testScalarizations(self):
        """Test that scalarizations are found, their weights left to the page"""
        entry = self.catalog["scalarizations"]["WeightedSum"]
        self.assertIs(entry["constructor"], scalarization.WeightedSum)
        self.assertEqual(entry["settings"], {})

    def testDescriptions(self):
        """Test that a setting carries its docstring description"""
        settings = self.catalog["generators"]["blobs"]["settings"]
        self.assertEqual(settings["balance"]["description"],
                         "How balanced the clusters' shares of items are, "
                         "0 to 1")

    def testGenerators(self):
        """Test that generators are found with whole and decimal pairs"""
        settings = self.catalog["generators"]["blobs"]["settings"]
        self.assertTrue(settings["size"]["pair"])
        self.assertTrue(settings["size"]["integer"])
        self.assertTrue(settings["centerInterval"]["pair"])
        self.assertFalse(settings["centerInterval"]["integer"])
        self.assertFalse(settings["balance"]["pair"])


if __name__ == "__main__":
    unittest.main()
