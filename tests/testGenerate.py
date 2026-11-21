# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np
from numpy.testing import assert_array_equal

# local files
from visselect import generate


# --- Test Classes -------------------------------------------------------------

class TestGenerate(unittest.TestCase):
    """
    Test the random dataset generators
    """

    def testReproducible(self):
        """Test that each generator gives the size asked for, same by seed"""
        for generator in (generate.uniform, generate.binary,
                          generate.categories, generate.normal, generate.blobs):
            data = generator((50, 3), seed=0)
            self.assertEqual(data.shape, (50, 3))
            assert_array_equal(data, generator((50, 3), seed=0))

    def testInterval(self):
        """Test that values stay within an inclusive interval"""
        data = generate.uniform((1000, 2), interval=(-1.0, 2.0), seed=0)
        self.assertTrue(np.all((data >= -1.0) & (data <= 2.0)))

    def testBalanced(self):
        """Test that a balance of 1 gives every category an equal share"""
        data = generate.categories(
            (90, 4), categoriesInterval=(3, 3), balance=1.0, seed=0
        )
        for feature in data.T:
            assert_array_equal(np.bincount(feature), [30, 30, 30])

    def testEveryCategory(self):
        """Test that every category appears even with few items per category"""
        data = generate.categories((9, 20), categoriesInterval=(9, 9), seed=0)
        for feature in data.T:
            assert_array_equal(np.sort(feature), np.arange(9))

    def testRejectBalance(self):
        """Test that a balance outside 0 to 1 raises a ValueError"""
        with self.assertRaises(ValueError):
            generate.blobs((50, 2), balance=1.5)

    def testRejectCategories(self):
        """Test that more categories than items raises a ValueError"""
        with self.assertRaises(ValueError):
            generate.categories((5, 2), categoriesInterval=(2, 8))


if __name__ == "__main__":
    unittest.main()
