# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# local files
from visselect.timer import Timer


# --- Test Classes -------------------------------------------------------------

class TestTimer(unittest.TestCase):
    """
    Test the Timer class
    """

    def testAccumulates(self):
        """Test elapsed time accumulates across runs and reads while running"""
        timer = Timer()
        self.assertEqual(timer.elapsed, 0.0)
        with timer:
            running = timer.elapsed
            self.assertTrue(timer.running)
        first = timer.elapsed
        with timer:
            pass
        self.assertFalse(timer.running)
        self.assertGreaterEqual(running, 0.0)
        self.assertGreater(first, 0.0)
        self.assertGreater(timer.elapsed, first)

    def testStoppedWhileReading(self):
        """Test elapsed time does not change while the timer is stopped"""
        timer = Timer()
        with timer:
            pass
        self.assertEqual(timer.elapsed, timer.elapsed)

    def testDoubleStartStop(self):
        """Test starting a running timer or stopping a stopped one raises RTE"""
        timer = Timer()
        with self.assertRaises(RuntimeError):
            timer.stop()
        timer.start()
        with self.assertRaises(RuntimeError):
            timer.start()
        timer.stop()


if __name__ == "__main__":
    unittest.main()
