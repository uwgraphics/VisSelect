# --- Imports ------------------------------------------------------------------

# standard library
import unittest

# third party
import numpy as np

# local files
from visselect import Dataset, Loss, Problem, Subset, metric, objective
from visselect.algorithm import Greedy, Random
from visselect.solver import Solver


# --- Test Classes -------------------------------------------------------------

class TestSolver(unittest.TestCase):
    """
    Test the Solver class
    """

    def setUp(self):
        """Create a test problem large enough for selected subsets to differ"""
        data = np.arange(60.0).reshape(20, 3)
        self.dataset = Dataset(data)
        self.loss = Loss([objective.Preserve(metric.mean)])
        self.problem = Problem(self.dataset, self.loss, size=5)

    def testNotCallable(self):
        """Test that a non-callable algorithm raises a TypeError"""
        with self.assertRaises(TypeError):
            Solver(5)

    def testDefaultCalibration(self):
        """Test that a mixed unbounded loss calibrates before solving"""
        objectives = [objective.Preserve(metric.mean),
                      objective.Distinctness()]
        problem = Problem(self.dataset, Loss(objectives), size=5)
        auto = Solver(Greedy(seed=0))(problem)

        solver = Solver(Greedy(seed=0))
        utopia, nadir, _ = solver.calibrate(self.dataset, objectives, size=5)
        bounded = Loss(objectives, utopia=utopia, nadir=nadir)
        manual = solver(Problem(self.dataset, bounded, size=5),
                        calibrate=False)
        self.assertEqual(auto.loss, manual.loss)

        # turned off, the same solve runs raw on the problem as given
        raw = Solver(Greedy(seed=0))(problem, calibrate=False)
        self.assertEqual(raw.loss, float(problem.loss(raw.subset)))

    def testCalibrationSkipsVariable(self):
        """Test that a variable size problem solves raw by default"""
        objectives = [objective.Preserve(metric.mean),
                      objective.Distinctness()]
        problem = Problem(self.dataset, Loss(objectives), epsilon=1000.0)
        results = Solver(Greedy(seed=0))(problem)
        self.assertEqual(results.status, "finished")
        self.assertIsNone(problem.loss.utopia)

    def testCalibrate(self):
        """Test that calibration bounds the objectives by the payoff table"""
        objectives = [objective.Preserve(metric.mean),
                      objective.Distinctness()]
        solver = Solver(Greedy(seed=0))
        utopia, nadir, results = solver.calibrate(
            self.dataset, objectives, size=5)
        self.assertEqual((len(utopia), len(nadir), len(results)), (2, 2, 2))
        for low, high in zip(utopia, nadir):
            self.assertLess(low, high)

        # each pure solution is at least as good on its own objective
        for index, result in enumerate(results):
            own = objectives[index](result.subset)
            other = objectives[index](results[1 - index].subset)
            self.assertLessEqual(own, other)

        # the bounds normalize a loss whose terms the solve can then mix
        loss = Loss(objectives, utopia=utopia, nadir=nadir)
        blended = Solver(Greedy(seed=0))(
            Problem(self.dataset, loss, size=5))
        self.assertLess(blended.loss, 1.0) # inside the trade-off region

        # the whole table is remembered: recalibrating solves nothing
        again, worst, results = solver.calibrate(
            self.dataset, objectives, size=5)
        self.assertEqual((again, worst, results), (utopia, nadir, []))

    def testCalibrateRejects(self):
        """Test the constant-payoff calibration error, naming its culprit"""
        solver = Solver(Greedy(seed=0))
        twice = [objective.Preserve(metric.mean),
                 objective.Preserve(metric.mean)]
        with self.assertRaisesRegex(ValueError, "Preserve has no range"):
            solver.calibrate(self.dataset, twice, size=5)

    def testCalibrateSingle(self):
        """Test a lone objective's bounds: declared or solved utopia,
        with the mean loss of random subsets as the nadir"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(20, 3)))
        solver = Solver(Greedy(seed=0))

        # a declared utopia needs no solve at all
        utopia, nadir, results = solver.calibrate(
            dataset, [objective.Crossings()], size=5)
        self.assertEqual((utopia[0], results), (0.0, []))
        self.assertGreater(nadir[0], 0.0)

        # an undeclared one is solved for, once: the measure is remembered
        preserve = objective.Preserve(metric.mean)
        utopia, nadir, results = solver.calibrate(dataset, [preserve], size=5)
        self.assertEqual(len(results), 1)
        self.assertLess(utopia[0], nadir[0])
        again, _, results = solver.calibrate(dataset, [preserve], size=5)
        self.assertEqual((again, results), (utopia, []))

        # later lone solves then normalize against the remembered utopia:
        # the same deterministic solve reaches it, so the loss is zero
        auto = solver(Problem(dataset, Loss([preserve]), size=5))
        self.assertAlmostEqual(auto.loss, 0.0)

    def testUtopiaCacheQuality(self):
        """Test that the cache only improves and skips stopped solves"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(20, 3)))
        solver = Solver(Greedy(seed=0))

        # a stopped calibration solve is not remembered
        preserve = objective.Preserve(metric.mean)
        key = (preserve, dataset, 5)
        solver.calibrate(dataset, [preserve], size=5,
                         callback=lambda results: False)
        self.assertNotIn(key, solver._utopias)

        # a better earlier measurement survives a worse payoff column
        solver._utopias[key] = -100.0
        utopia, _, _ = solver.calibrate(
            dataset, [preserve, objective.Distinctness()], size=5)
        self.assertEqual(utopia[0], -100.0)
        self.assertEqual(solver._utopias[key], -100.0)

        # stopped payoff solves are not remembered either: both stop on
        # the same seeded start, so the constant column raises
        distinct = objective.Distinctness()
        with self.assertRaisesRegex(ValueError, "no range"):
            solver.calibrate(dataset, [preserve, distinct], size=5,
                             callback=lambda results: False)
        self.assertNotIn((distinct, dataset, 5), solver._utopias)

    def testCalibrateSatisfied(self):
        """Test that a declared goal with no range falls back to a unit
        range instead of erroring: it is simply already satisfied"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(50, 2)))
        dataset.root.scale(name="scaled")
        solver = Solver(Greedy(seed=0))

        # tiny marks never overlap randomly: utopia and nadir both 0
        overdraw = objective.Overdraw(radius=0.001, on="scaled")
        utopia, nadir, _ = solver.calibrate(dataset, [overdraw], size=3)
        self.assertEqual((utopia, nadir), ([0.0], [1.0]))

        # a declared optimum ends the solve the moment it is reached
        results = solver(Problem(dataset, Loss([overdraw]), size=3))
        self.assertEqual((results.loss, results.status), (0.0, "finished"))
        self.assertEqual(results.evaluations, 1)

        # and in a blend the satisfied goal passes where it used to raise
        mixed = [overdraw, objective.Preserve(metric.mean, on="scaled")]
        utopia, nadir, _ = solver.calibrate(dataset, mixed, size=3)
        self.assertEqual((utopia[0], nadir[0]), (0.0, 1.0))

    def testCalibrateDeclared(self):
        """Test that a declared utopia replaces the measured one"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(20, 3)))
        utopia, nadir, _ = Solver(Greedy(seed=0)).calibrate(
            dataset, [objective.Crossings(),
                      objective.Preserve(metric.mean)], size=5)
        self.assertEqual(utopia[0], 0.0) # exact, even if never reached

        class Known: # declares a utopia computed from the problem
            def __call__(self, subset):
                return float(np.ptp(subset["data"]))
            def utopia(self, dataset, size):
                return -1.0
        utopia, nadir, _ = Solver(Greedy(seed=0)).calibrate(
            dataset, [Known(), objective.Preserve(metric.mean)], size=5)
        self.assertEqual(utopia[0], -1.0)

    def testDefaultSingleDeclared(self):
        """Test that a lone declaring objective solves normalized"""
        rng = np.random.default_rng(0)
        dataset = Dataset(rng.normal(size=(20, 3)))
        problem = Problem(dataset, Loss([objective.Crossings()]), size=5)
        raw = Solver(Greedy(seed=0))(problem, calibrate=False)
        auto = Solver(Greedy(seed=0))(problem)

        # normalization is monotone, so the same subset is found, and
        # the loss maps onto the declared-to-random scale
        self.assertTrue(np.array_equal(raw.subset.indicator,
                                       auto.subset.indicator))
        utopia, nadir, _ = Solver(Greedy(seed=0)).calibrate(
            dataset, [objective.Crossings()], size=5)
        expected = (raw.loss - utopia[0]) / (nadir[0] - utopia[0])
        self.assertAlmostEqual(auto.loss, expected)

    def testVariableSizeGate(self):
        """Test that a variable size problem needs a declaring algorithm"""
        variable = Problem(self.dataset, self.loss, epsilon=0.5)
        with self.assertRaisesRegex(ValueError, "fixed size"):
            Solver(Random(samples=1))(variable)

        class Resizer: # declares support through the variableSize member
            variableSize = True
            def __call__(self, problem, emit):
                indicator = np.arange(len(problem.dataset)) < 2
                subset = Subset(problem.dataset, indicator)
                emit(subset, problem.evaluate(subset))

        results = Solver(Resizer())(variable)
        self.assertEqual(len(results.losses), 1)

    def testBest(self):
        """Test that the results keep the lowest loss and its frozen subset"""
        results = Solver(Random(samples=20, seed=0))(self.problem)
        self.assertEqual(len(results.losses), 20)
        self.assertEqual(results._loss, min(results.losses))
        self.assertEqual(results._loss, float(self.problem.loss(results.subset)))

    def testNoLog(self):
        """Test that turning off logging keeps the best loss but no log"""
        solver = Solver(Random(samples=20, seed=0))
        logged = solver(self.problem)
        unlogged = solver(self.problem, log=False)
        self.assertEqual(unlogged.losses, [])
        self.assertEqual(unlogged._loss, logged._loss)

    def testTimes(self):
        """Test that logged times accumulate, one nondecreasing time per step"""
        results = Solver(Random(samples=20, seed=0))(self.problem)
        self.assertEqual(len(results._times), 20)
        self.assertGreater(results._times[0], 0.0)
        self.assertTrue(all(a <= b for a, b in zip(results._times,
                                                   results._times[1:])))
        self.assertGreaterEqual(results.time, results._times[-1])

    def testStop(self):
        """Test that stopping from the step callback keeps the results so far"""
        results = Solver(Random(samples=20, seed=0))(
            self.problem, callback=lambda results: len(results.losses) < 2
        )
        self.assertEqual(len(results.losses), 2)
        self.assertEqual(results._status, "stopped")


if __name__ == "__main__":
    unittest.main()
