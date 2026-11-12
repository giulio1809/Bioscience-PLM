import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_STACK = all(importlib.util.find_spec(p) is not None for p in ("numpy", "scipy", "matplotlib"))


@unittest.skipUnless(HAS_STACK, "Install scientific Python dependencies to test evaluation")
class EvaluationTests(unittest.TestCase):
    def test_rank_direction_and_ties(self):
        from tem1.evaluation import evaluate
        self.assertAlmostEqual(evaluate([1, 2, 2, 4], [10, 20, 20, 40])["spearman_rho"], 1.0)
        self.assertAlmostEqual(evaluate([1, 2, 3], [30, 20, 10])["spearman_rho"], -1.0)

    def test_invalid_inputs_are_not_silently_filtered(self):
        from tem1.evaluation import evaluate
        for observed, predicted in (([1, 2, 3], [1, 1, 1]), ([1, 2, 3], [1, float("nan"), 3]), ([1, 2], [1, 2]), ([1, 2, 3], [1, 2])):
            with self.subTest(observed=observed, predicted=predicted), self.assertRaises(ValueError):
                evaluate(observed, predicted)

    def test_headless_plot_artifacts(self):
        from tem1.evaluation import plot_results
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot_results([1, 2, 3], [3, 1, 2], [1, 1, 2], output, "Synthetic test")
            for name in ("correlation.png", "correlation.pdf", "position_profile.png"):
                self.assertGreater((output / name).stat().st_size, 0)
