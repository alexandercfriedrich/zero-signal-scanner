import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from research_workflow import RunResult, load_pit_csv, select_family_winners, symbols_for  # noqa: E402


class ResearchWorkflowTests(unittest.TestCase):
    def test_load_pit_csv_requires_columns(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("foo,bar\n1,2\n")
            path = f.name
        with self.assertRaises(ValueError):
            load_pit_csv(path)

    def test_symbols_for_prefers_pit_entries(self):
        pit = pd.DataFrame(
            [
                {"split": "in_sample", "universe": "sp500", "symbol": "AAPL"},
                {"split": "all", "universe": "all", "symbol": "MSFT"},
            ]
        )
        syms, survivorship = symbols_for("sp500", "in_sample", pit)
        self.assertEqual(sorted(syms), ["AAPL", "MSFT"])
        self.assertFalse(survivorship)

    def test_select_family_winners_ignores_non_finite_scores(self):
        dummy = pd.DataFrame()
        runs = [
            RunResult("sp500", "in_sample", "A_bad", "A bad", "A", {"Sortino_approx": float("nan"), "Expectancy_R": float("nan")}, dummy, True),
            RunResult("sp500", "in_sample", "A_good", "A good", "A", {"Sortino_approx": 0.4, "Expectancy_R": 0.1}, dummy, True),
            RunResult("nasdaq100", "in_sample", "A_good", "A good", "A", {"Sortino_approx": 0.5, "Expectancy_R": 0.2}, dummy, True),
        ]
        winners = select_family_winners(runs)
        self.assertIn("A_good", winners)
        self.assertNotIn("A_bad", winners)


if __name__ == "__main__":
    unittest.main()
