import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
import research_workflow as rw  # noqa: E402
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

    def test_main_creates_output_artifacts(self):
        dummy_summary = {
            "start": "2011-01-01",
            "end": "2018-12-31",
            "initial_cash": 100000.0,
            "final_equity": 101000.0,
            "CAGR": 0.01,
            "Volatility": 0.1,
            "Sharpe_approx": 0.2,
            "Sortino_approx": 0.3,
            "MaxDrawdown": -0.1,
            "Exposure": 0.5,
            "Turnover": 1.2,
            "Trades": 3,
            "ProfitFactor": 1.1,
            "WinRate": 0.5,
            "AvgWin": 10.0,
            "AvgLoss": -8.0,
            "AvgWin_R": 0.8,
            "AvgLoss_R": -0.6,
            "Expectancy_R": 0.1,
            "MedianHoldDays": 4.0,
            "MedianMFE_R": 0.4,
            "MedianMAE_R": -0.3,
            "Benchmark_CAGR": 0.02,
            "Benchmark_MaxDrawdown": -0.15,
            "Benchmark_Sortino": 0.25,
        }

        def fake_run_one_variant(universe, split, start, end, benchmark, symbols, variant, data, survivorship_biased):
            return RunResult(
                universe=universe,
                split=split,
                variant_id=variant["id"],
                variant_label=variant["label"],
                family=variant["family"],
                summary=dict(dummy_summary, start=start, end=end),
                trades_df=pd.DataFrame(),
                survivorship_biased=survivorship_biased,
            )

        with tempfile.TemporaryDirectory() as td:
            variants = [
                {"id": "baseline", "family": "baseline", "label": "Baseline", "overrides": {}},
                {"id": "A_pullback", "family": "A", "label": "A", "overrides": {}},
            ]
            with patch.object(rw, "VARIANTS", variants), \
                 patch.object(rw, "symbols_for", return_value=(["AAA"], True)), \
                 patch.object(rw, "download_data", return_value={"AAA": pd.DataFrame({"Close": [1.0]}), "SPY": pd.DataFrame({"Close": [1.0]})}), \
                 patch.object(rw, "run_one_variant", side_effect=fake_run_one_variant), \
                 patch.object(sys, "argv", ["research_workflow.py", "--output-dir", td, "--universe", "sp500"]):
                rw.main()

            self.assertTrue((Path(td) / "research_summary.csv").exists())
            self.assertTrue((Path(td) / "research_trades.csv").exists())
            self.assertTrue((Path(td) / "optimization_confirmation.csv").exists())
            self.assertTrue((Path(td) / "research_report.md").exists())


if __name__ == "__main__":
    unittest.main()
