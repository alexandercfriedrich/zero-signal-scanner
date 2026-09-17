import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
import research_workflow as rw  # noqa: E402
from research_workflow import (  # noqa: E402
    RunResult,
    load_pit_csv,
    select_is_winners,
    symbols_for,
    warmup_start_for,
)


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

    def test_select_is_winners_uses_rule_hierarchy(self):
        rows = [
            {
                "universe": "sp500",
                "split": "in_sample",
                "variant_id": "baseline",
                "family": "baseline",
                "Trades": 80,
                "Expectancy_R": 0.05,
                "ProfitFactor": 1.10,
                "Sortino_approx": 0.40,
                "MaxDrawdown": -0.20,
            },
            {
                "universe": "sp500",
                "split": "in_sample",
                "variant_id": "sens_a_bad",
                "family": "sens_pullback_tol",
                "Trades": 39,
                "Expectancy_R": 0.20,
                "ProfitFactor": 1.40,
                "Sortino_approx": 0.90,
                "MaxDrawdown": -0.19,
            },
            {
                "universe": "sp500",
                "split": "in_sample",
                "variant_id": "sens_a_good",
                "family": "sens_pullback_tol",
                "Trades": 60,
                "Expectancy_R": 0.11,
                "ProfitFactor": 1.25,
                "Sortino_approx": 0.75,
                "MaxDrawdown": -0.21,
            },
            {
                "universe": "sp500",
                "split": "in_sample",
                "variant_id": "sens_c_good",
                "family": "sens_ext",
                "Trades": 58,
                "Expectancy_R": 0.09,
                "ProfitFactor": 1.15,
                "Sortino_approx": 0.60,
                "MaxDrawdown": -0.20,
            },
        ]
        df = pd.DataFrame(rows)
        winners = select_is_winners(df, universe="sp500", min_trades=40)
        self.assertIn("sens_a_good", winners)
        self.assertIn("sens_c_good", winners)
        self.assertNotIn("sens_a_bad", winners)

    def test_warmup_start_provides_enough_history(self):
        split_start = "2023-01-03"
        warmup = warmup_start_for(split_start, 300)
        dates = pd.bdate_range(warmup, split_start)
        # Synthetic close series; with 300 warmup bars SMA200 and Mom126 must be valid at split start
        close = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
        sma200 = close.rolling(200, min_periods=200).mean()
        mom126 = close / close.shift(126) - 1
        self.assertTrue(np.isfinite(float(sma200.loc[pd.Timestamp(split_start)])))
        self.assertTrue(np.isfinite(float(mom126.loc[pd.Timestamp(split_start)])))

    def test_main_creates_output_artifacts(self):
        dummy_summary = {
            "initial_cash": 5000.0,
            "final_equity": 5200.0,
            "CAGR": 0.04,
            "Volatility": 0.10,
            "Sharpe_approx": 0.30,
            "Sortino_approx": 0.40,
            "MaxDrawdown": -0.10,
            "Exposure": 0.50,
            "Turnover": 1.20,
            "Trades": 50,
            "ProfitFactor": 1.20,
            "WinRate": 0.52,
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
            "Benchmark_Sharpe": 0.20,
            "Benchmark_Sortino": 0.25,
            "last_trading_day": "2026-09-16",
            "split_start": "2011-01-01",
            "split_end": "2018-12-31",
            "warmup_start": "2010-01-01",
            "as_of": "2026-09-17",
            "baseline_config_path": "app/config_best_2011_2026.json",
            "baseline_config_hash": "abc",
            "survivorship_biased": True,
            "entry_mode": "breakout_only",
            "signal_assumption": "signals_on_close_entry_next_open",
            "cost_assumption": "spread_bps_per_side=8",
            "data_source": "yfinance",
            "warmup_indicator_ready_fraction": 1.0,
        }

        def fake_run_one_variant(**kwargs):
            variant = kwargs["variant"]
            return RunResult(
                universe=kwargs["universe"],
                split=kwargs["split"],
                variant_id=variant["id"],
                variant_label=variant["label"],
                family=variant["family"],
                summary=dict(dummy_summary, split_start=kwargs["split_start"], split_end=kwargs["split_end"]),
                trades_df=pd.DataFrame(),
                survivorship_biased=True,
            )

        with tempfile.TemporaryDirectory() as td:
            fake_variants = [
                {"id": "baseline", "family": "baseline", "label": "Baseline", "overrides": {"entry_mode": "breakout_only"}},
                {"id": "sens_ext_0.50", "family": "sens_ext", "label": "sens", "overrides": {"entry_mode": "breakout_only"}},
            ]
            with patch.object(rw, "build_variants", return_value=fake_variants), \
                 patch.object(rw, "symbols_for", return_value=(["AAA"], True)), \
                 patch.object(rw, "download_data", return_value={"AAA": pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2010-01-01", periods=1)), "SPY": pd.DataFrame({"Close": [1.0]}, index=pd.bdate_range("2010-01-01", periods=1))}), \
                 patch.object(rw, "run_one_variant", side_effect=fake_run_one_variant), \
                 patch.object(rw, "read_baseline_config", return_value=({}, "app/config_best_2011_2026.json", "abc")), \
                 patch.object(sys, "argv", ["research_workflow.py", "--output-dir", td, "--universe", "sp500", "--as-of", "2026-09-17"]):
                rw.main()

            self.assertTrue((Path(td) / "research_summary.csv").exists())
            self.assertTrue((Path(td) / "research_trades.csv").exists())
            self.assertTrue((Path(td) / "optimization_confirmation.csv").exists())
            self.assertTrue((Path(td) / "research_report.md").exists())


if __name__ == "__main__":
    unittest.main()
