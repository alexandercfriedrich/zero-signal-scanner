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
    @staticmethod
    def _nasdaq_html(header: str, n: int = 100, table_id: str | None = "constituents") -> str:
        rows = []
        for i in range(n):
            sym = f"SYM{i:03d}"
            if i == 1:
                sym = "BRK.B"
            rows.append(f"<tr><td>{sym}</td><td>Name {i}</td></tr>")
        id_attr = f' id="{table_id}"' if table_id else ""
        table = (
            f"<table{id_attr}>"
            f"<thead><tr><th>{header}</th><th>Company</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
        )
        return f"<html><body><table><tr><th>Foo</th></tr><tr><td>Bar</td></tr></table>{table}</body></html>"

    @staticmethod
    def _resp(html: str):
        class _Resp:
            def __init__(self, text: str):
                self.text = text

            def raise_for_status(self):
                return None

        return _Resp(html)

    def test_load_nasdaq100_symbols_accepts_ticker_header(self):
        html = self._nasdaq_html("Ticker", 100, "constituents")
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            syms = rw.load_nasdaq100_symbols()
        self.assertEqual(len(syms), 100)
        self.assertIn("BRK-B", syms)

    def test_load_nasdaq100_symbols_accepts_symbol_header(self):
        html = self._nasdaq_html("Symbol", 100, "constituents")
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            syms = rw.load_nasdaq100_symbols()
        self.assertEqual(len(syms), 100)

    def test_load_nasdaq100_symbols_accepts_ticker_symbol_alias(self):
        html = self._nasdaq_html("Ticker Symbol", 100, None)
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            syms = rw.load_nasdaq100_symbols()
        self.assertEqual(len(syms), 100)

    def test_load_nasdaq100_symbols_accepts_ticker_parenthesized_alias(self):
        html = self._nasdaq_html("Ticker(s)", 100, "constituents")
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            syms = rw.load_nasdaq100_symbols()
        self.assertEqual(len(syms), 100)

    def test_load_nasdaq100_symbols_raises_when_no_matching_table(self):
        html = "<html><body><table id='constituents'><tr><th>Company</th></tr><tr><td>A</td></tr></table></body></html>"
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            with self.assertRaises(ValueError) as ctx:
                rw.load_nasdaq100_symbols()
        self.assertIn("Detected tables", str(ctx.exception))

    def test_load_nasdaq100_symbols_raises_on_implausible_count(self):
        html = self._nasdaq_html("Ticker", 10, "constituents")
        with patch.object(rw.requests, "get", return_value=self._resp(html)):
            with self.assertRaises(ValueError) as ctx:
                rw.load_nasdaq100_symbols()
        self.assertIn("implausible unique symbol count", str(ctx.exception))

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
            idx = pd.bdate_range("2010-01-01", periods=300)
            mk = lambda: pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.linspace(1, 2, len(idx)), "Adj Close": np.linspace(1, 2, len(idx)), "Volume": 1000}, index=idx)
            with patch.object(rw, "build_variants", return_value=fake_variants), \
                 patch.object(rw, "symbols_for", return_value=(["AAA"], True)), \
                 patch.object(rw, "download_data", return_value=(
                    {"AAA": mk(), "SPY": mk()},
                    {"AAA": {"cache_hit": True, "download_rows": 0}, "SPY": {"cache_hit": True, "download_rows": 0}}
                )), \
                patch.object(rw, "run_one_variant", side_effect=fake_run_one_variant), \
                patch.object(rw, "read_baseline_config", return_value=({}, "app/config_best_2011_2026.json", "abc")), \
                patch.object(sys, "argv", ["research_workflow.py", "--output-dir", td, "--universe", "sp500", "--as-of", "2026-09-17"]):
               rw.main()

            self.assertTrue((Path(td) / "research_summary.csv").exists())
            self.assertTrue((Path(td) / "research_trades.csv").exists())
            self.assertTrue((Path(td) / "optimization_confirmation.csv").exists())
            self.assertTrue((Path(td) / "research_report.md").exists())
            self.assertTrue((Path(td) / "cache_usage.csv").exists())

    def test_overlapping_symbols_downloaded_once_in_both_universe_mode(self):
        captured = {}

        def fake_symbols_for(universe, split, pit_df):
            if universe == "sp500":
                return (["AAPL", "MSFT", "NVDA"], True)
            return (["MSFT", "NVDA", "TSLA"], True)

        def fake_download(cache, symbols, start, end):
            captured["symbols"] = list(symbols)
            idx = pd.bdate_range("2010-01-01", periods=400)
            mk = lambda: pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.linspace(1, 2, len(idx)), "Adj Close": np.linspace(1, 2, len(idx)), "Volume": 1000}, index=idx)
            data = {s: mk() for s in symbols}
            infos = {s: {"cache_hit": True, "download_rows": 0} for s in symbols}
            return data, infos

        def fake_run_one_variant(**kwargs):
            v = kwargs["variant"]
            return RunResult(
                universe=kwargs["universe"],
                split=kwargs["split"],
                variant_id=v["id"],
                variant_label=v["label"],
                family=v["family"],
                summary={"Trades": 0, "Sortino_approx": np.nan, "Expectancy_R": np.nan, "MaxDrawdown": np.nan},
                trades_df=pd.DataFrame(),
                survivorship_biased=True,
            )

        with tempfile.TemporaryDirectory() as td:
            with patch.object(rw, "symbols_for", side_effect=fake_symbols_for), \
                 patch.object(rw, "download_data", side_effect=fake_download), \
            patch.object(rw, "run_one_variant", side_effect=fake_run_one_variant), \
            patch.object(rw, "read_baseline_config", return_value=({}, "app/config_best_2011_2026.json", "abc")), \
            patch.object(sys, "argv", ["research_workflow.py", "--output-dir", td, "--universe", "both", "--smoke-test"]):
                rw.main()

            syms = captured["symbols"]
            self.assertEqual(len(syms), len(set(syms)))
            self.assertIn("SPY", syms)
            self.assertIn("QQQ", syms)

    def test_smoke_test_marked_and_never_confirms(self):
        with tempfile.TemporaryDirectory() as td:
            idx = pd.bdate_range("2010-01-01", periods=400)
            mk = lambda: pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.linspace(1, 2, len(idx)), "Adj Close": np.linspace(1, 2, len(idx)), "Volume": 1000}, index=idx)
            def fake_run_one_variant(**kwargs):
                v = kwargs["variant"]
                return RunResult(
                    universe=kwargs["universe"],
                    split=kwargs["split"],
                    variant_id=v["id"],
                    variant_label=v["label"],
                    family=v["family"],
                    summary={"Trades": 10, "Sortino_approx": 0.1, "Expectancy_R": 0.01, "MaxDrawdown": -0.1},
                    trades_df=pd.DataFrame(),
                    survivorship_biased=True,
                )
            with patch.object(rw, "symbols_for", return_value=(["AAPL", "MSFT"], True)), \
                 patch.object(rw, "download_data", return_value=({"AAPL": mk(), "MSFT": mk(), "SPY": mk(), "QQQ": mk()}, {"AAPL": {}, "MSFT": {}, "SPY": {}, "QQQ": {}})), \
                 patch.object(rw, "run_one_variant", side_effect=fake_run_one_variant), \
                 patch.object(rw, "read_baseline_config", return_value=({}, "app/config_best_2011_2026.json", "abc")), \
                 patch.object(sys, "argv", ["research_workflow.py", "--output-dir", td, "--universe", "sp500", "--smoke-test"]):
                rw.main()

            report = (Path(td) / "research_report.md").read_text(encoding="utf-8")
            self.assertIn("SMOKE TEST — NOT A PERFORMANCE VALIDATION", report)
            conf = pd.read_csv(Path(td) / "optimization_confirmation.csv")
            if not conf.empty:
                self.assertFalse(conf["confirmed_in_both_oos"].astype(bool).any())

    def test_run_one_variant_does_not_mutate_shared_features(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        base_df = pd.DataFrame({"Open": [1, 1, 1, 1, 1], "High": [1, 1, 1, 1, 1], "Low": [1, 1, 1, 1, 1], "Close": [1, 1, 1, 1, 1], "Adj Close": [1, 1, 1, 1, 1], "Volume": [1000] * 5}, index=idx)
        prepared = {"AAA": base_df.copy(), "SPY": base_df.copy()}
        prepared_before = prepared["AAA"]["Close"].copy()

        def fake_run_backtest(data, cfg):
            data["AAA"]["Close"] = 0.0
            eq = pd.DataFrame({"Equity": [1000, 1001], "Exposure": [0.1, 0.2]}, index=pd.bdate_range("2020-01-02", periods=2))
            tr = pd.DataFrame(columns=["entry_date", "entry_px", "exit_px", "shares", "pnl"])
            return eq, tr, {}, {}

        with patch.object(rw, "run_backtest", side_effect=fake_run_backtest):
            rw.run_one_variant(
                universe="sp500",
                split="in_sample",
                split_start="2020-01-02",
                split_end="2020-01-10",
                warmup_start="2020-01-01",
                benchmark="SPY",
                symbols=["AAA"],
                variant={"id": "baseline", "label": "Baseline", "family": "baseline", "overrides": {}},
                prepared_split_data=prepared,
                baseline_cfg={"initial_cash": 1000},
                baseline_path="app/config_best_2011_2026.json",
                baseline_hash="abc",
                survivorship_biased=True,
                as_of="2020-01-10",
            )

        self.assertTrue(prepared["AAA"]["Close"].equals(prepared_before))


if __name__ == "__main__":
    unittest.main()
