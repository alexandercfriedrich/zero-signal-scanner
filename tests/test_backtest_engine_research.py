import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from backtest_engine import run_backtest  # noqa: E402


class BacktestResearchTests(unittest.TestCase):
    def _make_df(self, close_values: list[float], open_override: dict[int, float] | None = None) -> pd.DataFrame:
        idx = pd.bdate_range("2020-01-01", periods=len(close_values))
        rows = []
        for i, c in enumerate(close_values):
            o = float(c) if not open_override or i not in open_override else float(open_override[i])
            rows.append(
                {
                    "Date": idx[i],
                    "Open": o,
                    "High": float(max(o, c) + 1.0),
                    "Low": float(min(o, c) - 1.0),
                    "Close": float(c),
                    "Volume": 1_000_000 + i * 1000,
                }
            )
        return pd.DataFrame(rows).set_index("Date")

    def _base_cfg(self, start: str, end: str) -> dict:
        return {
            "symbols": ["AAA"],
            "regime_symbol": "SPY",
            "start": start,
            "end": end,
            "hard_risk_on": True,
            "max_new_trades_per_day": 1,
            "max_positions": 1,
            "weekly_rerank": False,
            "risk_per_trade": 0.02,
            "atr_period": 3,
            "atr_stop_mult": 1.5,
            "use_trailing_stop": False,
            "take_profit_R": 2.0,
            "atr_trail_mult": 3.0,
            "breakout_lookback": 10,
            "breakout_level_source": "close",
            "breakout_confirm_closes": 1,
            "sma_regime": 20,
            "max_holding_days": 1,
            "mom_lookback": 5,
            "spread_bps_per_side": 0,
            "min_price": 1.0,
            "min_dollar_volume": 0,
            "initial_cash": 10000,
            "enable_cwh": False,
        }

    def test_pullback_setup_generates_trade(self):
        close = list(np.linspace(100, 240, 220)) + [228, 226, 227, 229, 231, 232, 233, 234, 235, 236, 237, 238]
        data = {
            "AAA": self._make_df(close),
            "SPY": self._make_df(list(np.linspace(100, 130, len(close)))),
        }
        idx = data["AAA"].index
        cfg = self._base_cfg(str(idx[0].date()), str(idx[-1].date()))
        cfg.update(
            {
                "enable_pullback_entry": True,
                "pullback_sma_tolerance_atr": 3.0,
                "pullback_range_tolerance_atr": 3.0,
                "pullback_invalidation_atr": 0.5,
                "max_breakout_extension_atr": 0.0,
            }
        )

        _, trades, _, _ = run_backtest(data, cfg)

        self.assertFalse(trades.empty)
        self.assertIn("PULLBACK", set(trades["setup"].tolist()))

    def test_entry_uses_next_day_open_without_lookahead(self):
        close = [10.0] * 25 + [9.95, 9.98, 10.0, 10.02, 10.05, 10.08, 10.1, 11.5, 11.1, 11.2, 11.25]
        open_override = {33: 12.34}
        data = {
            "AAA": self._make_df(close, open_override=open_override),
            "SPY": self._make_df([20 + i * 0.1 for i in range(len(close))]),
        }
        idx = data["AAA"].index
        cfg = self._base_cfg(str(idx[0].date()), str(idx[-1].date()))

        _, trades, _, _ = run_backtest(data, cfg)

        self.assertFalse(trades.empty)
        matches = trades[np.isclose(trades["entry_px"], 12.34)]
        self.assertFalse(matches.empty)
        self.assertEqual(matches.iloc[0]["entry_date"], idx[33].date().isoformat())

    def test_summary_contains_research_metrics(self):
        close = [10 + i * 0.2 for i in range(40)]
        data = {
            "AAA": self._make_df(close),
            "SPY": self._make_df([30 + i * 0.15 for i in range(40)]),
        }
        idx = data["AAA"].index
        cfg = self._base_cfg(str(idx[0].date()), str(idx[-1].date()))

        _, trades, summary, _ = run_backtest(data, cfg)

        self.assertIn("Sortino_approx", summary)
        self.assertIn("Exposure", summary)
        self.assertIn("Turnover", summary)
        self.assertIn("MedianHoldDays", summary)
        if not trades.empty:
            self.assertIn("MFE_R", trades.columns)
            self.assertIn("MAE_R", trades.columns)


if __name__ == "__main__":
    unittest.main()
