import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))
from price_cache import MarketDataCache  # noqa: E402


class FakeDownloader:
    def __init__(self):
        self.calls = []

    def __call__(self, symbol: str, start: str, end: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        self.calls.append({
            "symbol": symbol,
            "start": start,
            "end": end,
            "interval": interval,
            "auto_adjust": auto_adjust,
        })
        idx = pd.bdate_range(start, end)
        if len(idx) == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
        df = pd.DataFrame({
            "Open": range(1, len(idx) + 1),
            "High": range(2, len(idx) + 2),
            "Low": range(0, len(idx)),
            "Close": range(1, len(idx) + 1),
            "Adj Close": range(1, len(idx) + 1),
            "Volume": [1000] * len(idx),
        }, index=idx)
        return df


class PriceCacheTests(unittest.TestCase):
    def test_first_run_downloads_full_range(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False)
            res = cache.fetch_symbol("AAA", "2020-01-01", "2020-01-10")
            self.assertEqual(len(dl.calls), 1)
            self.assertEqual(dl.calls[0]["start"], "2020-01-01")
            self.assertEqual(dl.calls[0]["end"], "2020-01-10")
            self.assertFalse(res.data.empty)

    def test_second_run_downloads_only_suffix_with_overlap(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False, overlap_bdays=5)
            cache.fetch_symbol("AAA", "2020-01-01", "2020-01-10")
            dl.calls.clear()
            cache.fetch_symbol("AAA", "2020-01-01", "2020-01-20")
            self.assertEqual(len(dl.calls), 1)
            self.assertEqual(dl.calls[0]["end"], "2020-01-20")
            self.assertNotEqual(dl.calls[0]["start"], "2020-01-01")

    def test_earlier_start_downloads_only_prefix(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False)
            cache.fetch_symbol("AAA", "2020-01-10", "2020-01-20")
            dl.calls.clear()
            cache.fetch_symbol("AAA", "2020-01-01", "2020-01-20")
            self.assertEqual(len(dl.calls), 1)
            self.assertEqual(dl.calls[0]["start"], "2020-01-01")
            self.assertLess(pd.Timestamp(dl.calls[0]["end"]), pd.Timestamp("2020-01-10"))

    def test_merged_data_sorted_and_unique(self):
        class DupDownloader(FakeDownloader):
            def __call__(self, symbol, start, end, interval, auto_adjust):
                base = super().__call__(symbol, start, end, interval, auto_adjust)
                if base.empty:
                    return base
                rev = base.iloc[::-1].copy()
                dup = pd.concat([rev, rev.iloc[:2]], axis=0)
                return dup

        dl = DupDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False)
            res = cache.fetch_symbol("AAA", "2020-01-01", "2020-01-20")
            self.assertTrue(res.data.index.is_monotonic_increasing)
            self.assertEqual(int(res.data.index.duplicated().sum()), 0)

    def test_refresh_ignores_existing_cache(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False)
            cache.fetch_symbol("AAA", "2020-01-01", "2020-01-10")
            dl.calls.clear()
            refresh_cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=True)
            refresh_cache.fetch_symbol("AAA", "2020-01-01", "2020-01-10")
            self.assertEqual(len(dl.calls), 1)
            self.assertEqual(dl.calls[0]["start"], "2020-01-01")

    def test_spy_qqq_and_single_ticker_treated_consistently(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=True, refresh=False)
            data, info = cache.fetch_many(["SPY", "QQQ", "AAPL"], "2020-01-01", "2020-01-10")
            self.assertEqual(set(data.keys()), {"SPY", "QQQ", "AAPL"})
            self.assertEqual(set(info.keys()), {"SPY", "QQQ", "AAPL"})
            manifest = Path(td) / "research_price_manifest.json"
            self.assertTrue(manifest.exists())

    def test_no_cache_reads_and_writes_nothing(self):
        dl = FakeDownloader()
        with tempfile.TemporaryDirectory() as td:
            cache = MarketDataCache(td, downloader=dl, use_cache=False, refresh=False)
            res = cache.fetch_symbol("SPY", "2020-01-01", "2020-01-10")
            self.assertFalse(res.data.empty)
            files = list(Path(td).glob("*"))
            self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
