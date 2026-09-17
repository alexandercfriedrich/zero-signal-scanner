from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


@dataclass
class CacheFetchResult:
    data: pd.DataFrame
    info: dict[str, Any]


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "_").replace("/", "_").replace(":", "_").replace(".", "-")


def _to_naive_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        dt = pd.to_datetime(out["Date"], errors="coerce")
        out = out.drop(columns=["Date"])
        out.index = dt
    elif not isinstance(out.index, pd.DatetimeIndex):
        first = out.columns[0]
        dt = pd.to_datetime(out[first], errors="coerce")
        out = out.drop(columns=[first])
        out.index = dt
    out = out[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    out.index = pd.to_datetime(out.index.date)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    out = _to_naive_datetime_index(df)
    cols = {str(c).strip(): c for c in out.columns}
    canonical = {}
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in cols:
            canonical[c] = pd.to_numeric(out[cols[c]], errors="coerce")
    if "Close" in canonical and "Adj Close" not in canonical:
        canonical["Adj Close"] = canonical["Close"]
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in canonical:
            canonical[c] = pd.Series(index=out.index, dtype="float64")

    res = pd.DataFrame(canonical, index=out.index)
    res = res.sort_index()
    res = res[~res.index.duplicated(keep="last")]
    return res


class MarketDataCache:
    def __init__(
        self,
        cache_dir: str | Path,
        provider: str = "yfinance",
        interval: str = "1d",
        auto_adjust: bool = False,
        schema_version: str = "research_v1",
        use_cache: bool = True,
        refresh: bool = False,
        overlap_bdays: int = 5,
        downloader: Callable[[str, str, str, str, bool], pd.DataFrame] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.schema_version = schema_version
        self.use_cache = use_cache
        self.refresh = refresh
        self.overlap_bdays = int(overlap_bdays)
        self.downloader = downloader or self._download_symbol
        self.manifest_path = self.cache_dir / "research_price_manifest.json"

    def _key_prefix(self, symbol: str) -> str:
        return f"{self.provider}__{_safe_symbol(symbol)}__{self.interval}__adj{int(self.auto_adjust)}__{self.schema_version}"

    def _parquet_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{self._key_prefix(symbol)}.parquet"

    def _csv_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{self._key_prefix(symbol)}.csv"

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": self.schema_version, "entries": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": self.schema_version, "entries": {}}

    def _atomic_write_text(self, path: Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _atomic_write_df(self, path: Path, df: pd.DataFrame, as_csv: bool = False) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        if as_csv:
            df.reset_index().rename(columns={"index": "Date"}).to_csv(tmp, index=False)
        else:
            df.reset_index().rename(columns={"index": "Date"}).to_parquet(tmp, index=False)
        tmp.replace(path)

    def _write_manifest_entry(self, symbol: str, entry: dict[str, Any]) -> None:
        m = self._load_manifest()
        m.setdefault("entries", {})[self._key_prefix(symbol)] = entry
        self._atomic_write_text(self.manifest_path, json.dumps(m, indent=2, sort_keys=True))

    def _download_symbol(self, symbol: str, start: str, end: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        raw = yf.download(
            tickers=symbol,
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=False,
        )
        if isinstance(raw, pd.DataFrame) and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(-1):
                sub = raw.xs(symbol, axis=1, level=-1)
                return _normalize_ohlcv(sub)
        return _normalize_ohlcv(raw)

    def _load_cached_df(self, symbol: str) -> tuple[pd.DataFrame, str | None]:
        p_parq = self._parquet_path(symbol)
        p_csv = self._csv_path(symbol)
        if p_parq.exists():
            try:
                return _normalize_ohlcv(pd.read_parquet(p_parq)), "parquet"
            except Exception:
                pass
        if p_csv.exists():
            try:
                return _normalize_ohlcv(pd.read_csv(p_csv)), "csv"
            except Exception:
                pass
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]), None

    def _save_cached_df(self, symbol: str, df: pd.DataFrame) -> str:
        p_parq = self._parquet_path(symbol)
        try:
            self._atomic_write_df(p_parq, df, as_csv=False)
            return "parquet"
        except Exception:
            p_csv = self._csv_path(symbol)
            self._atomic_write_df(p_csv, df, as_csv=True)
            return "csv"

    def _merge(self, base: pd.DataFrame, parts: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [base] + [p for p in parts if p is not None and not p.empty]
        if not frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
        merged = pd.concat(frames, axis=0)
        merged = _normalize_ohlcv(merged)
        return merged

    def fetch_symbol(self, symbol: str, start: str, end: str) -> CacheFetchResult:
        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)
        info: dict[str, Any] = {
            "symbol": symbol,
            "cache_used": bool(self.use_cache and not self.refresh),
            "cache_hit": False,
            "downloaded_ranges": [],
            "download_rows": 0,
            "cache_format": None,
        }

        if not self.use_cache:
            fresh = _normalize_ohlcv(self.downloader(symbol, start, end, self.interval, self.auto_adjust))
            clipped = fresh.loc[(fresh.index >= req_start) & (fresh.index <= req_end)]
            info.update({
                "cache_hit": False,
                "download_rows": int(len(fresh)),
                "range_start": str(clipped.index.min().date()) if not clipped.empty else None,
                "range_end": str(clipped.index.max().date()) if not clipped.empty else None,
                "last_success_update": pd.Timestamp.now("UTC").isoformat(),
            })
            return CacheFetchResult(clipped, info)

        cached, fmt = (pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]), None)
        if not self.refresh:
            cached, fmt = self._load_cached_df(symbol)
            if not cached.empty:
                info["cache_hit"] = True
                info["cache_format"] = fmt

        needs: list[tuple[str, str]] = []
        if self.refresh or cached.empty:
            needs.append((start, end))
        else:
            cache_start = cached.index.min()
            cache_end = cached.index.max()
            if req_start < cache_start:
                prefix_end = (cache_start - BDay(1)).date().isoformat()
                needs.append((start, prefix_end))
            if req_end > cache_end:
                suffix_start = max(req_start, cache_end - BDay(self.overlap_bdays))
                needs.append((suffix_start.date().isoformat(), end))

        new_parts = []
        for s, e in needs:
            if pd.Timestamp(s) > pd.Timestamp(e):
                continue
            downloaded = _normalize_ohlcv(self.downloader(symbol, s, e, self.interval, self.auto_adjust))
            info["downloaded_ranges"].append({"start": s, "end": e, "rows": int(len(downloaded))})
            info["download_rows"] += int(len(downloaded))
            if not downloaded.empty:
                new_parts.append(downloaded)

        merged = self._merge(cached, new_parts)
        if not merged.empty:
            saved_fmt = self._save_cached_df(symbol, merged)
            now_iso = pd.Timestamp.now("UTC").isoformat()
            entry = {
                "provider": self.provider,
                "symbol": symbol,
                "interval": self.interval,
                "auto_adjust": self.auto_adjust,
                "schema_version": self.schema_version,
                "range_start": str(merged.index.min().date()),
                "range_end": str(merged.index.max().date()),
                "rows": int(len(merged)),
                "last_success_update": now_iso,
                "format": saved_fmt,
            }
            self._write_manifest_entry(symbol, entry)
            info.update(entry)
            info["cache_format"] = saved_fmt
        else:
            info.update({
                "range_start": None,
                "range_end": None,
                "rows": 0,
                "last_success_update": None,
            })

        clipped = merged.loc[(merged.index >= req_start) & (merged.index <= req_end)] if not merged.empty else merged
        return CacheFetchResult(clipped, info)

    def fetch_many(self, symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
        data: dict[str, pd.DataFrame] = {}
        infos: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            r = self.fetch_symbol(sym, start, end)
            if isinstance(r.data, pd.DataFrame) and not r.data.empty:
                data[sym] = r.data
            infos[sym] = r.info
        return data, infos
