import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pandas.tseries.offsets import BDay

from backtest_engine import atr, compute_trade_stats, pct_return, rsi, run_backtest, sma
from price_cache import MarketDataCache


SPLITS = [
    ("in_sample", "2011-01-01", "2018-12-31"),
    ("oos_2019_2022", "2019-01-01", "2022-12-31"),
    ("oos_2023_plus", "2023-01-01", None),
]

SENS_GRID = {
    "max_breakout_extension_atr": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "pullback_sma_tolerance_atr": [0.25, 0.5, 0.75, 1.0],
    "pullback_invalidation_atr": [0.5, 1.0, 1.5],
    "vcp_atr_ratio_max": [0.70, 0.80, 0.90, 1.00],
    "vcp_range_frac_max": [0.04, 0.06, 0.08, 0.10],
    "vcp_close_pos_min": [0.65, 0.70, 0.75, 0.80],
    "rs63_min": [0.00, 0.025, 0.05],
    "min_rel_volume": [0.8, 1.0, 1.2, 1.5],
}

ENTRY_MODE_VARIANTS = [
    {"id": "mode_breakout_only", "family": "mode", "label": "breakout_only", "overrides": {"entry_mode": "breakout_only"}},
    {"id": "mode_pullback_only", "family": "mode", "label": "pullback_only", "overrides": {"entry_mode": "pullback_only", "enable_pullback_entry": True}},
    {"id": "mode_vcp_only", "family": "mode", "label": "vcp_only", "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True}},
    {
        "id": "mode_breakout_plus_pullback",
        "family": "mode",
        "label": "breakout_plus_pullback",
        "overrides": {"entry_mode": "breakout_plus_pullback", "enable_pullback_entry": True},
    },
    {
        "id": "mode_breakout_plus_vcp",
        "family": "mode",
        "label": "breakout_plus_vcp",
        "overrides": {"entry_mode": "breakout_plus_vcp", "enable_vcp_entry": True},
    },
]


@dataclass
class RunResult:
    universe: str
    split: str
    variant_id: str
    variant_label: str
    family: str
    summary: dict[str, Any]
    trades_df: pd.DataFrame
    survivorship_biased: bool


def load_sp500_symbols() -> list[str]:
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    syms = tables[0]["Symbol"].astype(str).str.strip().tolist()
    return [s.replace(".", "-") for s in syms]


def load_nasdaq100_symbols() -> list[str]:
    header_aliases = {
        "ticker",
        "symbol",
        "ticker symbol",
        "ticker symbols",
        "ticker s",
        "symbol s",
        "nasdaq symbol",
    }

    def _norm_header(col: Any) -> str:
        if isinstance(col, tuple):
            col = " ".join(str(part) for part in col if str(part).strip())
        txt = str(col).strip().lower()
        txt = re.sub(r"[^a-z0-9]+", " ", txt)
        return " ".join(txt.split())

    def _table_diag(tables: list[pd.DataFrame]) -> str:
        parts: list[str] = []
        for i, t in enumerate(tables):
            cols = [str(c) for c in t.columns]
            parts.append(f"#{i}: rows={len(t)}, cols={cols}")
        return "; ".join(parts) if parts else "no tables parsed"

    def _extract_symbols(df: pd.DataFrame) -> list[str]:
        normalized_cols = {_norm_header(c): c for c in df.columns}
        match_key = next((k for k in header_aliases if k in normalized_cols), None)
        if match_key is None:
            return []
        col = normalized_cols[match_key]
        syms = (
            df[col]
            .astype(str)
            .str.replace(r"\[[^\]]+\]", "", regex=True)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        syms = [s for s in syms.tolist() if s]
        unique_syms = sorted(set(syms))
        return unique_syms

    def _validate_symbols(syms: list[str], source: str, tables: list[pd.DataFrame]) -> list[str]:
        if not syms:
            raise ValueError(f"Nasdaq-100 parsing failed from {source}: symbol list is empty. Detected tables: {_table_diag(tables)}")
        invalid = [s for s in syms if not re.fullmatch(r"[A-Z0-9-]+", s)]
        if invalid:
            bad = ", ".join(invalid[:10])
            raise ValueError(f"Nasdaq-100 parsing failed from {source}: invalid Yahoo symbols: {bad}")
        if not (90 <= len(syms) <= 120):
            raise ValueError(
                f"Nasdaq-100 parsing failed from {source}: implausible unique symbol count={len(syms)} (expected 90..120). "
                f"Detected tables: {_table_diag(tables)}"
            )
        return syms

    source_urls = [
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    ]
    all_failures: list[str] = []

    for url in source_urls:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        html_text = resp.text
        all_tables = pd.read_html(StringIO(html_text))

        try:
            preferred_tables = pd.read_html(StringIO(html_text), attrs={"id": "constituents"})
        except ValueError:
            preferred_tables = []

        if preferred_tables:
            if len(preferred_tables) > 1:
                all_failures.append(
                    f"{url}: multiple tables matched id='constituents' ({len(preferred_tables)}); tables: {_table_diag(all_tables)}"
                )
                continue
            syms = _extract_symbols(preferred_tables[0])
            try:
                return _validate_symbols(syms, f"{url} table id=constituents", all_tables)
            except ValueError as exc:
                all_failures.append(str(exc))
                continue

        stable_candidates: list[tuple[int, list[str]]] = []
        for i, t in enumerate(all_tables):
            normalized_cols = {_norm_header(c) for c in t.columns}
            has_company_col = any(c in {"company", "company name", "name", "security"} for c in normalized_cols)
            syms = _extract_symbols(t)
            if has_company_col and syms:
                stable_candidates.append((i, syms))

        plausible = [(i, s) for i, s in stable_candidates if 90 <= len(s) <= 120]
        if len(plausible) == 1:
            return _validate_symbols(plausible[0][1], f"{url} stable fallback table #{plausible[0][0]}", all_tables)
        if len(plausible) > 1:
            all_failures.append(
                f"{url}: multiple plausible stable fallback tables={ [i for i, _ in plausible] }; tables: {_table_diag(all_tables)}"
            )
            continue

        all_failures.append(
            f"{url}: no valid constituents table with accepted headers "
            "['ticker','symbol','ticker symbol','ticker symbols','ticker(s)','symbol(s)','nasdaq symbol']; "
            f"tables: {_table_diag(all_tables)}"
        )

    raise ValueError("Nasdaq-100 constituents parsing failed across sources. " + " | ".join(all_failures))


def load_pit_csv(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"split", "universe", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PIT-CSV muss Spalten enthalten: {sorted(required)}; fehlt: {sorted(missing)}")
    out = df.copy()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["universe"] = out["universe"].astype(str).str.strip().str.lower()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return out


def symbols_for(universe: str, split: str, pit_df: pd.DataFrame | None) -> tuple[list[str], bool]:
    if pit_df is not None:
        mask = pit_df["split"].isin([split, "all"]) & pit_df["universe"].isin([universe, "all"])
        syms = sorted(pit_df.loc[mask, "symbol"].dropna().unique().tolist())
        if syms:
            return syms, False
    if universe == "sp500":
        return load_sp500_symbols(), True
    return load_nasdaq100_symbols(), True


def warmup_start_for(split_start: str, warmup_bdays: int) -> str:
    return (pd.Timestamp(split_start) - BDay(int(warmup_bdays))).date().isoformat()


def split_end_for(split_name: str, as_of: str) -> str:
    if split_name == "oos_2023_plus":
        return as_of
    for s_name, _, s_end in SPLITS:
        if s_name == split_name:
            return str(s_end)
    raise ValueError(f"Unknown split: {split_name}")


def download_data(cache: MarketDataCache, symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    return cache.fetch_many(symbols, start, end)


def _normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.set_index("Date")
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_convert(None)
    d.index = pd.to_datetime(d.index.date)
    d = d.sort_index()
    d = d[~d.index.duplicated(keep="last")]
    return d


def prepare_base_features(raw_data: dict[str, pd.DataFrame], feature_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prepared = {s: _normalize_daily_df(df) for s, df in raw_data.items()}
    atr_period = int(feature_cfg.get("atr_period", 14))
    breakout_lookback = int(feature_cfg.get("breakout_lookback", 55))
    mom_n = int(feature_cfg.get("mom_lookback", 126))
    rsi_p = int(feature_cfg.get("rsi_period", 14))
    sma_regime = int(feature_cfg.get("sma_regime", 200))

    for s, df in prepared.items():
        out = df.copy()
        out["ATR"] = atr(out, atr_period)
        out["ATR10"] = atr(out, 10)
        out["ATR50"] = atr(out, 50)
        out["ATR10_50_Ratio"] = out["ATR10"] / (out["ATR50"] + 1e-12)
        out["ATR10_50_RatioPrev"] = out["ATR10_50_Ratio"].shift(1)
        out["ATR10_50_RatioPrev2"] = out["ATR10_50_Ratio"].shift(2)
        out["SMA_regime"] = sma(out["Close"], sma_regime)
        out["SMA20"] = sma(out["Close"], 20)
        out["SMA50"] = sma(out["Close"], 50)
        out["SMA200"] = sma(out["Close"], 200)
        out["PivotClose"] = out["Close"].shift(1).rolling(breakout_lookback).max()
        out["HH"] = out["High"].shift(1).rolling(breakout_lookback).max()
        out["LL"] = out["Low"].shift(1).rolling(breakout_lookback).min()
        out["RangeLow10"] = out["Low"].shift(1).rolling(10).min()
        out["RangeLow20"] = out["Low"].shift(1).rolling(20).min()
        out["RangeHigh10"] = out["High"].shift(1).rolling(10).max()
        out["DollarVol"] = out["Close"] * out["Volume"]
        out["Mom"] = pct_return(out["Close"], mom_n)
        out["VolSMA50"] = sma(out["Volume"], 50)
        out["VolSMA20"] = sma(out["Volume"], 20)
        out["RelVol"] = out["Volume"] / (out["VolSMA50"] + 1e-12)
        out["VolContracting20_50"] = out["VolSMA20"] < out["VolSMA50"]
        out["RSI"] = rsi(out["Close"], rsi_p)
        prepared[s] = out
    return prepared


def augment_relative_strength(prepared: dict[str, pd.DataFrame], benchmark: str) -> dict[str, pd.DataFrame]:
    if benchmark not in prepared:
        return {k: v.copy() for k, v in prepared.items()}
    bench = prepared[benchmark]["Close"]
    out = {}
    for s, df in prepared.items():
        d = df.copy()
        d["RS63"] = pct_return(d["Close"], 63) - pct_return(bench, 63)
        d["RS126"] = pct_return(d["Close"], 126) - pct_return(bench, 126)
        out[s] = d
    return out


def summarize_window(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    if equity_df is None or equity_df.empty:
        base = compute_trade_stats(pd.DataFrame())
        return {
            "initial_cash": float(initial_cash),
            "final_equity": float(initial_cash),
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe_approx": np.nan,
            "Sortino_approx": np.nan,
            "MaxDrawdown": np.nan,
            "Exposure": np.nan,
            "Turnover": np.nan,
            "Trades": 0,
            **base,
        }
    eq = equity_df["Equity"].astype(float)
    ret = eq.pct_change().fillna(0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(1, len(eq) - 1)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (vol + 1e-12)
    downside = ret[ret < 0]
    sortino = (ret.mean() * 252) / ((downside.std() * np.sqrt(252)) + 1e-12)
    max_dd = (eq / eq.cummax() - 1).min()
    exp_series = equity_df["Exposure"].dropna() if "Exposure" in equity_df.columns else pd.Series(dtype=float)
    exposure = float(exp_series.mean()) if not exp_series.empty else np.nan
    turnover = np.nan
    if trades_df is not None and not trades_df.empty:
        notional = (trades_df["entry_px"] * trades_df["shares"]).abs() + (trades_df["exit_px"] * trades_df["shares"]).abs()
        avg_equity = float(eq.mean()) if len(eq) else np.nan
        if np.isfinite(avg_equity) and avg_equity > 0:
            years = max(len(eq) / 252.0, 1.0 / 252.0)
            turnover = float((notional.sum() / avg_equity) / years)
    trade_stats = compute_trade_stats(trades_df if trades_df is not None else pd.DataFrame())
    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(eq.iloc[-1]),
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe_approx": float(sharpe),
        "Sortino_approx": float(sortino),
        "MaxDrawdown": float(max_dd),
        "Exposure": exposure,
        "Turnover": turnover,
        "Trades": int(0 if trades_df is None else len(trades_df)),
        **trade_stats,
    }


def calc_benchmark_summary(bench_df: pd.DataFrame) -> dict[str, float]:
    close_col = "Adj Close" if "Adj Close" in bench_df.columns else "Close"
    close = bench_df[close_col].dropna()
    if close.empty:
        return {"Benchmark_CAGR": np.nan, "Benchmark_MaxDrawdown": np.nan, "Benchmark_Sharpe": np.nan, "Benchmark_Sortino": np.nan}
    ret = close.pct_change().fillna(0)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (252 / max(1, len(close) - 1)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (vol + 1e-12)
    downside = ret[ret < 0]
    sortino = (ret.mean() * 252) / ((downside.std() * np.sqrt(252)) + 1e-12)
    dd = (close / close.cummax() - 1).min()
    return {
        "Benchmark_CAGR": float(cagr),
        "Benchmark_MaxDrawdown": float(dd),
        "Benchmark_Sharpe": float(sharpe),
        "Benchmark_Sortino": float(sortino),
    }


def build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"id": "baseline", "family": "baseline", "label": "Baseline", "overrides": {"entry_mode": "breakout_only"}},
        *ENTRY_MODE_VARIANTS,
    ]
    for v in SENS_GRID["max_breakout_extension_atr"]:
        variants.append({"id": f"sens_ext_{v:.2f}", "family": "sens_ext", "label": f"Ext {v:.2f}", "overrides": {"entry_mode": "breakout_only", "max_breakout_extension_atr": float(v)}})
    for v in SENS_GRID["pullback_sma_tolerance_atr"]:
        variants.append({"id": f"sens_pb_tol_{v:.2f}", "family": "sens_pullback_tol", "label": f"PB Tol {v:.2f}", "overrides": {"entry_mode": "pullback_only", "enable_pullback_entry": True, "pullback_sma_tolerance_atr": float(v), "pullback_range_tolerance_atr": float(v)}})
    for v in SENS_GRID["pullback_invalidation_atr"]:
        variants.append({"id": f"sens_pb_inv_{v:.2f}", "family": "sens_pullback_inv", "label": f"PB Inv {v:.2f}", "overrides": {"entry_mode": "pullback_only", "enable_pullback_entry": True, "pullback_invalidation_atr": float(v)}})
    for v in SENS_GRID["vcp_atr_ratio_max"]:
        variants.append({"id": f"sens_vcp_ratio_{v:.2f}", "family": "sens_vcp_ratio", "label": f"VCP Ratio {v:.2f}", "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_atr_ratio_max": float(v)}})
    for v in SENS_GRID["vcp_range_frac_max"]:
        variants.append({"id": f"sens_vcp_range_{v:.2f}", "family": "sens_vcp_range", "label": f"VCP Range {v:.2f}", "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_range_frac_max": float(v)}})
    for v in SENS_GRID["vcp_close_pos_min"]:
        variants.append({"id": f"sens_vcp_close_{v:.2f}", "family": "sens_vcp_close", "label": f"VCP Close {v:.2f}", "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_close_pos_min": float(v)}})
    for v in SENS_GRID["rs63_min"]:
        variants.append({"id": f"sens_rs_{v:.3f}", "family": "sens_rs", "label": f"RS {v:.1%}", "overrides": {"entry_mode": "breakout_only", "enable_relative_strength_filter": True, "rs63_min": float(v), "rs126_min": float(v)}})
    for v in SENS_GRID["min_rel_volume"]:
        variants.append({"id": f"sens_relvol_{v:.2f}", "family": "sens_relvol", "label": f"RelVol {v:.2f}", "overrides": {"entry_mode": "breakout_only", "min_rel_volume": float(v)}})
    return variants


def read_baseline_config(path: Path) -> tuple[dict[str, Any], str, str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    return cfg, str(path), cfg_hash


def split_data_for_period(data_all: dict[str, pd.DataFrame], start: str, end: str) -> dict[str, pd.DataFrame]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = {}
    for k, df in data_all.items():
        idx = df.index
        ws = s.tz_localize(idx.tz) if getattr(idx, "tz", None) is not None else s
        we = e.tz_localize(idx.tz) if getattr(idx, "tz", None) is not None else e
        w = df.loc[(idx >= ws) & (idx <= we)].copy()
        if not w.empty:
            out[k] = w
    return out


def filter_trades_from_start(trades_df: pd.DataFrame, split_start: str) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    entry = pd.to_datetime(trades_df["entry_date"], errors="coerce")
    return trades_df.loc[entry >= pd.Timestamp(split_start)].copy()


def indicator_ready_fraction(data_window: dict[str, pd.DataFrame], symbols: list[str], split_start: str, cfg: dict[str, Any]) -> float:
    s_ts = pd.Timestamp(split_start)
    ok = 0
    total = 0
    sma_n = int(cfg.get("sma_regime", 200))
    mom_n = int(cfg.get("mom_lookback", 126))
    for sym in symbols:
        if sym not in data_window:
            continue
        df = data_window[sym]
        total += 1
        close = df["Close"]
        s = close.rolling(sma_n, min_periods=sma_n).mean()
        m = close / close.shift(mom_n) - 1
        ts = s_ts.tz_localize(df.index.tz) if getattr(df.index, "tz", None) is not None else s_ts
        if ts in df.index and np.isfinite(s.loc[ts]) and np.isfinite(m.loc[ts]):
            ok += 1
    return float(ok / total) if total > 0 else np.nan


def passes_selection_rules(row: pd.Series, base: pd.Series, min_trades: int) -> bool:
    trades_ok = int(row.get("Trades", 0)) >= int(min_trades)
    exp = row.get("Expectancy_R", np.nan)
    pf = row.get("ProfitFactor", np.nan)
    sortino = row.get("Sortino_approx", np.nan)
    maxdd = row.get("MaxDrawdown", np.nan)
    b_sortino = base.get("Sortino_approx", np.nan)
    b_dd = base.get("MaxDrawdown", np.nan)
    return bool(
        trades_ok
        and np.isfinite(exp) and float(exp) > 0
        and np.isfinite(pf) and float(pf) > 1.0
        and np.isfinite(sortino) and np.isfinite(b_sortino) and float(sortino) > float(b_sortino)
        and np.isfinite(maxdd) and np.isfinite(b_dd) and float(maxdd) >= (float(b_dd) - 0.02)
    )


def select_is_winners(summary_df: pd.DataFrame, universe: str, min_trades: int) -> set[str]:
    is_df = summary_df[(summary_df["universe"] == universe) & (summary_df["split"] == "in_sample")]
    if is_df.empty:
        return set()
    baseline = is_df[is_df["variant_id"] == "baseline"]
    if baseline.empty:
        return set()
    b = baseline.iloc[0]
    selected = set()
    for fam in sorted([x for x in is_df["family"].dropna().unique() if x != "baseline"]):
        fam_df = is_df[is_df["family"] == fam].copy()
        fam_df["passes"] = fam_df.apply(lambda r: passes_selection_rules(r, b, min_trades), axis=1)
        valid = fam_df[fam_df["passes"]]
        if valid.empty:
            continue
        valid = valid.sort_values(["Sortino_approx", "Expectancy_R", "ProfitFactor"], ascending=False)
        selected.add(str(valid.iloc[0]["variant_id"]))
    return selected


def evaluate_oos_confirmation(summary_df: pd.DataFrame, min_trades: int, smoke_test: bool) -> pd.DataFrame:
    rows = []
    for universe in sorted(summary_df["universe"].dropna().unique()):
        uni_df = summary_df[summary_df["universe"] == universe]
        baseline = uni_df[uni_df["variant_id"] == "baseline"]
        for vid in sorted([v for v in uni_df["variant_id"].dropna().unique() if v != "baseline"]):
            vdf = uni_df[uni_df["variant_id"] == vid]
            both_pass = not smoke_test
            if both_pass:
                for split in ["oos_2019_2022", "oos_2023_plus"]:
                    b = baseline[baseline["split"] == split]
                    v = vdf[vdf["split"] == split]
                    if b.empty or v.empty or (not passes_selection_rules(v.iloc[0], b.iloc[0], min_trades=min_trades)):
                        both_pass = False
                        break
            rows.append({"universe": universe, "variant_id": vid, "confirmed_in_both_oos": bool(both_pass)})
    return pd.DataFrame(rows)


def stable_regions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in sorted(df["family"].dropna().unique()):
        fam_df = df[df["family"] == fam].copy()
        if fam_df.empty:
            continue
        top = fam_df.sort_values("Sortino_approx", ascending=False).head(3)
        rows.append({
            "family": fam,
            "top_variant_ids": ", ".join(top["variant_id"].astype(str).tolist()),
            "top_sortino_range": f"{top['Sortino_approx'].min():.3f}..{top['Sortino_approx'].max():.3f}" if not top.empty else "-",
            "top_expectancy_range": f"{top['Expectancy_R'].min():.3f}..{top['Expectancy_R'].max():.3f}" if not top.empty else "-",
        })
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def run_one_variant(
    universe: str,
    split: str,
    split_start: str,
    split_end: str,
    warmup_start: str,
    benchmark: str,
    symbols: list[str],
    variant: dict[str, Any],
    prepared_split_data: dict[str, pd.DataFrame],
    baseline_cfg: dict[str, Any],
    baseline_path: str,
    baseline_hash: str,
    survivorship_biased: bool,
    as_of: str,
) -> RunResult:
    cfg = dict(baseline_cfg)
    cfg.update(variant["overrides"])
    available = set(prepared_split_data.keys())
    symbols_run = [s for s in symbols if s in available and s != benchmark]
    inv = cfg.get("inverse_map", {}) if isinstance(cfg.get("inverse_map", {}), dict) else {}
    inv = {k: v for k, v in inv.items() if v in available}
    cfg.update({
        "start": warmup_start,
        "end": split_end,
        "symbols": symbols_run,
        "regime_symbol": benchmark,
        "inverse_map": inv,
        "features_precomputed": True,
    })

    if benchmark not in available or not symbols_run:
        eq = pd.DataFrame()
        trades = pd.DataFrame()
    else:
        variant_input = {s: df.copy() for s, df in prepared_split_data.items()}
        eq, trades, _, _ = run_backtest(variant_input, cfg)

    split_ts = pd.Timestamp(split_start)
    eq_eval = eq.loc[eq.index >= split_ts].copy() if not eq.empty else pd.DataFrame()
    trades_eval = filter_trades_from_start(trades, split_start)

    summary = summarize_window(eq_eval, trades_eval, float(cfg.get("initial_cash", 0.0)))
    bdf = prepared_split_data.get(benchmark, pd.DataFrame())
    b_eval = bdf.loc[bdf.index >= split_ts] if not bdf.empty else pd.DataFrame()
    summary.update(calc_benchmark_summary(b_eval))
    summary["last_trading_day"] = str(b_eval.index.max().date()) if not b_eval.empty else None
    summary["split_start"] = split_start
    summary["split_end"] = split_end
    summary["warmup_start"] = warmup_start
    summary["as_of"] = as_of
    summary["baseline_config_path"] = baseline_path
    summary["baseline_config_hash"] = baseline_hash
    summary["survivorship_biased"] = bool(survivorship_biased)
    summary["entry_mode"] = cfg.get("entry_mode", "legacy")
    summary["signal_assumption"] = "signals_on_close_entry_next_open"
    summary["cost_assumption"] = f"spread_bps_per_side={cfg.get('spread_bps_per_side', 'n/a')}"
    summary["data_source"] = "yfinance"
    summary["warmup_indicator_ready_fraction"] = indicator_ready_fraction(prepared_split_data, symbols_run, split_start, cfg)

    return RunResult(universe, split, variant["id"], variant["label"], variant["family"], summary, trades_eval, survivorship_biased)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduzierbarer Research-/Backtest-Workflow")
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--pit-constituents-csv", default=None)
    parser.add_argument("--universe", choices=["sp500", "nasdaq100", "both"], default="both")
    parser.add_argument("--as-of", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--warmup-bdays", type=int, default=300)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--baseline-config", default=str((Path(__file__).parent / "config_best_2011_2026.json")))
    parser.add_argument("--cache-dir", default=str(Path.home() / ".zero_swing_cache/market_data/v1"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-overlap-bdays", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-research", action="store_true")
    parser.add_argument("--max-symbols-per-universe", type=int, default=25)
    args = parser.parse_args()

    if not args.smoke_test and not args.full_research:
        args.smoke_test = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pit_df = load_pit_csv(args.pit_constituents_csv)
    universes = ["sp500", "nasdaq100"] if args.universe == "both" else [args.universe]

    baseline_cfg, baseline_path, baseline_hash = read_baseline_config(Path(args.baseline_config))
    variants = build_variants()
    cache = MarketDataCache(
        cache_dir=args.cache_dir,
        provider="yfinance",
        interval="1d",
        auto_adjust=False,
        schema_version="research_v1",
        use_cache=not bool(args.no_cache),
        refresh=bool(args.refresh),
        overlap_bdays=int(args.cache_overlap_bdays),
    )

    universe_symbols: dict[str, list[str]] = {}
    survivorship_by_uni: dict[str, bool] = {}
    all_syms = {"SPY", "QQQ"}
    run_notes: list[str] = []

    for uni in universes:
        syms_union = set()
        surv = False
        for split_name, _, _ in SPLITS:
            syms, bias = symbols_for(uni, split_name, pit_df)
            syms_union.update(syms)
            surv = surv or bool(bias)
        syms_sorted = sorted(syms_union)
        if args.smoke_test:
            syms_sorted = syms_sorted[: int(args.max_symbols_per_universe)]
        universe_symbols[uni] = syms_sorted
        survivorship_by_uni[uni] = surv
        all_syms.update(syms_sorted)

    full_start = warmup_start_for(SPLITS[0][1], int(args.warmup_bdays))
    data_all, cache_infos = download_data(cache, sorted(all_syms), full_start, args.as_of)

    cache_rows = []
    for sym, info in cache_infos.items():
        cache_rows.append({
            "symbol": sym,
            "cache_hit": bool(info.get("cache_hit", False)),
            "download_rows": int(info.get("download_rows", 0) or 0),
            "range_start": info.get("range_start"),
            "range_end": info.get("range_end"),
            "last_success_update": info.get("last_success_update"),
            "cache_format": info.get("cache_format"),
            "downloaded_ranges": json.dumps(info.get("downloaded_ranges", []), ensure_ascii=False),
        })

    missing = sorted(all_syms - set(data_all.keys()))
    if missing:
        run_notes.append(f"{len(missing)} Symbole ohne Daten (Beispiele: {', '.join(missing[:10])})")

    base_features_all = prepare_base_features(data_all, baseline_cfg)

    all_results: list[RunResult] = []
    all_trades: list[pd.DataFrame] = []

    for uni in universes:
        bench = "SPY" if uni == "sp500" else "QQQ"
        syms = [s for s in universe_symbols.get(uni, []) if s in base_features_all]
        if bench not in base_features_all or not syms:
            run_notes.append(f"{uni}: Benchmark oder Symbole fehlen, Universe übersprungen.")
            continue

        split_rows = []
        for split_name, split_start, split_end_static in SPLITS:
            split_end = split_end_for(split_name, args.as_of) if split_end_static is None else str(split_end_static)
            warmup_start = warmup_start_for(split_start, int(args.warmup_bdays))
            split_raw = split_data_for_period(base_features_all, warmup_start, split_end)
            split_prepared = augment_relative_strength(split_raw, benchmark=bench)

            run_variants = variants if split_name == "in_sample" else [{"id": "baseline", "family": "baseline", "label": "Baseline", "overrides": {"entry_mode": "breakout_only"}}]
            for variant in run_variants:
                res = run_one_variant(
                    universe=uni,
                    split=split_name,
                    split_start=split_start,
                    split_end=split_end,
                    warmup_start=warmup_start,
                    benchmark=bench,
                    symbols=syms,
                    variant=variant,
                    prepared_split_data=split_prepared,
                    baseline_cfg=baseline_cfg,
                    baseline_path=baseline_path,
                    baseline_hash=baseline_hash,
                    survivorship_biased=survivorship_by_uni.get(uni, True),
                    as_of=args.as_of,
                )
                all_results.append(res)
                split_rows.append({"universe": res.universe, "split": res.split, "variant_id": res.variant_id, "family": res.family, **res.summary})
                if res.trades_df is not None and not res.trades_df.empty:
                    t = res.trades_df.copy()
                    t.insert(0, "variant_id", res.variant_id)
                    t.insert(0, "split", res.split)
                    t.insert(0, "universe", res.universe)
                    t.insert(0, "survivorship_biased", bool(res.survivorship_biased))
                    all_trades.append(t)

            if split_name == "in_sample":
                temp_df = pd.DataFrame(split_rows)
                selected_ids = set()
                if args.full_research:
                    selected_ids = select_is_winners(temp_df, universe=uni, min_trades=int(args.min_trades))
                selected_variants = [v for v in variants if v["id"] in selected_ids]

                for oos_name, oos_start, oos_end_static in SPLITS[1:]:
                    oos_end = split_end_for(oos_name, args.as_of) if oos_end_static is None else str(oos_end_static)
                    oos_warmup = warmup_start_for(oos_start, int(args.warmup_bdays))
                    oos_raw = split_data_for_period(base_features_all, oos_warmup, oos_end)
                    oos_prepared = augment_relative_strength(oos_raw, benchmark=bench)
                    for v in selected_variants:
                        res = run_one_variant(
                            universe=uni,
                            split=oos_name,
                            split_start=oos_start,
                            split_end=oos_end,
                            warmup_start=oos_warmup,
                            benchmark=bench,
                            symbols=syms,
                            variant=v,
                            prepared_split_data=oos_prepared,
                            baseline_cfg=baseline_cfg,
                            baseline_path=baseline_path,
                            baseline_hash=baseline_hash,
                            survivorship_biased=survivorship_by_uni.get(uni, True),
                            as_of=args.as_of,
                        )
                        all_results.append(res)
                        if res.trades_df is not None and not res.trades_df.empty:
                            t = res.trades_df.copy()
                            t.insert(0, "variant_id", res.variant_id)
                            t.insert(0, "split", res.split)
                            t.insert(0, "universe", res.universe)
                            t.insert(0, "survivorship_biased", bool(res.survivorship_biased))
                            all_trades.append(t)

    summary_rows = []
    for r in all_results:
        row = {"universe": r.universe, "split": r.split, "variant_id": r.variant_id, "variant_label": r.variant_label, "family": r.family, "survivorship_biased": bool(r.survivorship_biased)}
        row.update(r.summary)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=["universe", "split", "variant_id", "variant_label", "family", "survivorship_biased", "Trades", "CAGR", "MaxDrawdown", "Sharpe_approx", "Sortino_approx", "ProfitFactor", "Expectancy_R", "MedianMFE_R", "MedianMAE_R", "Exposure", "Turnover", "confirmed_in_both_oos"])
    summary_df.to_csv(out_dir / "research_summary.csv", index=False)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["universe", "split", "variant_id", "survivorship_biased", "symbol", "side", "entry_date", "entry_px", "exit_date", "exit_px", "shares", "pnl", "reason", "setup", "R_multiple", "MFE_R", "MAE_R", "holding_days"])
    trades_df.to_csv(out_dir / "research_trades.csv", index=False)

    confirm_df = evaluate_oos_confirmation(summary_df, min_trades=int(args.min_trades), smoke_test=bool(args.smoke_test)) if not summary_df.empty else pd.DataFrame()
    if confirm_df.empty:
        confirm_df = pd.DataFrame(columns=["universe", "variant_id", "confirmed_in_both_oos"])
    confirm_df.to_csv(out_dir / "optimization_confirmation.csv", index=False)

    cache_df = pd.DataFrame(cache_rows)
    if cache_df.empty:
        cache_df = pd.DataFrame(columns=["symbol", "cache_hit", "download_rows", "range_start", "range_end", "last_success_update", "cache_format", "downloaded_ranges"])
    cache_df.to_csv(out_dir / "cache_usage.csv", index=False)

    is_sens = summary_df[summary_df["split"] == "in_sample"].copy() if not summary_df.empty else pd.DataFrame()
    stab_df = stable_regions(is_sens) if not is_sens.empty else pd.DataFrame()

    md_lines = [
        "# Reproduzierbarer Research-/Backtest-Report",
        "",
        "## Laufmodus",
        "SMOKE TEST — NOT A PERFORMANCE VALIDATION" if args.smoke_test else "FULL RESEARCH",
        "",
        "## Laufparameter",
        f"- as_of: `{args.as_of}`",
        f"- warmup_bdays: `{args.warmup_bdays}`",
        f"- min_trades: `{args.min_trades}`",
        f"- baseline_config: `{baseline_path}`",
        f"- baseline_config_hash: `{baseline_hash}`",
        f"- cache_dir: `{args.cache_dir}`",
        f"- cache_enabled: `{not bool(args.no_cache)}`",
        f"- refresh: `{bool(args.refresh)}`",
        f"- cache_overlap_bdays: `{args.cache_overlap_bdays}`",
        "",
        "## Survivorship-Bias und Datenlimitierung",
        "- Belastbare Tests benötigen **PIT-Mitgliedschaften** und **historische delistingbereinigte Preisreihen**.",
        "- Läufe mit aktuellen Wikipedia-Komponenten + Yahoo-Daten sind `survivorship_biased=true` und **kein belastbarer Performance-Nachweis**.",
        "",
        "## Laufnotizen",
    ]
    if run_notes:
        md_lines.extend([f"- {n}" for n in run_notes])
    else:
        md_lines.append("- Keine besonderen Laufnotizen.")

    md_lines += ["", "## Ergebnisübersicht (Universe/Split/Variante)"]
    if summary_df.empty:
        md_lines.append("Keine auswertbaren Ergebnisse erzeugt.")
    else:
        keep_cols = [
            "universe", "split", "variant_id", "entry_mode", "survivorship_biased",
            "Trades", "WinRate", "AvgWin_R", "AvgLoss_R", "ProfitFactor", "Expectancy_R",
            "CAGR", "MaxDrawdown", "Sharpe_approx", "Sortino_approx", "Exposure", "Turnover",
            "MedianHoldDays", "MedianMFE_R", "MedianMAE_R", "Benchmark_CAGR", "Benchmark_Sortino",
            "warmup_indicator_ready_fraction", "last_trading_day",
        ]
        cols = [c for c in keep_cols if c in summary_df.columns]
        md_lines.append(dataframe_to_markdown(summary_df[cols]))

    md_lines += ["", "## OOS-Bestätigung", "Variante gilt nur als bestätigt, wenn beide OOS-Phasen die Akzeptanzregeln erfüllen."]
    md_lines.append(dataframe_to_markdown(confirm_df) if not confirm_df.empty else "Keine bestätigbaren Varianten.")

    md_lines += ["", "## Sensitivitätsstabilität (In-Sample, Top-Nachbarn je Familie)"]
    md_lines.append(dataframe_to_markdown(stab_df) if not stab_df.empty else "Keine Sensitivitätsdaten verfügbar.")

    md_lines += ["", "## Cache-Nutzung"]
    if cache_df.empty:
        md_lines.append("Keine Cache-Metadaten verfügbar.")
    else:
        keep = ["symbol", "cache_hit", "download_rows", "range_start", "range_end", "last_success_update", "cache_format", "downloaded_ranges"]
        md_lines.append(dataframe_to_markdown(cache_df[keep]))

    md_lines += ["", "## Feature-Preparation"]
    md_lines.append("Basisfeatures (ATR/SMA/Momentum/Range/Volumen) werden pro Symbol vor Variantenbewertung einmal vorbereitet; Varianten nutzen nur Signal-/Filter-/Stop-Parameter und erhalten defensive Kopien, um Mutation zwischen Varianten zu vermeiden.")

    (out_dir / "research_report.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
