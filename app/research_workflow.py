import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from backtest_engine import compute_trade_stats, run_backtest


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
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    syms = tables[0]["Symbol"].astype(str).str.strip().tolist()
    return [s.replace(".", "-") for s in syms]


def load_nasdaq100_symbols() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    frame = None
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if "ticker" in cols and len(t) >= 50:
            frame = t
            break
    if frame is None:
        raise ValueError("Nasdaq-100 constituents table with 'Ticker' column not found on source page.")
    sym_col = "Ticker" if "Ticker" in frame.columns else frame.columns[1]
    syms = frame[sym_col].astype(str).str.strip().tolist()
    return [s.replace(".", "-") for s in syms]


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


def _chunked(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _normalize_downloaded(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(x) for x in df.columns.get_level_values(0)]
        lvl1 = [str(x) for x in df.columns.get_level_values(1)]
        prices_first = "Open" in set(lvl0)

        for t in tickers:
            if prices_first:
                cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if (c, t) in df.columns]
                if not cols:
                    continue
                sub = pd.DataFrame({c: df[(c, t)] for c in cols}).dropna(how="all")
            else:
                cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if (t, c) in df.columns]
                if not cols:
                    continue
                sub = pd.DataFrame({c: df[(t, c)] for c in cols}).dropna(how="all")
            if not sub.empty:
                out[t] = sub
        return out

    if len(tickers) == 1:
        out[tickers[0]] = df
    return out


def download_data(symbols: list[str], start: str, end: str, chunk_size: int = 80) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for batch in _chunked(symbols, chunk_size):
        raw = yf.download(
            tickers=batch,
            start=start,
            end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        normalized = _normalize_downloaded(raw, batch)
        out.update({k: v for k, v in normalized.items() if isinstance(v, pd.DataFrame) and not v.empty})
    return out


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
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min()

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


def build_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"id": "baseline", "family": "baseline", "label": "Baseline", "overrides": {"entry_mode": "breakout_only"}},
        *ENTRY_MODE_VARIANTS,
    ]

    for v in SENS_GRID["max_breakout_extension_atr"]:
        variants.append({
            "id": f"sens_ext_{v:.2f}",
            "family": "sens_ext",
            "label": f"Breakout extension <= {v:.2f} ATR",
            "overrides": {"entry_mode": "breakout_only", "max_breakout_extension_atr": float(v)},
        })

    for v in SENS_GRID["pullback_sma_tolerance_atr"]:
        variants.append({
            "id": f"sens_pb_tol_{v:.2f}",
            "family": "sens_pullback_tol",
            "label": f"Pullback tolerance {v:.2f} ATR",
            "overrides": {"entry_mode": "pullback_only", "enable_pullback_entry": True, "pullback_sma_tolerance_atr": float(v), "pullback_range_tolerance_atr": float(v)},
        })

    for v in SENS_GRID["pullback_invalidation_atr"]:
        variants.append({
            "id": f"sens_pb_inv_{v:.2f}",
            "family": "sens_pullback_inv",
            "label": f"Pullback invalidation {v:.2f} ATR",
            "overrides": {"entry_mode": "pullback_only", "enable_pullback_entry": True, "pullback_invalidation_atr": float(v)},
        })

    for v in SENS_GRID["vcp_atr_ratio_max"]:
        variants.append({
            "id": f"sens_vcp_atr_ratio_{v:.2f}",
            "family": "sens_vcp_ratio",
            "label": f"VCP ATR10/ATR50 <= {v:.2f}",
            "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_atr_ratio_max": float(v)},
        })

    for v in SENS_GRID["vcp_range_frac_max"]:
        variants.append({
            "id": f"sens_vcp_range_{v:.2f}",
            "family": "sens_vcp_range",
            "label": f"VCP range <= {v:.2%}",
            "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_range_frac_max": float(v)},
        })

    for v in SENS_GRID["vcp_close_pos_min"]:
        variants.append({
            "id": f"sens_vcp_close_{v:.2f}",
            "family": "sens_vcp_close",
            "label": f"VCP close position >= {v:.2f}",
            "overrides": {"entry_mode": "vcp_only", "enable_vcp_entry": True, "vcp_close_pos_min": float(v)},
        })

    for v in SENS_GRID["rs63_min"]:
        variants.append({
            "id": f"sens_rs_{v:.3f}",
            "family": "sens_rs",
            "label": f"RS63/126 >= {v:.1%}",
            "overrides": {
                "entry_mode": "breakout_only",
                "enable_relative_strength_filter": True,
                "rs63_min": float(v),
                "rs126_min": float(v),
            },
        })

    for v in SENS_GRID["min_rel_volume"]:
        variants.append({
            "id": f"sens_relvol_{v:.2f}",
            "family": "sens_relvol",
            "label": f"RelVol >= {v:.2f}",
            "overrides": {"entry_mode": "breakout_only", "min_rel_volume": float(v)},
        })

    return variants


def read_baseline_config(path: Path) -> tuple[dict[str, Any], str, str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid baseline config JSON: {path}")
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    return cfg, str(path), cfg_hash


def split_data_for_period(data_all: dict[str, pd.DataFrame], start: str, end: str) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = {}
    for s, df in data_all.items():
        d = df.copy()
        idx = d.index
        if getattr(idx, "tz", None) is not None:
            s_ts = start_ts.tz_localize(idx.tz)
            e_ts = end_ts.tz_localize(idx.tz)
        else:
            s_ts = start_ts
            e_ts = end_ts
        w = d.loc[(idx >= s_ts) & (idx <= e_ts)]
        if not w.empty:
            out[s] = w
    return out


def filter_trades_from_start(trades_df: pd.DataFrame, split_start: str) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    s = pd.Timestamp(split_start).date().isoformat()
    entry = pd.to_datetime(trades_df["entry_date"], errors="coerce")
    return trades_df.loc[entry >= pd.Timestamp(s)].copy()


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
        c = df["Close"]
        s = c.rolling(sma_n, min_periods=sma_n).mean()
        m = c / c.shift(mom_n) - 1
        idx = df.index
        check_ts = s_ts.tz_localize(idx.tz) if getattr(idx, "tz", None) is not None else s_ts
        if check_ts in df.index:
            if np.isfinite(s.loc[check_ts]) and np.isfinite(m.loc[check_ts]):
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

    exp_ok = np.isfinite(exp) and float(exp) > 0
    pf_ok = np.isfinite(pf) and float(pf) > 1.0
    sortino_ok = np.isfinite(sortino) and np.isfinite(b_sortino) and float(sortino) > float(b_sortino)
    dd_ok = np.isfinite(maxdd) and np.isfinite(b_dd) and float(maxdd) >= (float(b_dd) - 0.02)
    return bool(trades_ok and exp_ok and pf_ok and sortino_ok and dd_ok)


def select_is_winners(summary_df: pd.DataFrame, universe: str, min_trades: int) -> set[str]:
    is_df = summary_df[(summary_df["universe"] == universe) & (summary_df["split"] == "in_sample")]
    if is_df.empty:
        return set()
    baseline = is_df[is_df["variant_id"] == "baseline"]
    if baseline.empty:
        return set()
    b = baseline.iloc[0]

    selected = set()
    for fam in sorted([x for x in is_df["family"].dropna().unique() if x not in {"baseline"}]):
        fam_df = is_df[is_df["family"] == fam].copy()
        if fam_df.empty:
            continue
        fam_df["passes"] = fam_df.apply(lambda r: passes_selection_rules(r, b, min_trades), axis=1)
        valid = fam_df[fam_df["passes"]]
        if valid.empty:
            continue
        valid = valid.sort_values(["Sortino_approx", "Expectancy_R", "ProfitFactor"], ascending=False)
        selected.add(str(valid.iloc[0]["variant_id"]))
    return selected


def evaluate_oos_confirmation(summary_df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    rows = []
    for universe in sorted(summary_df["universe"].dropna().unique()):
        uni_df = summary_df[summary_df["universe"] == universe]
        baseline = uni_df[uni_df["variant_id"] == "baseline"]
        for vid in sorted([v for v in uni_df["variant_id"].dropna().unique() if v != "baseline"]):
            vdf = uni_df[uni_df["variant_id"] == vid]
            both_pass = True
            for split in ["oos_2019_2022", "oos_2023_plus"]:
                b = baseline[baseline["split"] == split]
                v = vdf[vdf["split"] == split]
                if b.empty or v.empty:
                    both_pass = False
                    break
                if not passes_selection_rules(v.iloc[0], b.iloc[0], min_trades=min_trades):
                    both_pass = False
                    break
            rows.append({"universe": universe, "variant_id": vid, "confirmed_in_both_oos": both_pass})
    return pd.DataFrame(rows)


def stable_regions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in sorted(df["family"].dropna().unique()):
        fam_df = df[df["family"] == fam].copy()
        if fam_df.empty:
            continue
        fam_df = fam_df.sort_values("Sortino_approx", ascending=False)
        top = fam_df.head(3)
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
    data_window: dict[str, pd.DataFrame],
    baseline_cfg: dict[str, Any],
    baseline_path: str,
    baseline_hash: str,
    survivorship_biased: bool,
    as_of: str,
) -> RunResult:
    cfg = dict(baseline_cfg)
    cfg.update(variant["overrides"])
    cfg.update({
        "start": warmup_start,
        "end": split_end,
        "symbols": symbols,
        "regime_symbol": benchmark,
    })

    eq, trades, _, _ = run_backtest(data_window, cfg)
    split_ts = pd.Timestamp(split_start)
    eq_eval = eq.loc[eq.index >= split_ts].copy()
    trades_eval = filter_trades_from_start(trades, split_start)

    summary = summarize_window(eq_eval, trades_eval, initial_cash=float(cfg.get("initial_cash", 0.0)))
    bdf = data_window.get(benchmark, pd.DataFrame())
    if not bdf.empty:
        b_eval = bdf.loc[bdf.index >= split_ts]
        summary.update(calc_benchmark_summary(b_eval))
        summary["last_trading_day"] = str(b_eval.index.max().date()) if not b_eval.empty else None
    else:
        summary.update(calc_benchmark_summary(pd.DataFrame()))
        summary["last_trading_day"] = None

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

    ready = indicator_ready_fraction(data_window, symbols, split_start=split_start, cfg=cfg)
    summary["warmup_indicator_ready_fraction"] = ready

    return RunResult(
        universe=universe,
        split=split,
        variant_id=variant["id"],
        variant_label=variant["label"],
        family=variant["family"],
        summary=summary,
        trades_df=trades_eval,
        survivorship_biased=survivorship_biased,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduzierbarer Research-/Backtest-Workflow")
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--pit-constituents-csv", default=None)
    parser.add_argument("--universe", choices=["sp500", "nasdaq100", "both"], default="both")
    parser.add_argument("--as-of", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--warmup-bdays", type=int, default=300)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--baseline-config", default=str((Path(__file__).parent / "config_best_2011_2026.json")))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pit_df = load_pit_csv(args.pit_constituents_csv)
    universes = ["sp500", "nasdaq100"] if args.universe == "both" else [args.universe]
    variants = build_variants()

    baseline_cfg, baseline_path, baseline_hash = read_baseline_config(Path(args.baseline_config))

    all_results: list[RunResult] = []
    all_trades: list[pd.DataFrame] = []
    run_notes: list[str] = []

    for uni in universes:
        bench = "SPY" if uni == "sp500" else "QQQ"

        syms_union = set()
        survivorship = False
        for split_name, split_start, _ in SPLITS:
            syms, bias = symbols_for(uni, split_name, pit_df)
            syms_union.update(syms)
            survivorship = survivorship or bool(bias)
        symbols_all = sorted(syms_union)

        full_start = warmup_start_for(SPLITS[0][1], int(args.warmup_bdays))
        data_all = download_data(sorted(set(symbols_all + [bench])), full_start, args.as_of)
        symbols_with_data = [s for s in symbols_all if s in data_all]
        missing_symbols = sorted(set(symbols_all + [bench]) - set(data_all.keys()))
        if missing_symbols:
            run_notes.append(f"{uni}: {len(missing_symbols)} Symbole ohne Preisdaten (Beispiele: {', '.join(missing_symbols[:10])})")
        if bench not in data_all or not symbols_with_data:
            run_notes.append(f"{uni}: Benchmark/Universum konnte nicht geladen werden; Ergebnisse ausgelassen.")
            continue

        split_result_rows = []
        for split_name, split_start, split_end_static in SPLITS:
            split_end = split_end_for(split_name, args.as_of) if split_end_static is None else str(split_end_static)
            warmup_start = warmup_start_for(split_start, int(args.warmup_bdays))
            data_window = split_data_for_period(data_all, warmup_start, split_end)

            for variant in variants:
                if split_name != "in_sample":
                    # OOS: run baseline and later selected winners only
                    if variant["id"] != "baseline":
                        continue
                res = run_one_variant(
                    universe=uni,
                    split=split_name,
                    split_start=split_start,
                    split_end=split_end,
                    warmup_start=warmup_start,
                    benchmark=bench,
                    symbols=symbols_with_data,
                    variant=variant,
                    data_window=data_window,
                    baseline_cfg=baseline_cfg,
                    baseline_path=baseline_path,
                    baseline_hash=baseline_hash,
                    survivorship_biased=survivorship,
                    as_of=args.as_of,
                )
                all_results.append(res)
                split_result_rows.append({
                    "universe": res.universe,
                    "split": res.split,
                    "variant_id": res.variant_id,
                    "family": res.family,
                    **res.summary,
                })
                if res.trades_df is not None and not res.trades_df.empty:
                    t = res.trades_df.copy()
                    t.insert(0, "variant_id", res.variant_id)
                    t.insert(0, "split", res.split)
                    t.insert(0, "universe", res.universe)
                    t.insert(0, "survivorship_biased", bool(res.survivorship_biased))
                    all_trades.append(t)

            if split_name == "in_sample":
                temp_df = pd.DataFrame(split_result_rows)
                selected_ids = select_is_winners(temp_df, universe=uni, min_trades=int(args.min_trades))
                selected_variants = [v for v in variants if v["id"] in selected_ids]
                for oos_name, oos_start, oos_end_static in SPLITS[1:]:
                    oos_end = split_end_for(oos_name, args.as_of) if oos_end_static is None else str(oos_end_static)
                    oos_warmup_start = warmup_start_for(oos_start, int(args.warmup_bdays))
                    oos_window = split_data_for_period(data_all, oos_warmup_start, oos_end)
                    for v in selected_variants:
                        res = run_one_variant(
                            universe=uni,
                            split=oos_name,
                            split_start=oos_start,
                            split_end=oos_end,
                            warmup_start=oos_warmup_start,
                            benchmark=bench,
                            symbols=symbols_with_data,
                            variant=v,
                            data_window=oos_window,
                            baseline_cfg=baseline_cfg,
                            baseline_path=baseline_path,
                            baseline_hash=baseline_hash,
                            survivorship_biased=survivorship,
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
        row = {
            "universe": r.universe,
            "split": r.split,
            "variant_id": r.variant_id,
            "variant_label": r.variant_label,
            "family": r.family,
            "survivorship_biased": bool(r.survivorship_biased),
        }
        row.update(r.summary)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "universe", "split", "variant_id", "variant_label", "family", "survivorship_biased",
            "Trades", "CAGR", "MaxDrawdown", "Sharpe_approx", "Sortino_approx", "ProfitFactor", "Expectancy_R",
            "MedianMFE_R", "MedianMAE_R", "Exposure", "Turnover", "confirmed_in_both_oos",
        ])
    summary_df.to_csv(out_dir / "research_summary.csv", index=False)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=[
            "universe", "split", "variant_id", "survivorship_biased", "symbol", "side", "entry_date",
            "entry_px", "exit_date", "exit_px", "shares", "pnl", "reason", "setup", "R_multiple",
            "MFE_R", "MAE_R", "holding_days",
        ])
    trades_df.to_csv(out_dir / "research_trades.csv", index=False)

    confirm_df = evaluate_oos_confirmation(summary_df, min_trades=int(args.min_trades)) if not summary_df.empty else pd.DataFrame()
    if confirm_df.empty:
        confirm_df = pd.DataFrame(columns=["universe", "variant_id", "confirmed_in_both_oos"])
    confirm_df.to_csv(out_dir / "optimization_confirmation.csv", index=False)

    is_sens = summary_df[summary_df["split"] == "in_sample"].copy() if not summary_df.empty else pd.DataFrame()
    stab_df = stable_regions(is_sens) if not is_sens.empty else pd.DataFrame()

    md_lines = [
        "# Reproduzierbarer Research-/Backtest-Report",
        "",
        "## Laufparameter",
        f"- as_of: `{args.as_of}`",
        f"- warmup_bdays: `{args.warmup_bdays}`",
        f"- min_trades: `{args.min_trades}`",
        f"- baseline_config: `{baseline_path}`",
        f"- baseline_config_hash: `{baseline_hash}`",
        "",
        "## Baseline (produktionsnah, unverändert im Live-Scanner)",
        "- 55-Tage-Breakout",
        "- Regimefilter Close > SMA200",
        "- Momentum 126 Tage",
        "- ATR-Stop / ATR-Trailing laut Baseline-Config",
        "- Entry-Annahme: Signal am Close, Entry frühestens nächster handelbarer Open",
        "- Kosten: spread_bps_per_side je Seite",
        "",
        "## Survivorship-Bias und Datenlimitierung",
        "- Belastbare Tests benötigen **PIT-Mitgliedschaften** und **historische delistingbereinigte Preisreihen**.",
        "- Läufe mit aktuellen Wikipedia-Komponenten + Yahoo-Daten sind `survivorship_biased=true` und **kein belastbarer Performance-Nachweis**.",
        "",
        "## Laufnotizen",
    ]
    if run_notes:
        md_lines.extend([f"- {note}" for note in run_notes])
    else:
        md_lines.append("- Keine besonderen Laufnotizen.")

    md_lines += [
        "",
        "## Ergebnisübersicht (Universe/Split/Variante)",
    ]
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
    if confirm_df.empty:
        md_lines.append("Keine bestätigbaren Varianten.")
    else:
        md_lines.append(dataframe_to_markdown(confirm_df))

    md_lines += ["", "## Sensitivitätsstabilität (In-Sample, Top-Nachbarn je Familie)"]
    if stab_df.empty:
        md_lines.append("Keine Sensitivitätsdaten verfügbar.")
    else:
        md_lines.append(dataframe_to_markdown(stab_df))

    (out_dir / "research_report.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
