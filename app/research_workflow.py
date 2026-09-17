import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_engine import run_backtest


SPLITS = [
    ("in_sample", "2011-01-01", "2018-12-31"),
    ("oos_2019_2022", "2019-01-01", "2022-12-31"),
    ("oos_2023_2026", "2023-01-01", pd.Timestamp.today().date().isoformat()),
]


BASE_CFG = {
    "hard_risk_on": True,
    "max_new_trades_per_day": 2,
    "max_positions": 5,
    "weekly_rerank": True,
    "weekly_rebalance_weekday": 0,
    "risk_per_trade": 0.01,
    "atr_period": 14,
    "atr_stop_mult": 2.0,
    "use_trailing_stop": True,
    "atr_trail_mult": 3.0,
    "trailing_reference": "close",
    "take_profit_R": 2.0,
    "breakout_lookback": 55,
    "breakout_level_source": "close",
    "breakout_confirm_closes": 1,
    "sma_regime": 200,
    "max_holding_days": 30,
    "mom_lookback": 126,
    "spread_bps_per_side": 8,
    "min_price": 2.0,
    "min_dollar_volume": 2_000_000,
    "initial_cash": 100_000,
    "enable_cwh": False,
}


VARIANTS = [
    {
        "id": "baseline",
        "family": "baseline",
        "label": "55d Breakout Baseline",
        "overrides": {},
    },
    {
        "id": "A_pullback",
        "family": "A",
        "label": "Pullback Entry im Uptrend",
        "overrides": {
            "enable_pullback_entry": True,
            "pullback_sma_tolerance_atr": 0.5,
            "pullback_range_tolerance_atr": 0.5,
            "pullback_invalidation_atr": 1.0,
        },
    },
    {
        "id": "B_vcp",
        "family": "B",
        "label": "Volatility Contraction / Range Compression",
        "overrides": {
            "enable_vcp_entry": True,
            "vcp_atr_ratio_max": 1.0,
            "vcp_range_frac_max": 0.08,
            "vcp_close_pos_min": 0.7,
        },
    },
    {
        "id": "C_ext_0.5",
        "family": "C",
        "label": "Breakout max Extension 0.5 ATR",
        "overrides": {"max_breakout_extension_atr": 0.5},
    },
    {
        "id": "C_ext_1.0",
        "family": "C",
        "label": "Breakout max Extension 1.0 ATR",
        "overrides": {"max_breakout_extension_atr": 1.0},
    },
    {
        "id": "C_ext_1.5",
        "family": "C",
        "label": "Breakout max Extension 1.5 ATR",
        "overrides": {"max_breakout_extension_atr": 1.5},
    },
    {
        "id": "D_rs",
        "family": "D",
        "label": "Relative Stärke ggü. Benchmark",
        "overrides": {
            "enable_relative_strength_filter": True,
            "rs63_min": 0.0,
            "rs126_min": 0.0,
        },
    },
    {
        "id": "E_relvol_1.0",
        "family": "E",
        "label": "RelVol >= 1.0",
        "overrides": {"min_rel_volume": 1.0},
    },
    {
        "id": "E_relvol_1.5",
        "family": "E",
        "label": "RelVol >= 1.5",
        "overrides": {"min_rel_volume": 1.5},
    },
    {
        "id": "E_relvol_1.0_contract",
        "family": "E",
        "label": "RelVol >= 1.0 + Volumen-Kontraktion",
        "overrides": {"min_rel_volume": 1.0, "require_volume_contraction": True},
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


def download_data(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = yf.download(s, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[s] = df
    return out


def score_for_selection(summary: dict[str, Any]) -> float:
    sortino = summary.get("Sortino_approx", np.nan)
    expectancy = summary.get("Expectancy_R", np.nan)
    has_finite = bool(np.isfinite(sortino) or np.isfinite(expectancy))
    if not has_finite:
        return np.nan
    s = 0.0
    s += float(sortino) if np.isfinite(sortino) else 0.0
    s += float(expectancy) if np.isfinite(expectancy) else 0.0
    return s


def select_family_winners(in_sample_results: list[RunResult], universe: str | None = None) -> set[str]:
    selected_ids: set[str] = set()
    for fam in ["A", "B", "C", "D", "E"]:
        fam_rows = []
        for r in in_sample_results:
            if universe is not None and r.universe != universe:
                continue
            if r.family != fam:
                continue
            sc = score_for_selection(r.summary)
            if np.isfinite(sc):
                fam_rows.append({"variant_id": r.variant_id, "score": sc})
        if not fam_rows:
            continue
        score_df = pd.DataFrame(fam_rows)
        best_id = score_df.groupby("variant_id", dropna=False)["score"].mean().sort_values(ascending=False).index[0]
        selected_ids.add(str(best_id))
    return selected_ids


def benchmark_summary(bench_df: pd.DataFrame) -> dict[str, float]:
    close_col = "Adj Close" if "Adj Close" in bench_df.columns else "Close"
    close = bench_df[close_col].dropna()
    if close.empty:
        return {"Benchmark_CAGR": np.nan, "Benchmark_MaxDrawdown": np.nan, "Benchmark_Sortino": np.nan}
    ret = close.pct_change().fillna(0)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (252 / max(1, len(close) - 1)) - 1
    dd = (close / close.cummax() - 1).min()
    downside = ret[ret < 0]
    sortino = (ret.mean() * 252) / ((downside.std() * np.sqrt(252)) + 1e-12)
    return {
        "Benchmark_CAGR": float(cagr),
        "Benchmark_MaxDrawdown": float(dd),
        "Benchmark_Sortino": float(sortino),
    }


def run_one_variant(universe: str, split: str, start: str, end: str, benchmark: str, symbols: list[str], variant: dict[str, Any], data: dict[str, pd.DataFrame], survivorship_biased: bool) -> RunResult:
    cfg = dict(BASE_CFG)
    cfg.update(variant["overrides"])
    cfg.update({
        "start": start,
        "end": end,
        "symbols": symbols,
        "regime_symbol": benchmark,
    })
    eq, trades, summary, _ = run_backtest(data, cfg)
    summary = dict(summary)
    bdf = data[benchmark]
    bidx = bdf.index
    if getattr(bidx, "tz", None) is not None:
        start_ts = pd.Timestamp(start).tz_localize(bidx.tz)
        end_ts = pd.Timestamp(end).tz_localize(bidx.tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    bdf_split = bdf.loc[(bidx >= start_ts) & (bidx <= end_ts)]
    summary.update(benchmark_summary(bdf_split))
    return RunResult(
        universe=universe,
        split=split,
        variant_id=variant["id"],
        variant_label=variant["label"],
        family=variant["family"],
        summary=summary,
        trades_df=trades,
        survivorship_biased=survivorship_biased,
    )


def evaluate_acceptance(result_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for universe in sorted(result_df["universe"].unique()):
        uni_df = result_df[result_df["universe"] == universe]
        baseline = uni_df[uni_df["variant_id"] == "baseline"]
        for vid in sorted(uni_df["variant_id"].unique()):
            if vid == "baseline":
                continue
            vdf = uni_df[uni_df["variant_id"] == vid]
            passed = True
            for split in ["oos_2019_2022", "oos_2023_2026"]:
                b = baseline[baseline["split"] == split]
                v = vdf[vdf["split"] == split]
                if b.empty or v.empty:
                    passed = False
                    break
                b_exp, v_exp = b.iloc[0]["Expectancy_R"], v.iloc[0]["Expectancy_R"]
                b_sort, v_sort = b.iloc[0]["Sortino_approx"], v.iloc[0]["Sortino_approx"]
                b_dd, v_dd = b.iloc[0]["MaxDrawdown"], v.iloc[0]["MaxDrawdown"]
                imp = (np.isfinite(v_exp) and np.isfinite(b_exp) and v_exp > b_exp) or (
                    np.isfinite(v_sort) and np.isfinite(b_sort) and v_sort > b_sort
                )
                dd_ok = np.isfinite(v_dd) and np.isfinite(b_dd) and v_dd >= (b_dd - 0.02)
                if not (imp and dd_ok):
                    passed = False
                    break
            rows.append({"universe": universe, "variant_id": vid, "confirmed_in_both_oos": passed})
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduzierbarer Research-/Backtest-Workflow")
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--pit-constituents-csv", default=None)
    parser.add_argument("--universe", choices=["sp500", "nasdaq100", "both"], default="both")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pit_df = load_pit_csv(args.pit_constituents_csv)
    universes = ["sp500", "nasdaq100"] if args.universe == "both" else [args.universe]

    all_results: list[RunResult] = []
    all_trades: list[pd.DataFrame] = []

    # Phase 1: In-sample runs for all universes (variant selection only from IS)
    in_sample_results: list[RunResult] = []
    split_is, start_is, end_is = SPLITS[0]
    for uni in universes:
        bench = "SPY" if uni == "sp500" else "QQQ"
        syms, survivorship_biased = symbols_for(uni, split_is, pit_df)
        run_symbols = sorted(set(syms + [bench]))
        data = download_data(run_symbols, start_is, end_is)
        syms_in_data = [s for s in syms if s in data]
        if bench not in data or not syms_in_data:
            continue
        for variant in VARIANTS:
            res = run_one_variant(uni, split_is, start_is, end_is, bench, syms_in_data, variant, data, survivorship_biased)
            in_sample_results.append(res)
            all_results.append(res)
            if res.trades_df is not None and not res.trades_df.empty:
                t = res.trades_df.copy()
                t.insert(0, "variant_id", res.variant_id)
                t.insert(0, "split", res.split)
                t.insert(0, "universe", res.universe)
                all_trades.append(t)

    # Select one IS winner per (universe, family)
    selected_ids_by_universe = {uni: select_family_winners(in_sample_results, universe=uni) for uni in universes}

    for uni in universes:
        bench = "SPY" if uni == "sp500" else "QQQ"
        oos_variants = [{"id": "baseline", "family": "baseline", "label": "55d Breakout Baseline", "overrides": {}}]
        oos_variants += [v for v in VARIANTS if v["id"] in selected_ids_by_universe.get(uni, set())]
        for split, start, end in SPLITS[1:]:
            syms, survivorship_biased = symbols_for(uni, split, pit_df)
            run_symbols = sorted(set(syms + [bench]))
            data = download_data(run_symbols, start, end)
            syms_in_data = [s for s in syms if s in data]
            if bench not in data or not syms_in_data:
                continue
            for variant in oos_variants:
                res = run_one_variant(uni, split, start, end, bench, syms_in_data, variant, data, survivorship_biased)
                all_results.append(res)
                if res.trades_df is not None and not res.trades_df.empty:
                    t = res.trades_df.copy()
                    t.insert(0, "variant_id", res.variant_id)
                    t.insert(0, "split", res.split)
                    t.insert(0, "universe", res.universe)
                    all_trades.append(t)

    summary_rows = []
    for r in all_results:
        row = {
            "universe": r.universe,
            "split": r.split,
            "variant_id": r.variant_id,
            "variant_label": r.variant_label,
            "family": r.family,
            "survivorship_biased": r.survivorship_biased,
        }
        row.update(r.summary)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "research_summary.csv", index=False)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades_df.to_csv(out_dir / "research_trades.csv", index=False)

    confirm_df = evaluate_acceptance(summary_df) if not summary_df.empty else pd.DataFrame()
    confirm_df.to_csv(out_dir / "optimization_confirmation.csv", index=False)

    md_lines = [
        "# Reproduzierbarer Research-/Backtest-Report",
        "",
        "## Baseline (Ist-Stand)",
        "- Einstieg: 55-Tage-Breakout (Close über Vortages-Breakout-Level)",
        "- Regimefilter: Close > SMA200",
        "- Momentum: 126 Tage (als Ranking/Multi-Faktor in der Kandidatenauswahl)",
        "- Stop: ATR × 2.0, Trailing: ATR × 3.0",
        "- Entry-Annahme: Signal am Tagesende, Ausführung frühestens nächster handelbarer Open",
        "- Kostenannahme: spread_bps_per_side pro Entry und Exit-Seite",
        "",
        "## Daten-/Zeit-Splits",
        "- In-Sample: 2011-2018 (nur Parameterauswahl)",
        "- OOS-1: 2019-2022",
        "- OOS-2: ab 2023 bis letzter verfügbarer Handelstag",
        "",
        "## Survivorship-Bias Hinweis",
    ]
    if not summary_df.empty and summary_df["survivorship_biased"].any():
        md_lines.append("**ACHTUNG: Ergebnisse mit aktuellen Indexbestandteilen sind survivorship-biased und nicht als belastbarer Performance-Nachweis zu interpretieren.**")
    else:
        md_lines.append("PIT-Konstituenten-CSV genutzt (split/universe/symbol), Survivorship-Bias reduziert.")

    md_lines.append("")
    md_lines.append("## Ergebnisübersicht")
    if summary_df.empty:
        md_lines.append("Keine auswertbaren Ergebnisse erzeugt.")
    else:
        show_cols = [
            "universe", "split", "variant_id", "Trades", "WinRate", "AvgWin_R", "AvgLoss_R",
            "ProfitFactor", "Expectancy_R", "CAGR", "MaxDrawdown", "Sharpe_approx", "Sortino_approx",
            "Exposure", "Turnover", "MedianHoldDays", "MedianMFE_R", "MedianMAE_R",
            "Benchmark_CAGR", "Benchmark_Sortino",
        ]
        cols = [c for c in show_cols if c in summary_df.columns]
        md_lines.append(dataframe_to_markdown(summary_df[cols]))

    md_lines.append("")
    md_lines.append("## Optimierungs-Akzeptanz (beide OOS)")
    if confirm_df.empty:
        md_lines.append("Keine bestätigbaren Varianten.")
    else:
        md_lines.append(dataframe_to_markdown(confirm_df))

    (out_dir / "research_report.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
