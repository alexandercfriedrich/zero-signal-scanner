# zero-signal-scanner

Streamlit app that:
- Loads S&P 500 constituents automatically (Wikipedia)
- Resolves custom inputs (ISIN/WKN/Name/Yahoo ticker) via Yahoo Finance search (region preference DE→AT→US)
- Downloads market data via `yfinance` with local disk caching
- Runs scans:
  - **Intraday signalscan**: latest intraday price breaks above daily breakout level (20D high) in risk-on regime
  - **Daily signalscan**: daily close breaks above breakout level in risk-on regime
  - **Daily backtest** (5y): simple swing system with regime filter + breakout entries + ATR stops

> Note: Yahoo search endpoint is unofficial and may rate-limit (HTTP 429). The app uses throttling + a local resolve cache to mitigate.

## Run locally

```bash
cd app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Cache
The app stores downloaded data and resolve results in:

- `~/.zero_swing_cache/`

## Disclaimer
Educational tool, no investment advice.

## Reproduzierbarer Research-/Backtest-Workflow

Für Research (ohne die Live-Standardstrategie im Streamlit-UI zu ändern) gibt es ein separates Skript:

```bash
cd app
python research_workflow.py \
  --output-dir ./research_outputs \
  --universe both \
  --as-of 2026-09-17 \
  --warmup-bdays 300 \
  --baseline-config ./config_best_2011_2026.json
```

Optional mit Point-in-Time-Konstituenten (verringert Survivorship-Bias):

```bash
python research_workflow.py \
  --output-dir ./research_outputs \
  --universe both \
  --pit-constituents-csv /absolute/path/to/pit_constituents.csv
```

### Erwartetes PIT-CSV-Format

Spalten: `split,universe,symbol`

- `split`: `in_sample`, `oos_2019_2022`, `oos_2023_plus` oder `all`
- `universe`: `sp500`, `nasdaq100` oder `all`
- `symbol`: Yahoo-kompatibles Ticker-Symbol

### Baseline (Dokumentation Ist-Stand)

- 55-Tage-Breakout (Close über Breakout-Level auf Schlusskursbasis)
- 200-Tage-Regimefilter (`Close > SMA200`)
- 126-Tage-Momentum im Kandidaten-Scoring
- Initial-Stop: `ATR * 2.0`
- Baseline-Config wird aus `app/config_best_2011_2026.json` geladen (inkl. `max_holding_days: 365`)
- Entry-Annahme ohne Look-ahead: Signal auf Tag *t*, Entry frühestens nächster handelbarer Open (*t+1*)
- Kosten: `spread_bps_per_side` je Seite (Entry und Exit)

### Varianten im Research-Workflow

- **Explizite Entry-Modi**: `breakout_only`, `pullback_only`, `vcp_only`, `breakout_plus_pullback`, `breakout_plus_vcp`
- **A Pullback**: Uptrend + Pullback nahe SMA20 bzw. 10/20-Tage-Range-Low mit ATR-Invalidierung
- **B Volatility Contraction / Range Compression**
- **C Breakout-Extension-Filter** mit Sensitivitätswerten `0.25/0.5/0.75/1.0/1.25/1.5`
- **D Relative Stärke** ggü. SPY/QQQ über 63/126 Tage
- **E Volumenbestätigung** über RelVol (`0.8/1.0/1.2/1.5`) und optional Volumenkontraktion

Zusätzlich wird ein Sensitivitätsgitter für Pullback-/VCP-/RS-Parameter gefahren und im Report als stabile Nachbarregionen ausgewiesen.

### Splits, Outputs und Akzeptanzkriterium

- In-Sample (nur Parameterauswahl): 2011-2018
- OOS-1: 2019-2022
- OOS-2: ab 2023 bis letzter verfügbarer Handelstag
- Pro Split werden mindestens `warmup_bdays` (Default: 300) Handelstage Vorlaufdaten geladen; Kennzahlen/Trades zählen erst ab eigentlichem Splitstart.
- Output-Dateien:
  - `research_summary.csv`
  - `research_trades.csv`
  - `optimization_confirmation.csv`
  - `research_report.md`

Eine Variante gilt nur dann als bestätigt, wenn sie in **beiden** OOS-Perioden folgende Regeln erfüllt: Mindestanzahl Trades (`--min-trades`, Default 40), `Expectancy_R > 0`, `ProfitFactor > 1`, `Sortino_approx` besser als Baseline und Max Drawdown nicht mehr als **2 Prozentpunkte** schlechter als Baseline.

### Datenhinweis / Einschränkung

Für belastbare historische Aussagen reichen PIT-Mitgliedschaften alleine nicht aus: zusätzlich sind delistingbereinigte historische Preisreihen notwendig. Läufe auf heutigen Wikipedia-Bestandteilen mit Yahoo-Daten werden in den Outputs als `survivorship_biased=true` markiert und dürfen nicht als belastbarer Performance-Nachweis interpretiert werden.
