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
