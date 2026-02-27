import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

from backtest_engine import run_backtest, rsi

st.set_page_config(page_title='3S- Stock Signal Scanner by AF', layout='wide')

CACHE_DIR = Path.home() / '.zero_swing_cache'
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_CFG = {
  "start": "2021-02-27",
  "end": "2026-02-27",
  "regime_symbol": "SPY",
  "inverse_map": {"SPY": "SH"},
  "hard_risk_on": True,

  "max_new_trades_per_day": 2,
  "max_positions": 5,
  "weekly_rerank": True,
  "weekly_rebalance_weekday": 0,

  "risk_per_trade": 0.01,

  "atr_period": 14,
  "atr_stop_mult": 2.0,
  "use_trailing_stop": True,
  "atr_trail_mult": 2.5,
  "trailing_reference": "high",

  "breakout_lookback": 55,
  "breakout_level_source": "close",
  "breakout_confirm_closes": 1,
  "sma_regime": 200,
  "max_holding_days": 30,

  "min_breakout_vol_mult": 0.0,
  "rsi_period": 0,
  "rsi_max": 100,
  "max_breakout_extension_atr": 1e9,

  "mom_lookback": 126,

  "enable_cwh": True,
  "cwh_cup_min_bars": 30,
  "cwh_cup_max_bars": 130,
  "cwh_handle_min_bars": 5,
  "cwh_handle_max_bars": 20,
  "cwh_max_cup_depth": 0.35,
  "cwh_max_handle_depth": 0.15,
  "cwh_trend_sma": 50,
  "cwh_vol_bonus": 0.3,

  "corr_lookback_days": 60,
  "max_pair_corr": 1.0,
  "max_positions_per_sector": 999,

  "spread_bps_per_side": 8,
  "min_price": 2.0,
  "min_dollar_volume": 2_000_000,
  "initial_cash": 5000
}

# Presets (loaded from JSON files in app/)
PRESETS = {
    'Swing (Top-5, Risk-On only)': None,  # uses DEFAULT_CFG
    'Best (2011-2026)': 'config_best_2011_2026.json',
    'Best (2011-2026) – mehr Signale': 'config_best_2011_2026_mehr_signale.json',
    'Best (2011-2026) – höhere Trefferquote': 'config_best_2011_2026_hoehere_trefferquote.json',
}


def _merge_cfg(base: dict, override: dict) -> dict:
    """Shallow-merge override into base, but merge inverse_map dicts."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k == 'inverse_map' and isinstance(v, dict):
            out[k] = dict(out.get(k, {}) or {})
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_preset_cfg(preset_name: str) -> dict:
    """Load preset config dict, always including DEFAULT_CFG keys.

    This ensures new optional parameters are visible/editable in the config textarea
    even when older preset JSON files don't contain those keys.
    """
    p = PRESETS.get(preset_name)
    if not p:
        return dict(DEFAULT_CFG)

    preset_path = (Path(__file__).parent / p)
    try:
        raw = json.loads(preset_path.read_text())
        if not isinstance(raw, dict):
            return dict(DEFAULT_CFG)
        return _merge_cfg(DEFAULT_CFG, raw)
    except Exception:
        # Fall back to default if file is missing or invalid
        return dict(DEFAULT_CFG)


st.title('3S- Stock Signal Scanner by AF')
st.markdown("""
**Was macht dieser Scanner?**  
Der 3S Stock Signal Scanner durchsucht täglich Aktien aus verschiedenen Indizes nach technischen Ausbruchssignalen auf Tagesbasis.

**Gesuchtes Signal:**
- 📊 **Daily Close-Breakout**: Tagesschlusskurs überschreitet das 55-Tage-Hoch

**Einstieg & Ausstieg (Backtest-Logik):**
- **Einstieg**: Am nächsten Handelstag zum Eröffnungskurs (Next-Day Open)
- **Stop-Loss**: ATR-basierter initialer Stop (ATR × Multiplikator)
- **Trailing Stop** *(optional)*: ATR-Trailing aktivierbar
- **Take-Profit** *(optional)*: Nur wenn Trailing deaktiviert
- **Maximale Haltedauer**: Zeitbasierter Exit nach konfigurierbaren Tagen
- **Weekly Rerank** *(optional)*: Schwache Positionen werden wöchentlich ersetzt
- **Regime-Filter**: Käufe nur wenn SPY > SMA 200 (Risk-On)
""")

with st.sidebar:
    st.header('Universe')
    universe_mode = st.radio(
        'Quelle',
        ['S&P 500 (Wikipedia)', 'Nasdaq 100 (Wikipedia)', 'DAX 40 (Wikipedia)', 'ATX (Wikipedia)', 'Custom'],
        index=0,
    )
    custom_list = ''
    if universe_mode == 'Custom':
        custom_list = st.text_area(
            'Ticker-Liste',
            value='AAPL\nMSFT\nVIG.VI\nDB1.DE',
            help=(
                'Nur Yahoo-Finance-Ticker-Symbole werden akzeptiert (ein Symbol pro Zeile). '
                'Alternativ können auch Firmenname, ISIN oder WKN eingegeben werden; '
                'der Resolver versucht diese automatisch in Yahoo-Ticker umzuwandeln '
                '(kann durch Rate-Limiting zeitweise langsam sein).'
            ),
        )

    st.header('Aktion')
    action = st.radio('Modus', ['Daily Signalscan', 'Backtest (Daily, 5y)'], index=0)

    st.header('Resolver Präferenzen')
    prefer_regions = st.multiselect('Yahoo region Reihenfolge', ['DE','AT','US','GB'], default=['DE','AT','US'])
    throttle_s = st.slider('Resolver Throttle (Sek.)', 0.0, 1.0, 0.2, 0.05)

    st.header('Konfiguration')

    preset = st.selectbox('Preset', list(PRESETS.keys()), index=1)

    # Initialize config text once from the selected default preset
    if 'cfg_text' not in st.session_state:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)

    # When preset changes, replace textarea content
    prev_preset = st.session_state.get('preset_name')
    if prev_preset != preset:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)
        st.session_state['preset_name'] = preset

    cfg_text = st.text_area('config.json (ohne symbols)', value=st.session_state['cfg_text'], height=420)

    with st.expander('ℹ️ Neue Konfigurations-Parameter'):
        st.markdown("""
| Parameter | Standard | Beschreibung |
|---|---|---|
| `trailing_reference` | `"high"` | Trailing-Stop-Basis: `"high"` = Chandelier (Höchstkurs), `"close"` = Schlusskurs |
| `breakout_level_source` | `"close"` | Ausbruchsniveau: `"close"` = Schlusskurshoch, `"high"` = Tageshoch |
| `breakout_confirm_closes` | `1` | Anzahl aufeinanderfolgender Closes über Breakout-Level (1 = sofort, 2 = bestätigt) |
| `min_breakout_vol_mult` | `0.0` | Mindest-Volumen: Volume ≥ N × VolSMA50 am Breakout-Tag (0 = deaktiviert) |
| `rsi_period` | `0` | RSI-Periode für Overbought-Filter (0 = deaktiviert) |
| `rsi_max` | `100` | Max. RSI beim Einstieg (z.B. 70 = kein Einstieg bei überkauft) |
| `max_breakout_extension_atr` | `1e9` | Max. Ausdehnung in ATR-Einheiten über dem Level (1e9 = deaktiviert) |
| `corr_lookback_days` | `60` | Lookback-Tage für Korrelationsfilter zwischen Positionen |
| `max_pair_corr` | `1.0` | Max. Korrelation zu offenen Positionen (1.0 = deaktiviert) |
| `max_positions_per_sector` | `999` | Max. Positionen pro Sektor (999 = deaktiviert, benötigt S&P 500 Universe) |
""")

    run_btn = st.button('Start', type='primary')


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_sp500_symbols() -> list[str]:
    """Load S&P 500 tickers."""

    wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        r = requests.get(
            wiki_url,
            timeout=25,
            headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'}
        )
        r.raise_for_status()
        tables = pd.read_html(r.text)
        df = tables[0]
        syms = df['Symbol'].astype(str).str.upper().tolist()
        return [s.replace('.', '-') for s in syms]
    except Exception:
        pass

    raw_csv = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    r = requests.get(raw_csv, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
    r.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    syms = df['Symbol'].astype(str).str.upper().tolist()
    return [s.replace('.', '-') for s in syms]


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_nasdaq100_symbols() -> list[str]:
    """Load Nasdaq 100 tickers from Wikipedia."""
    wiki_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    try:
        r = requests.get(wiki_url, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
                syms = df[col].astype(str).str.upper().tolist()
                return [s.replace('.', '-') for s in syms if s not in ('NAN', '')]
    except Exception:
        pass
    return []


def _add_exchange_suffix(syms: list[str], suffix: str) -> list[str]:
    """Append exchange suffix (e.g. '.DE', '.VI') to plain ticker symbols without an existing exchange suffix.
    Symbols already containing a dot, or longer than 6 characters, are left unchanged.
    """
    result = []
    for s in syms:
        # Max 6 chars is typical for European exchange tickers without suffix
        if '.' not in s and len(s) <= 6:
            result.append(s + suffix)
        else:
            result.append(s)
    return result


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_dax40_symbols() -> list[str]:
    """Load DAX 40 tickers from Wikipedia, appending .DE suffix for yfinance."""
    wiki_url = 'https://en.wikipedia.org/wiki/DAX'
    try:
        r = requests.get(wiki_url, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
                syms = df[col].astype(str).str.upper().tolist()
                syms = [s for s in syms if s not in ('NAN', '')]
                return _add_exchange_suffix(syms, '.DE')
    except Exception:
        pass
    return []


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_atx_symbols() -> list[str]:
    """Load ATX tickers from Wikipedia, appending .VI suffix for yfinance."""
    wiki_url = 'https://en.wikipedia.org/wiki/Austrian_Traded_Index'
    try:
        r = requests.get(wiki_url, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
                syms = df[col].astype(str).str.upper().tolist()
                syms = [s for s in syms if s not in ('NAN', '')]
                return _add_exchange_suffix(syms, '.VI')
    except Exception:
        pass
    return []


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_sp500_sector_map() -> dict:
    """Load S&P 500 sector mapping from Wikipedia.

    Searches for the first column containing 'symbol' or 'ticker' and the first
    column containing 'sector' or 'gics' (e.g. 'GICS Sector'). Returns a dict
    mapping ticker → sector string, or an empty dict on failure.
    """
    wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        r = requests.get(wiki_url, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        df = tables[0]
        cols_lower = {str(c).lower(): c for c in df.columns}
        sym_col = next((cols_lower[k] for k in cols_lower if 'symbol' in k or 'ticker' in k), None)
        sec_col = next((cols_lower[k] for k in cols_lower if 'sector' in k or 'gics' in k), None)
        if sym_col and sec_col:
            result = {}
            for _, row in df.iterrows():
                sym = str(row[sym_col]).upper().replace('.', '-')
                result[sym] = str(row[sec_col])
            return result
    except Exception:
        pass
    return {}


def yahoo_search(query: str, region: str, lang: str):
    url = 'https://query2.finance.yahoo.com/v1/finance/search'
    params = {
        'q': query,
        'quotesCount': 6,
        'newsCount': 0,
        'listsCount': 0,
        'region': region,
        'lang': lang,
        'enableFuzzyQuery': 'false'
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 429:
        raise RuntimeError('Yahoo Search rate-limited (HTTP 429).')
    r.raise_for_status()
    return r.json().get('quotes', [])


def load_resolve_cache() -> dict:
    p = CACHE_DIR / 'resolve_cache.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_resolve_cache(cache: dict):
    (CACHE_DIR / 'resolve_cache.json').write_text(json.dumps(cache, indent=2))


def score_symbol(sym: str) -> int:
    sym = (sym or '').upper()
    prefs = ['.DE','.VI','.PA','.AS','.MI','.BR','.LS','.S','.ST','.OL','.CO','.HE']
    for i, suf in enumerate(prefs):
        if sym.endswith(suf):
            return 100 - i
    return 0


def resolve_one(query: str, regions: list[str]) -> str | None:
    q = query.strip()
    if not q:
        return None
    if any(ch in q for ch in ['.','-','=']) or (q.isalpha() and 1 <= len(q) <= 7):
        return q.upper()

    best = None
    best_score = -1
    for region in regions:
        lang = {'DE':'de-DE','AT':'de-AT','US':'en-US','GB':'en-GB'}.get(region, 'en-US')
        for item in yahoo_search(q, region=region, lang=lang):
            sym = item.get('symbol')
            if not sym:
                continue
            sc = score_symbol(sym)
            if sc > best_score:
                best = sym
                best_score = sc
        time.sleep(throttle_s)
    return best


def resolve_inputs(lines: list[str], regions: list[str]):
    cache = load_resolve_cache()
    resolved = []
    rows = []
    for raw in lines:
        q = raw.strip()
        if not q:
            continue
        key = q.upper()
        if key in cache:
            sym = cache[key]
        else:
            sym = resolve_one(q, regions)
            cache[key] = sym
            save_resolve_cache(cache)
        if sym:
            resolved.append(sym.upper())
        rows.append({'input': q, 'resolved_symbol': sym})
    return sorted(list(dict.fromkeys(resolved))), pd.DataFrame(rows)


def cache_path(sym: str, kind: str) -> Path:
    safe = sym.replace('^','_').replace('/','_')
    return CACHE_DIR / f'{safe}.{kind}.parquet'


def fetch_yf(symbols, start=None, end=None, interval='1d', period=None):
    if period is not None:
        return yf.download(symbols, period=period, interval=interval, auto_adjust=False, group_by='ticker', threads=True, progress=False)
    return yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=False, group_by='ticker', threads=True, progress=False)


def unpack_download(df, batch):
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for sym in batch:
            if sym in df.columns.levels[0]:
                sub = df[sym].dropna().reset_index()
                if not sub.empty:
                    out[sym] = sub
    else:
        sym = batch[0]
        sub = df.dropna().reset_index()
        if not sub.empty:
            out[sym] = sub
    return out


def load_daily(symbols: list[str], start: str, end: str, progress_cb=None) -> dict[str, pd.DataFrame]:
    out = {}
    need = []
    for s in symbols:
        p = cache_path(s, '1d')
        if p.exists():
            try:
                d = pd.read_parquet(p)
                d['Date'] = pd.to_datetime(d['Date'])
                if d['Date'].min() <= pd.Timestamp(start) and d['Date'].max() >= pd.Timestamp(end) - pd.Timedelta(days=3):
                    out[s] = d
                    continue
            except Exception:
                pass
        need.append(s)

    if need:
        chunk = 80
        total = int(np.ceil(len(need) / chunk))
        for k, i in enumerate(range(0, len(need), chunk)):
            batch = need[i:i+chunk]
            if progress_cb is not None:
                progress_cb(k, total, f'Daily: {i}/{len(need)}')
            got = unpack_download(fetch_yf(batch, start=start, end=end, interval='1d'), batch)
            for sym, sub in got.items():
                out[sym] = sub
                sub.to_parquet(cache_path(sym, '1d'), index=False)
        if progress_cb is not None:
            progress_cb(total, total, 'Daily: done')

    return out


if run_btn:
    cfg = json.loads(cfg_text)

    with st.spinner('Universe laden & ggf. auflösen...'):
        if universe_mode == 'S&P 500 (Wikipedia)':
            symbols = load_sp500_symbols()
            resolve_table = pd.DataFrame({'input': symbols, 'resolved_symbol': symbols})
        elif universe_mode == 'Nasdaq 100 (Wikipedia)':
            symbols = load_nasdaq100_symbols()
            resolve_table = pd.DataFrame({'input': symbols, 'resolved_symbol': symbols})
        elif universe_mode == 'DAX 40 (Wikipedia)':
            symbols = load_dax40_symbols()
            resolve_table = pd.DataFrame({'input': symbols, 'resolved_symbol': symbols})
        elif universe_mode == 'ATX (Wikipedia)':
            symbols = load_atx_symbols()
            resolve_table = pd.DataFrame({'input': symbols, 'resolved_symbol': symbols})
        else:
            lines = [x for x in custom_list.splitlines()]
            symbols, resolve_table = resolve_inputs(lines, prefer_regions)

    st.subheader('Symbol‑Auflösung')
    st.dataframe(resolve_table, use_container_width=True)

    cfg = dict(cfg)
    cfg['symbols'] = [s for s in symbols if s not in [cfg['regime_symbol']] and s not in cfg.get('inverse_map', {}).values()]

    # Load sector map if available (used for sector-cap filter)
    sector_map = {}
    if universe_mode == 'S&P 500 (Wikipedia)':
        sector_map = load_sp500_sector_map()
    cfg['sector_map'] = sector_map

    needed_daily = sorted(list(set(cfg['symbols'] + [cfg['regime_symbol']] + list(cfg.get('inverse_map', {}).values()))))

    ui_prog = st.progress(0.0)
    ui_status = st.empty()

    def prog_step(done, total, msg):
        frac = 0.0 if total <= 0 else float(done) / float(total)
        ui_prog.progress(min(1.0, max(0.0, frac)))
        ui_status.caption(msg)

    if action == 'Backtest (Daily, 5y)':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'], progress_cb=lambda d,t,m: prog_step(d, t, m))
        cfg['symbols'] = [s for s in cfg['symbols'] if s in daily]

        ui_status.caption('Backtest läuft...')
        bt_prog = st.progress(0.0)
        bt_status = st.empty()

        def bt_step(done, total, date_or_msg):
            frac = 0.0 if total <= 0 else float(done) / float(total)
            bt_prog.progress(min(1.0, max(0.0, frac)))
            if isinstance(date_or_msg, str):
                bt_status.caption(date_or_msg)
            else:
                bt_status.caption(f'Backtest: {done}/{total}  ({date_or_msg.date().isoformat()})')

        equity_df, trades_df, summary, breakdown = run_backtest(daily, cfg, progress_cb=bt_step)
        bt_status.caption('Backtest: done')

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Endkapital', f"{summary['final_equity']:.2f}")
        m2.metric('CAGR', f"{summary['CAGR']*100:.2f}%")
        m3.metric('Max. Drawdown', f"{summary['MaxDrawdown']*100:.2f}%")
        m4.metric('Volatilität', f"{summary['Volatility']*100:.2f}%")
        m5.metric('Trades', str(summary['Trades']))

        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric('Profit Factor', '-' if pd.isna(summary.get('ProfitFactor')) else f"{summary.get('ProfitFactor'):.2f}")
        m7.metric('Win Rate', '-' if pd.isna(summary.get('WinRate')) else f"{summary.get('WinRate')*100:.1f}%")
        m8.metric('Avg Win', '-' if pd.isna(summary.get('AvgWin')) else f"{summary.get('AvgWin'):.2f}")
        m9.metric('Avg Loss', '-' if pd.isna(summary.get('AvgLoss')) else f"{summary.get('AvgLoss'):.2f}")
        m10.metric('Expectancy (R)', '-' if pd.isna(summary.get('Expectancy_R')) else f"{summary.get('Expectancy_R'):.2f}")

        with st.expander('ℹ️ Kennzahlen-Erklärung'):
            st.markdown("""
| Kennzahl | Erklärung |
|---|---|
| **Endkapital** | Gesamtkapital am Ende des Backtest-Zeitraums |
| **CAGR** | Jährliche Wachstumsrate (Compound Annual Growth Rate) |
| **Max. Drawdown** | Größter prozentualer Rückgang vom Höchststand bis zum Tiefpunkt |
| **Volatilität** | Annualisierte Standardabweichung der täglichen Renditen |
| **Trades** | Gesamtanzahl abgeschlossener Trades |
| **Profit Factor** | Verhältnis Gesamtgewinn / Gesamtverlust (>1 = profitabel) |
| **Win Rate** | Anteil gewinnbringender Trades an allen Trades |
| **Avg Win** | Durchschnittlicher Gewinn je gewinnbringendem Trade (in €) |
| **Avg Loss** | Durchschnittlicher Verlust je verlustbringendem Trade (in €) |
| **Expectancy (R)** | Erwartungswert je Trade in Vielfachen des Risikos (R); >0 = positiv |
""")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(px.line(equity_df.reset_index(), x='Date', y='Equity', title='Equity Curve'), use_container_width=True)
        with c2:
            eq = equity_df['Equity']
            dd = (eq/eq.cummax() - 1).rename('Drawdown')
            st.plotly_chart(px.area(dd.reset_index(), x='Date', y='Drawdown', title='Drawdown'), use_container_width=True)

        st.subheader('Downloads')
        st.download_button(
            'Download trades.csv',
            data=trades_df.to_csv(index=False).encode('utf-8'),
            file_name='trades.csv',
            mime='text/csv',
        )
        st.download_button(
            'Download equity.csv',
            data=equity_df.reset_index().to_csv(index=False).encode('utf-8'),
            file_name='equity.csv',
            mime='text/csv',
        )

        st.subheader('Breakdown')
        b1, b2 = st.columns(2)
        breakdown_col_cfg = {
            'setup': st.column_config.TextColumn('Setup', help='Signal-Typ (z.B. BREAKOUT, CWH)'),
            'reason': st.column_config.TextColumn('Grund', help='Exit-Grund (z.B. STOP, TRAIL, TIME, RERANK)'),
            'Trades': st.column_config.NumberColumn('Trades', help='Anzahl abgeschlossener Trades'),
            'WinRate': st.column_config.NumberColumn('Win Rate', help='Anteil gewinnbringender Trades', format='%.1f%%'),
            'ProfitFactor': st.column_config.NumberColumn('Profit Factor', help='Gesamtgewinn / Gesamtverlust', format='%.2f'),
            'AvgPnL': st.column_config.NumberColumn('Ø PnL', help='Durchschnittlicher Gewinn/Verlust je Trade (€)', format='%.2f'),
            'AvgR': st.column_config.NumberColumn('Ø R', help='Durchschnittliches R-Vielfaches je Trade', format='%.2f'),
        }
        with b1:
            st.caption('Nach Setup')
            st.dataframe(breakdown.get('by_setup', pd.DataFrame()), column_config=breakdown_col_cfg, use_container_width=True)
        with b2:
            st.caption('Nach Exit-Grund')
            st.dataframe(breakdown.get('by_reason', pd.DataFrame()), column_config=breakdown_col_cfg, use_container_width=True)

        trades_col_cfg = {
            'symbol': st.column_config.TextColumn('Symbol', help='Aktien-Ticker-Symbol'),
            'side': st.column_config.TextColumn('Richtung', help='Handelsrichtung (LONG/SHORT)'),
            'entry_date': st.column_config.TextColumn('Einstiegsdatum', help='Datum des Einstiegs'),
            'entry_px': st.column_config.NumberColumn('Einstiegskurs', help='Kurs beim Einstieg', format='%.2f'),
            'exit_date': st.column_config.TextColumn('Ausstiegsdatum', help='Datum des Ausstiegs'),
            'exit_px': st.column_config.NumberColumn('Ausstiegskurs', help='Kurs beim Ausstieg', format='%.2f'),
            'shares': st.column_config.NumberColumn('Stück', help='Anzahl gehandelter Aktien'),
            'pnl': st.column_config.NumberColumn('PnL (€)', help='Gewinn oder Verlust in Euro', format='%.2f'),
            'reason': st.column_config.TextColumn('Exitgrund', help='Grund für den Ausstieg (STOP, TRAIL, TIME, RERANK)'),
            'setup': st.column_config.TextColumn('Setup', help='Signal-Typ (z.B. BREAKOUT, CWH)'),
            'initial_risk_per_share': st.column_config.NumberColumn('Init. Risiko/Aktie', help='Initialer Risikobetrag je Aktie beim Einstieg (€)', format='%.2f'),
            'R_multiple': st.column_config.NumberColumn('R-Vielfaches', help='Gewinn/Verlust in Vielfachen des initialen Risikos (R)', format='%.2f'),
        }
        st.subheader('Trades')
        st.dataframe(trades_df, column_config=trades_col_cfg, use_container_width=True)

    elif action == 'Daily Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'], progress_cb=lambda d,t,m: prog_step(d, t, m))

        reg = daily[cfg['regime_symbol']].copy(); reg['Date']=pd.to_datetime(reg['Date']); reg=reg.sort_values('Date').set_index('Date')
        risk_on = bool(reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])

        bl_src = cfg.get('breakout_level_source', 'close')
        scan_confirm = int(cfg.get('breakout_confirm_closes', 1))
        scan_min_bvm = float(cfg.get('min_breakout_vol_mult', 0.0))
        scan_rsi_p = int(cfg.get('rsi_period', 0))
        scan_rsi_max = float(cfg.get('rsi_max', 100))
        scan_max_ext = float(cfg.get('max_breakout_extension_atr', 1e9))

        rows = []
        for sym in cfg['symbols']:
            if sym not in daily:
                continue
            df = daily[sym].copy(); df['Date']=pd.to_datetime(df['Date']); df=df.sort_values('Date').set_index('Date')

            if bl_src == 'high':
                bl_series = df['High'].shift(1).rolling(cfg['breakout_lookback']).max()
            else:
                bl_series = df['Close'].shift(1).rolling(cfg['breakout_lookback']).max()

            high, low, close = df['High'], df['Low'], df['Close']
            tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
            atr_v = tr.rolling(cfg['atr_period']).mean().iloc[-1]
            bl = bl_series.iloc[-1]
            if pd.isna(bl) or pd.isna(atr_v) or float(atr_v) <= 0:
                continue

            px = float(df['Close'].iloc[-1])
            if not risk_on or px <= float(bl):
                continue

            # Consecutive-close confirmation
            if scan_confirm >= 2:
                if len(df) < scan_confirm:
                    continue
                confirmed = all(
                    (not pd.isna(bl_series.iloc[-(k+1)])) and
                    float(df['Close'].iloc[-(k+1)]) > float(bl_series.iloc[-(k+1)])
                    for k in range(scan_confirm)
                )
                if not confirmed:
                    continue

            breakout_strength = (px - float(bl)) / float(atr_v)

            # Extension cap
            if scan_max_ext < 1e9 and breakout_strength > scan_max_ext:
                continue

            # RSI filter
            if scan_rsi_p > 0 and scan_rsi_max < 100:
                rsi_v = float(rsi(df['Close'], scan_rsi_p).iloc[-1])
                if pd.isna(rsi_v) or rsi_v > scan_rsi_max:
                    continue

            # Volume data
            vol_sma50 = df['Volume'].rolling(50).mean().iloc[-1]
            vol_today = df['Volume'].iloc[-1]

            # Volume confirmation filter
            if scan_min_bvm > 0:
                if pd.isna(vol_sma50) or float(vol_sma50) <= 0 or pd.isna(vol_today) or float(vol_today) < scan_min_bvm * float(vol_sma50):
                    continue

            vol_ratio = float(vol_today) / float(vol_sma50) if (not pd.isna(vol_sma50) and float(vol_sma50) > 0 and not pd.isna(vol_today)) else 1.0

            risk_per_share = float(cfg['atr_stop_mult'] * float(atr_v))
            stop_price = float(px - risk_per_share)
            tp_price = float(px + float(cfg.get('take_profit_R', 2.0)) * risk_per_share)
            shares_for_1000eur = int(max(0, (1000.0 * float(cfg['risk_per_trade'])) // max(1e-9, risk_per_share)))
            rows.append({
                'symbol': sym, 'side': 'LONG', 'price': px,
                'breakout_level': float(bl), 'asof': str(df.index[-1].date()),
                'atr': float(atr_v), 'risk_per_share': risk_per_share,
                'stop_price': stop_price, 'tp_price': tp_price,
                'shares_for_1000eur': shares_for_1000eur,
                'vol_ratio': round(vol_ratio, 2),
                'breakout_strength': round(breakout_strength, 2),
            })

        ui_status.caption('Daily scan: done')
        st.subheader(f'Daily Signale (RiskOn={risk_on})')
        df_sig = pd.DataFrame(rows)
        if not df_sig.empty:
            df_sig['Entry-Nähe ★'] = (10.0 / (1.0 + df_sig['breakout_strength'])).round(1)
            df_sig['Follow-Through ★'] = (df_sig['breakout_strength'] * 2.0 * df_sig['vol_ratio'].pow(0.5)).clip(upper=10.0).round(1)
            df_sig = df_sig.drop(columns=['breakout_strength']).sort_values('Follow-Through ★', ascending=False)

            def _color_score(val):
                ratio = min(1.0, float(val) / 10.0)
                r = int(255 * (1.0 - ratio))
                g = int(200 * ratio)
                return f'background-color: rgba({r},{g},80,0.35)'

            styled = df_sig.style.map(_color_score, subset=['Entry-Nähe ★', 'Follow-Through ★'])
            col_cfg = {
                'symbol': st.column_config.TextColumn('Symbol', help='Aktien-Ticker-Symbol'),
                'side': st.column_config.TextColumn('Richtung', help='Handelsrichtung (LONG/SHORT)'),
                'price': st.column_config.NumberColumn('Kurs', help='Letzter Schlusskurs', format='%.2f'),
                'breakout_level': st.column_config.NumberColumn('Ausbruchsniveau', help='Breakout-Level (Close- oder High-Basis je nach `breakout_level_source`)', format='%.2f'),
                'asof': st.column_config.TextColumn('Datum', help='Datum des letzten Handelstags'),
                'atr': st.column_config.NumberColumn('ATR', help='Average True Range (14 Tage) – Maß für Volatilität', format='%.2f'),
                'risk_per_share': st.column_config.NumberColumn('Risiko/Aktie', help='Risiko je Aktie = ATR × Stop-Multiplikator', format='%.2f'),
                'stop_price': st.column_config.NumberColumn('Stop-Loss', help='Stop-Loss-Kurs = Kurs − Risiko/Aktie', format='%.2f'),
                'tp_price': st.column_config.NumberColumn('Take-Profit', help='Take-Profit = Kurs + 2 × Risiko/Aktie', format='%.2f'),
                'shares_for_1000eur': st.column_config.NumberColumn('Stück/1000€', help='Stückzahl bei 1.000 € Konto und 1 % Risiko pro Trade'),
                'vol_ratio': st.column_config.NumberColumn('Vol-Ratio', help='Heutiges Volumen / 50-Tage-Ø-Volumen. >1 = überdurchschnittliches Volumen', format='%.2f'),
                'Entry-Nähe ★': st.column_config.NumberColumn('Entry-Nähe ★', help='Nähe zum Ausbruchsniveau (0–10). Höher = weniger überschossen, sichererer Einstieg. 10 = direkt am Level.', format='%.1f'),
                'Follow-Through ★': st.column_config.NumberColumn('Follow-Through ★', help='Ausbruchsstärke inkl. Volumen (0–10). Höher = stärkerer Ausbruch mit gutem Volumen. Sortierung nach diesem Wert.', format='%.1f'),
            }
            st.dataframe(styled, column_config=col_cfg, use_container_width=True)
            st.download_button(
                '⬇ Download daily_signals.csv',
                data=df_sig.to_csv(index=False).encode('utf-8'),
                file_name='daily_signals.csv',
                mime='text/csv',
            )
        else:
            st.info('Keine Daily-Signale gefunden.')
