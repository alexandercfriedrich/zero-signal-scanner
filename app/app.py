import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

from backtest_engine import run_backtest, run_short_backtest, rsi

st.set_page_config(page_title='3S- Stock Signal Scanner by AF', layout='wide')

CACHE_DIR = Path.home() / '.zero_swing_cache'
CACHE_DIR.mkdir(exist_ok=True)

_TODAY = datetime.date.today().isoformat()

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
GOLD = '#F59E0B'
EMERALD = '#10B981'
RUBY = '#EF4444'
SAPPHIRE = '#3B82F6'
AMETHYST = '#8B5CF6'
BG_DARK = '#0B0F14'
BG_CARD = '#0F1621'
TEXT_COLOR = '#E6EAF2'
TEXT_DIM = '#8B95A5'

# ---------------------------------------------------------------------------
# Premium CSS injection
# ---------------------------------------------------------------------------
PREMIUM_CSS = """
<style>
/* --- Global overrides --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #070A0E 0%, #0B0F14 40%, #0D1117 100%) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0B0F14; }
::-webkit-scrollbar-thumb { background: #2A3040; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #F59E0B; }

/* --- Header styling --- */
.premium-header {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(15,22,33,0.9) 0%, rgba(11,15,20,0.9) 100%);
    border-bottom: 1px solid rgba(245,158,11,0.15);
    border-radius: 0 0 16px 16px;
}
.premium-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F59E0B, #FBBF24, #F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.premium-header p {
    color: #8B95A5;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 300;
}

/* --- Glass-morphism metric cards --- */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, rgba(15,22,33,0.85) 0%, rgba(20,28,40,0.75) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(245,158,11,0.12);
    border-radius: 12px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(245,158,11,0.35);
    box-shadow: 0 0 20px rgba(245,158,11,0.08);
    transform: translateY(-2px);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(245,158,11,0.4), transparent);
}
.metric-card .label {
    font-size: 0.72rem;
    color: #8B95A5;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.metric-card .value.positive { color: #10B981; }
.metric-card .value.negative { color: #EF4444; }
.metric-card .value.neutral { color: #F59E0B; }
.metric-card .icon {
    position: absolute;
    top: 12px; right: 14px;
    font-size: 1.4rem;
    opacity: 0.4;
}

/* --- Section headers with gradient underline --- */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #E6EAF2;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, #F59E0B, transparent) 1;
}

/* --- Animated gradient border panels --- */
.glass-panel {
    background: linear-gradient(135deg, rgba(15,22,33,0.8) 0%, rgba(20,28,40,0.7) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(245,158,11,0.10);
    border-radius: 14px;
    padding: 20px;
    margin: 12px 0;
    position: relative;
    overflow: hidden;
}
.glass-panel::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px; bottom: -1px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(245,158,11,0.15), transparent, rgba(59,130,246,0.10));
    z-index: -1;
}

/* --- Signal cards (expandable) --- */
.signal-card {
    background: linear-gradient(135deg, rgba(15,22,33,0.85) 0%, rgba(20,28,40,0.75) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(245,158,11,0.08);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: all 0.3s ease;
}
.signal-card:hover {
    border-color: rgba(245,158,11,0.25);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-long { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
.badge-short { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-neutral { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }

/* --- Data table styling --- */
.stDataFrame > div > div > div > div > div > table {
    border-collapse: separate;
    border-spacing: 0;
}
.stDataFrame thead th {
    background: rgba(15,22,33,0.9) !important;
    color: #F59E0B !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border-bottom: 2px solid rgba(245,158,11,0.2) !important;
}
.stDataFrame tbody tr:hover td {
    background: rgba(245,158,11,0.04) !important;
}

/* --- Streamlit tabs modernize --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: rgba(15,22,33,0.5);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    color: #8B95A5;
    transition: all 0.3s;
}
.stTabs [aria-selected="true"] {
    background: rgba(245,158,11,0.12) !important;
    color: #F59E0B !important;
    border-bottom-color: transparent !important;
}

/* --- Sidebar styling --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0F14 0%, #0D1218 100%) !important;
    border-right: 1px solid rgba(245,158,11,0.08);
}
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #F59E0B, #D97706) !important;
    color: #0B0F14 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    box-shadow: 0 0 20px rgba(245,158,11,0.3) !important;
    transform: translateY(-1px) !important;
}

/* --- Download buttons --- */
.stDownloadButton button {
    background: transparent !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    color: #F59E0B !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s !important;
}
.stDownloadButton button:hover {
    background: rgba(245,158,11,0.08) !important;
    border-color: #F59E0B !important;
}

/* --- Footer --- */
.app-footer {
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 3rem;
    border-top: 1px solid rgba(245,158,11,0.08);
    color: #555;
    font-size: 0.75rem;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] {
    background: transparent !important;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Premium metric cards HTML
# ---------------------------------------------------------------------------
def render_metric_cards(metrics: list[dict]):
    """Render a row of glass-morphism metric cards.
    Each dict: {label, value, icon, css_class} where css_class in [positive, negative, neutral].
    """
    cards_html = '<div class="metric-row">'
    for m in metrics:
        css = m.get('css_class', 'neutral')
        icon = m.get('icon', '')
        cards_html += f'''
        <div class="metric-card">
            <div class="icon">{icon}</div>
            <div class="label">{m['label']}</div>
            <div class="value {css}">{m['value']}</div>
        </div>'''
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Dark-themed Plotly layout
# ---------------------------------------------------------------------------
def apply_dark_layout(fig, title='', height=450):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(11,15,20,0.6)',
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR, family='Inter'),
                   x=0.02, y=0.98),
        font=dict(family='Inter', color=TEXT_DIM, size=11),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        xaxis=dict(gridcolor='rgba(42,48,64,0.3)', zerolinecolor='rgba(42,48,64,0.3)'),
        yaxis=dict(gridcolor='rgba(42,48,64,0.3)', zerolinecolor='rgba(42,48,64,0.3)'),
    )
    return fig


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
  "start": "2021-01-01",
  "end": _TODAY,
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
  "trailing_reference": "close",
  "take_profit_R": 2.0,

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
  "initial_cash": 5000,

  # Short-Scan Parameter
  "short_rsi_min": 75,
  "short_ema20_dist_min": 0.12,
  "short_5d_perf_min": 0.10,
  "short_vol_mult_min": 1.0,
}

# Presets (loaded from JSON files in app/)
PRESETS = {
    'Swing (Top-5, Risk-On only)': None,  # uses DEFAULT_CFG
    'Best (2011-2026)': 'config_best_2011_2026.json',
    'Best (2011-2026) \u2013 mehr Signale': 'config_best_2011_2026_mehr_signale.json',
    'Best (2011-2026) \u2013 h\u00f6here Trefferquote': 'config_best_2011_2026_hoehere_trefferquote.json',
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
    """Load preset config dict, always including DEFAULT_CFG keys."""
    p = PRESETS.get(preset_name)
    if not p:
        cfg = dict(DEFAULT_CFG)
        cfg['end'] = _TODAY
        return cfg

    preset_path = (Path(__file__).parent / p)
    try:
        raw = json.loads(preset_path.read_text())
        if not isinstance(raw, dict):
            cfg = dict(DEFAULT_CFG)
            cfg['end'] = _TODAY
            return cfg
        merged = _merge_cfg(DEFAULT_CFG, raw)
        merged['end'] = _TODAY
        return merged
    except Exception:
        cfg = dict(DEFAULT_CFG)
        cfg['end'] = _TODAY
        return cfg


# ---------------------------------------------------------------------------
# Premium header
# ---------------------------------------------------------------------------
st.markdown('''
<div class="premium-header">
    <h1>3S \u2013 Stock Signal Scanner</h1>
    <p>Premium Swing-Trading Analyse \u2022 Signale \u2022 Backtesting &emsp; | &emsp; by AF</p>
</div>
''', unsafe_allow_html=True)

st.markdown("""
<div class="glass-panel">
<strong>Was macht dieser Scanner?</strong><br>
Der 3S Stock Signal Scanner durchsucht t\u00e4glich Aktien aus verschiedenen Indizes nach technischen Signalen auf Tagesbasis.<br><br>
<span class="signal-badge badge-long">LONG</span> Daily Close-Breakout \u00fcber das 55-Tage-Hoch &emsp;
<span class="signal-badge badge-short">SHORT</span> \u00dcberhitzte Aktien mit RSI > 75, Abstand zur EMA20 > 12\u00a0%, 5-Tage-Performance > 10\u00a0%, hohes Volumen
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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
                'Alternativ k\u00f6nnen auch Firmenname, ISIN oder WKN eingegeben werden; '
                'der Resolver versucht diese automatisch in Yahoo-Ticker umzuwandeln.'
            ),
        )

    st.header('Aktion')
    action = st.radio(
        'Modus',
        ['Daily Long Signalscan', 'Daily Short Signalscan', 'Backtest (Daily)'],
        index=0,
    )

    st.header('Resolver Pr\u00e4ferenzen')
    prefer_regions = st.multiselect('Yahoo region Reihenfolge', ['DE', 'AT', 'US', 'GB'], default=['DE', 'AT', 'US'])
    throttle_s = st.slider('Resolver Throttle (Sek.)', 0.0, 1.0, 0.2, 0.05)

    st.header('Konfiguration')

    preset = st.selectbox('Preset', list(PRESETS.keys()), index=1)

    if 'cfg_text' not in st.session_state:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)

    prev_preset = st.session_state.get('preset_name')
    if prev_preset != preset:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)
        st.session_state['preset_name'] = preset

    cfg_text = st.text_area('config.json (ohne symbols)', value=st.session_state['cfg_text'], height=420)

    with st.expander('\u2139\ufe0f Konfigurations-Parameter'):
        st.markdown("""
| Parameter | Standard | Beschreibung |
|---|---|---|
| `start` | `"2021-01-01"` | Startdatum (YYYY-MM-DD) |
| `end` | *(heute)* | Enddatum \u2013 wird automatisch auf heute gesetzt |
| `regime_symbol` | `"SPY"` | Regime/Benchmark-Symbol |
| `inverse_map` | `{ "SPY": "SH" }` | Mapping Regime\u2192Inverse-ETF |
| `hard_risk_on` | `true` | Keine neuen Long-Trades bei Risk-Off |
| `max_new_trades_per_day` | `2` | Max. neue Einstiege pro Tag |
| `max_positions` | `5` | Max. gleichzeitige Positionen |
| `weekly_rerank` | `true` | W\u00f6chentlich neu ranken |
| `weekly_rebalance_weekday` | `0` | Wochentag f\u00fcrs Rerank (0=Mo \u2026 4=Fr) |
| `risk_per_trade` | `0.01` | Risiko pro Trade (Anteil vom Equity) |
| `atr_period` | `14` | ATR-Periode |
| `atr_stop_mult` | `2.0` | Initialer Stop = ATR \u00d7 Mult |
| `use_trailing_stop` | `true` | Trailing-Stop aktiv |
| `atr_trail_mult` | `2.5` | Trailing-Stop = ATR \u00d7 Mult |
| `trailing_reference` | `"close"` | `"high"` = Chandelier, `"close"` = Schlusskurs |
| `take_profit_R` | `2.0` | Take-Profit in R (nur ohne Trailing) |
| `breakout_lookback` | `55` | Lookback f\u00fcr Breakout-Level |
| `breakout_level_source` | `"close"` | `"close"` oder `"high"` |
| `breakout_confirm_closes` | `1` | Anzahl Closes \u00fcber Level |
| `sma_regime` | `200` | SMA-L\u00e4nge f\u00fcr Regime-Filter |
| `max_holding_days` | `30` | Max. Haltedauer (Tage) |
| `min_breakout_vol_mult` | `0.0` | Vol \u2265 N \u00d7 VolSMA50 (0 = aus) |
| `rsi_period` | `0` | RSI-Periode f\u00fcr Long-Filter (0 = aus) |
| `rsi_max` | `100` | Max. RSI beim Long-Einstieg |
| `max_breakout_extension_atr` | `1e9` | Max. Ausdehnung \u00fcber Level in ATR |
| `mom_lookback` | `126` | Momentum-Lookback (Tage) |
| `enable_cwh` | `true` | Cup-with-Handle aktivieren |
| `cwh_cup_min_bars` | `30` | CWH: min. Cup-L\u00e4nge |
| `cwh_cup_max_bars` | `130` | CWH: max. Cup-L\u00e4nge |
| `cwh_handle_min_bars` | `5` | CWH: min. Handle-L\u00e4nge |
| `cwh_handle_max_bars` | `20` | CWH: max. Handle-L\u00e4nge |
| `cwh_max_cup_depth` | `0.35` | CWH: max. Cup-Tiefe |
| `cwh_max_handle_depth` | `0.15` | CWH: max. Handle-Tiefe |
| `cwh_trend_sma` | `50` | CWH: Trendfilter SMA |
| `cwh_vol_bonus` | `0.3` | CWH: Volumen-Qualit\u00e4ts-Bonus |
| `corr_lookback_days` | `60` | Lookback f\u00fcr Korrelationsfilter |
| `max_pair_corr` | `1.0` | Max. Korrelation (1.0 = aus) |
| `max_positions_per_sector` | `999` | Max. Positionen pro Sektor |
| `spread_bps_per_side` | `8` | Spread in Basispunkten |
| `min_price` | `2.0` | Mindestkurs |
| `min_dollar_volume` | `2000000` | Mindest-Dollar-Volume |
| `initial_cash` | `5000` | Startkapital |
| `short_rsi_min` | `75` | **Short-Scan**: Min. RSI |
| `short_ema20_dist_min` | `0.12` | **Short-Scan**: Min. Abstand zur EMA20 (z.B. 0.12 = 12\u00a0%) |
| `short_5d_perf_min` | `0.10` | **Short-Scan**: Min. 5-Tage-Performance |
| `short_vol_mult_min` | `1.0` | **Short-Scan**: Min. Vol-Ratio (Vol / VolSMA50) |
""")

    st.markdown("---")
    force_refresh = st.checkbox('\u267b\ufe0f Force Refresh (Cache ignorieren)', value=False,
                                help='Alle gecachten Daten neu herunterladen')
    run_btn = st.button('Start', type='primary')


# ---------------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_sp500_symbols() -> list[str]:
    wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        r = requests.get(wiki_url, timeout=25, headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_nasdaq100_symbols() -> list[str]:
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
    result = []
    for s in syms:
        if '.' not in s and len(s) <= 6:
            result.append(s + suffix)
        else:
            result.append(s)
    return result


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_dax40_symbols() -> list[str]:
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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_atx_symbols() -> list[str]:
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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_sp500_sector_map() -> dict:
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


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
def yahoo_search(query: str, region: str, lang: str):
    url = 'https://query2.finance.yahoo.com/v1/finance/search'
    params = {'q': query, 'quotesCount': 6, 'newsCount': 0, 'listsCount': 0,
               'region': region, 'lang': lang, 'enableFuzzyQuery': 'false'}
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
    prefs = ['.DE', '.VI', '.PA', '.AS', '.MI', '.BR', '.LS', '.S', '.ST', '.OL', '.CO', '.HE']
    for i, suf in enumerate(prefs):
        if sym.endswith(suf):
            return 100 - i
    return 0


def resolve_one(query: str, regions: list[str]) -> str | None:
    q = query.strip()
    if not q:
        return None
    if any(ch in q for ch in ['.', '-', '=']) or (q.isalpha() and 1 <= len(q) <= 7):
        return q.upper()
    best = None
    best_score = -1
    for region in regions:
        lang = {'DE': 'de-DE', 'AT': 'de-AT', 'US': 'en-US', 'GB': 'en-GB'}.get(region, 'en-US')
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


# ---------------------------------------------------------------------------
# Smart caching
# ---------------------------------------------------------------------------
def cache_path(sym: str, kind: str) -> Path:
    safe = sym.replace('^', '_').replace('/', '_')
    return CACHE_DIR / f'{safe}.{kind}.parquet'


def _cache_is_fresh(p: Path, start: str) -> bool:
    """Check if cache is fresh enough: data must cover start and be from today or yesterday."""
    if not p.exists():
        return False
    try:
        d = pd.read_parquet(p)
        d['Date'] = pd.to_datetime(d['Date'])
        cache_end = d['Date'].max()
        today_ts = pd.Timestamp(_TODAY)
        # Cache is fresh if it includes data from today or the previous trading day (1 day buffer)
        if d['Date'].min() <= pd.Timestamp(start) and cache_end >= today_ts - pd.Timedelta(days=1):
            return True
    except Exception:
        pass
    return False


def fetch_yf(symbols, start=None, end=None, interval='1d', period=None):
    if period is not None:
        return yf.download(symbols, period=period, interval=interval, auto_adjust=False,
                           group_by='ticker', threads=True, progress=False)
    return yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=False,
                       group_by='ticker', threads=True, progress=False)


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


def load_daily(symbols: list[str], start: str, end: str,
               progress_cb=None, force=False) -> dict[str, pd.DataFrame]:
    out = {}
    need = []

    for s in symbols:
        p = cache_path(s, '1d')
        if not force and _cache_is_fresh(p, start):
            try:
                d = pd.read_parquet(p)
                d['Date'] = pd.to_datetime(d['Date'])
                out[s] = d
                continue
            except Exception:
                pass
        need.append(s)

    if need:
        chunk = 80
        total = int(np.ceil(len(need) / chunk))
        for k, i in enumerate(range(0, len(need), chunk)):
            batch = need[i:i + chunk]
            if progress_cb is not None:
                progress_cb(k, total, f'Daily: {i}/{len(need)}')
            got = unpack_download(fetch_yf(batch, start=start, end=end, interval='1d'), batch)
            for sym, sub in got.items():
                out[sym] = sub
                sub.to_parquet(cache_path(sym, '1d'), index=False)
        if progress_cb is not None:
            progress_cb(total, total, 'Daily: done')

    return out


# ---------------------------------------------------------------------------
# Short-Scan helper
# ---------------------------------------------------------------------------
def run_short_scan(daily: dict, cfg: dict) -> pd.DataFrame:
    """Scan for exhausted/overbought stocks as short candidates."""
    rsi_min = float(cfg.get('short_rsi_min', 75))
    ema_dist_min = float(cfg.get('short_ema20_dist_min', 0.12))
    perf5_min = float(cfg.get('short_5d_perf_min', 0.10))
    vol_mult_min = float(cfg.get('short_vol_mult_min', 1.0))
    atr_p = int(cfg.get('atr_period', 14))
    bl_lookback = int(cfg.get('breakout_lookback', 55))
    bl_src = cfg.get('breakout_level_source', 'close')

    rows = []
    for sym in cfg.get('symbols', []):
        if sym not in daily:
            continue
        df = daily[sym].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        if len(df) < 30:
            continue

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        # Indicators
        ema20 = close.ewm(span=20, adjust=False).mean()
        rsi_series = rsi(close, 14)
        vol_sma50 = vol.rolling(50).mean()

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_series = tr.rolling(atr_p).mean()

        # Breakout level (last pivot = potential TP zone)
        if bl_src == 'high':
            bl_series = high.shift(1).rolling(bl_lookback).max()
        else:
            bl_series = close.shift(1).rolling(bl_lookback).max()

        # Latest values
        px = float(close.iloc[-1])
        ema20_v = float(ema20.iloc[-1])
        rsi_v = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else np.nan
        vol_today = float(vol.iloc[-1])
        vol_sma50_v = float(vol_sma50.iloc[-1]) if not pd.isna(vol_sma50.iloc[-1]) else np.nan
        atr_v = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else np.nan
        bl_v = float(bl_series.iloc[-1]) if not pd.isna(bl_series.iloc[-1]) else np.nan

        # 5-day performance
        if len(close) < 6:
            continue
        perf5 = float(close.iloc[-1] / close.iloc[-6] - 1)

        # Filters
        if pd.isna(rsi_v) or rsi_v < rsi_min:
            continue
        if ema20_v <= 0 or (px / ema20_v - 1) < ema_dist_min:
            continue
        if perf5 < perf5_min:
            continue
        vol_ratio = vol_today / vol_sma50_v if (vol_sma50_v and vol_sma50_v > 0) else 0.0
        if vol_ratio < vol_mult_min:
            continue
        if px <= 0 or pd.isna(atr_v) or atr_v <= 0:
            continue

        # TP zones
        tp_ema20 = round(ema20_v, 2)
        tp_breakout = round(bl_v, 2) if not pd.isna(bl_v) else np.nan

        # Fib 38.2%: from recent swing high to swing low in last 20 bars
        recent_high = float(high.iloc[-20:].max())
        recent_low = float(low.iloc[-20:].min())
        fib38 = round(recent_high - 0.382 * (recent_high - recent_low), 2)

        # Short stop-loss above recent high (ATR-based)
        short_stop = round(recent_high + atr_v, 2)

        # Überhitzungs-Score (0–10): RSI-Überschuss + EMA-Abstand + 5d-Perf kombiniert
        rsi_score = min(1.0, (rsi_v - rsi_min) / 25.0)
        ema_score = min(1.0, (px / ema20_v - 1) / 0.30)
        perf_score = min(1.0, perf5 / 0.30)
        vol_score = min(1.0, (vol_ratio - 1.0) / 3.0)
        ueberhitzung = round((rsi_score * 3 + ema_score * 3 + perf_score * 2 + vol_score * 2), 1)
        ueberhitzung = round(min(10.0, ueberhitzung), 1)

        rows.append({
            'symbol': sym,
            'side': 'SHORT',
            'price': round(px, 2),
            'asof': str(df.index[-1].date()),
            'rsi': round(rsi_v, 1),
            'ema20_dist_%': round((px / ema20_v - 1) * 100, 1),
            '5d_perf_%': round(perf5 * 100, 1),
            'vol_ratio': round(vol_ratio, 2),
            'atr': round(atr_v, 2),
            'short_stop': short_stop,
            'tp_ema20': tp_ema20,
            'tp_breakout_level': tp_breakout,
            'tp_fib38': fib38,
            'ueberhitzung_score': ueberhitzung,
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values('ueberhitzung_score', ascending=False)
    return df_out


# ---------------------------------------------------------------------------
# Signal chart builder (candlestick + RSI + Volume + TP zones)
# ---------------------------------------------------------------------------
def build_signal_chart(daily_df: pd.DataFrame, sym: str, signal_row: dict,
                       side: str = 'SHORT', n_bars: int = 60) -> go.Figure:
    """Build a multi-subplot chart for a signal candidate."""
    df = daily_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df.iloc[-n_bars:]

    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']
    ema20 = close.ewm(span=20, adjust=False).mean()
    rsi_series = rsi(close, 14)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=None,
    )

    # Candlestick
    colors = [EMERALD if c >= o else RUBY
              for o, c in zip(df['Open'], close)]
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=high, low=low, close=close,
        increasing=dict(line=dict(color=EMERALD), fillcolor='rgba(16,185,129,0.3)'),
        decreasing=dict(line=dict(color=RUBY), fillcolor='rgba(239,68,68,0.3)'),
        name='Kurs', showlegend=False,
    ), row=1, col=1)

    # EMA20 overlay with glow
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20, mode='lines',
        line=dict(color='rgba(245,158,11,0.15)', width=6),
        name='EMA20 glow', showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20, mode='lines',
        line=dict(color=GOLD, width=2),
        name='EMA20',
    ), row=1, col=1)

    # TP zones and stop markers
    if side == 'SHORT':
        tp_ema20 = signal_row.get('tp_ema20')
        tp_bl = signal_row.get('tp_breakout_level')
        tp_fib = signal_row.get('tp_fib38')
        stop = signal_row.get('short_stop')
        entry = signal_row.get('price')

        if entry:
            fig.add_hline(y=entry, line=dict(color=AMETHYST, width=1.5, dash='dash'),
                          annotation_text='Entry', annotation_font_color=AMETHYST,
                          row=1, col=1)
        if stop and not pd.isna(stop):
            fig.add_hline(y=stop, line=dict(color=RUBY, width=1.5, dash='dash'),
                          annotation_text='Stop-Loss', annotation_font_color=RUBY,
                          row=1, col=1)
        if tp_ema20 and not pd.isna(tp_ema20):
            fig.add_hline(y=tp_ema20, line=dict(color=EMERALD, width=1, dash='dot'),
                          annotation_text='TP: EMA20', annotation_font_color=EMERALD,
                          row=1, col=1)
        if tp_bl and not pd.isna(tp_bl):
            fig.add_hline(y=tp_bl, line=dict(color=SAPPHIRE, width=1, dash='dot'),
                          annotation_text='TP: Breakout', annotation_font_color=SAPPHIRE,
                          row=1, col=1)
        if tp_fib and not pd.isna(tp_fib):
            fig.add_hline(y=tp_fib, line=dict(color=GOLD, width=1, dash='dot'),
                          annotation_text='TP: Fib38%', annotation_font_color=GOLD,
                          row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_series, mode='lines',
        line=dict(color=AMETHYST, width=1.5),
        name='RSI(14)',
    ), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(239,68,68,0.4)', width=1, dash='dash'),
                  row=2, col=1)
    fig.add_hline(y=30, line=dict(color='rgba(16,185,129,0.4)', width=1, dash='dash'),
                  row=2, col=1)
    # RSI fill overbought zone
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.05)', line_width=0,
                  row=2, col=1)

    # Volume bars
    vol_colors = [EMERALD if c >= o else RUBY
                  for o, c in zip(df['Open'], close)]
    fig.add_trace(go.Bar(
        x=df.index, y=vol, name='Volumen',
        marker=dict(color=vol_colors, opacity=0.6),
        showlegend=False,
    ), row=3, col=1)

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(11,15,20,0.5)',
        font=dict(family='Inter', color=TEXT_DIM, size=10),
        height=500,
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.02,
                    font=dict(size=9)),
        xaxis3=dict(gridcolor='rgba(42,48,64,0.2)'),
        yaxis=dict(gridcolor='rgba(42,48,64,0.2)', title='Kurs'),
        yaxis2=dict(gridcolor='rgba(42,48,64,0.2)', title='RSI'),
        yaxis3=dict(gridcolor='rgba(42,48,64,0.2)', title='Vol'),
        xaxis_rangeslider_visible=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Backtest chart helpers
# ---------------------------------------------------------------------------
def build_equity_chart(equity_df: pd.DataFrame, title: str = 'Equity Curve') -> go.Figure:
    eq = equity_df.reset_index()
    fig = go.Figure()

    # Gradient fill area
    fig.add_trace(go.Scatter(
        x=eq['Date'], y=eq['Equity'], mode='lines',
        line=dict(color='rgba(245,158,11,0.1)', width=0),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.05)',
        showlegend=False,
    ))

    # Glow line
    fig.add_trace(go.Scatter(
        x=eq['Date'], y=eq['Equity'], mode='lines',
        line=dict(color='rgba(245,158,11,0.15)', width=5),
        showlegend=False,
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=eq['Date'], y=eq['Equity'], mode='lines',
        line=dict(color=GOLD, width=2),
        name='Equity',
    ))

    apply_dark_layout(fig, title, height=380)
    fig.update_yaxes(title='Kapital (\u20ac)')
    return fig


def build_drawdown_chart(equity_df: pd.DataFrame, title: str = 'Drawdown') -> go.Figure:
    eq = equity_df['Equity']
    dd = (eq / eq.cummax() - 1) * 100
    dd_df = dd.reset_index()
    dd_df.columns = ['Date', 'Drawdown']

    fig = go.Figure()

    # Fill
    fig.add_trace(go.Scatter(
        x=dd_df['Date'], y=dd_df['Drawdown'], mode='lines',
        line=dict(color=RUBY, width=1.5),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.12)',
        name='Drawdown',
    ))

    apply_dark_layout(fig, title, height=280)
    fig.update_yaxes(title='Drawdown (%)')
    return fig


def build_monthly_returns_heatmap(equity_df: pd.DataFrame) -> go.Figure:
    eq = equity_df['Equity'].copy()
    eq.index = pd.to_datetime(eq.index)

    monthly = eq.resample('ME').last()
    monthly_ret = monthly.pct_change().dropna() * 100

    years = monthly_ret.index.year.unique()
    months = list(range(1, 13))
    month_labels = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']

    z = []
    for y in sorted(years):
        row = []
        for m in months:
            vals = monthly_ret[(monthly_ret.index.year == y) & (monthly_ret.index.month == m)]
            row.append(float(vals.iloc[0]) if len(vals) > 0 else np.nan)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=month_labels, y=[str(y) for y in sorted(years)],
        colorscale=[
            [0.0, RUBY],
            [0.5, '#1a1a2e'],
            [1.0, EMERALD],
        ],
        zmid=0,
        text=[[f'{v:.1f}%' if not pd.isna(v) else '' for v in row] for row in z],
        texttemplate='%{text}',
        textfont=dict(size=10, color=TEXT_COLOR),
        hoverongaps=False,
        colorbar=dict(title='%', tickfont=dict(color=TEXT_DIM)),
    ))

    apply_dark_layout(fig, 'Monatliche Renditen (%)', height=max(200, len(years) * 35 + 80))
    fig.update_xaxes(side='top')
    return fig


def build_trade_distribution(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        fig = go.Figure()
        apply_dark_layout(fig, 'Keine Trades', 250)
        return fig

    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] < 0])
    flat = len(trades_df[trades_df['pnl'] == 0])

    labels = ['Gewinn', 'Verlust', 'Neutral']
    values = [wins, losses, flat]
    colors = [EMERALD, RUBY, GOLD]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG_DARK, width=2)),
        textinfo='label+percent',
        textfont=dict(size=11, color=TEXT_COLOR),
    ))

    apply_dark_layout(fig, 'Win/Loss Verteilung', height=300)
    return fig


def build_pnl_histogram(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        fig = go.Figure()
        apply_dark_layout(fig, 'Keine Trades', 250)
        return fig

    pnl = trades_df['pnl']
    colors = [EMERALD if v > 0 else RUBY for v in pnl]

    fig = go.Figure(go.Histogram(
        x=pnl, nbinsx=30,
        marker=dict(color=GOLD, opacity=0.7, line=dict(color=GOLD, width=0.5)),
        name='PnL',
    ))

    apply_dark_layout(fig, 'PnL Verteilung', height=280)
    fig.update_xaxes(title='PnL (\u20ac)')
    fig.update_yaxes(title='Anzahl Trades')
    return fig


# ---------------------------------------------------------------------------
# Helper: display backtest results
# ---------------------------------------------------------------------------
def display_backtest_results(equity_df, trades_df, summary, breakdown, label=''):
    prefix = f'{label} ' if label else ''

    # Metric cards row 1
    cagr_v = summary.get('CAGR', 0)
    maxdd_v = summary.get('MaxDrawdown', 0)
    pf_v = summary.get('ProfitFactor', np.nan)
    wr_v = summary.get('WinRate', np.nan)
    exp_v = summary.get('Expectancy_R', np.nan)

    row1 = [
        {'label': f'{prefix}Endkapital', 'value': f"{summary['final_equity']:,.2f} \u20ac",
         'icon': '\U0001f4b0', 'css_class': 'positive' if summary['final_equity'] > summary['initial_cash'] else 'negative'},
        {'label': 'CAGR', 'value': f"{cagr_v * 100:.2f}%",
         'icon': '\U0001f4c8', 'css_class': 'positive' if cagr_v > 0 else 'negative'},
        {'label': 'Max. Drawdown', 'value': f"{maxdd_v * 100:.2f}%",
         'icon': '\U0001f4c9', 'css_class': 'negative' if maxdd_v < -0.1 else 'neutral'},
        {'label': 'Volatilit\u00e4t', 'value': f"{summary.get('Volatility', 0) * 100:.2f}%",
         'icon': '\U0001f30a', 'css_class': 'neutral'},
        {'label': 'Trades', 'value': str(summary.get('Trades', 0)),
         'icon': '\U0001f4ca', 'css_class': 'neutral'},
    ]
    render_metric_cards(row1)

    row2 = [
        {'label': 'Profit Factor', 'value': f"{pf_v:.2f}" if np.isfinite(pf_v) else '-',
         'icon': '\u2696\ufe0f', 'css_class': 'positive' if (np.isfinite(pf_v) and pf_v > 1) else 'negative'},
        {'label': 'Win Rate', 'value': f"{wr_v * 100:.1f}%" if not pd.isna(wr_v) else '-',
         'icon': '\U0001f3af', 'css_class': 'positive' if (not pd.isna(wr_v) and wr_v > 0.5) else 'negative'},
        {'label': 'Avg Win', 'value': f"{summary.get('AvgWin', 0):.2f} \u20ac" if not pd.isna(summary.get('AvgWin')) else '-',
         'icon': '\u2705', 'css_class': 'positive'},
        {'label': 'Avg Loss', 'value': f"{summary.get('AvgLoss', 0):.2f} \u20ac" if not pd.isna(summary.get('AvgLoss')) else '-',
         'icon': '\u274c', 'css_class': 'negative'},
        {'label': 'Expectancy (R)', 'value': f"{exp_v:.2f}" if np.isfinite(exp_v) else '-',
         'icon': '\U0001f52e', 'css_class': 'positive' if (np.isfinite(exp_v) and exp_v > 0) else 'negative'},
    ]
    render_metric_cards(row2)

    with st.expander('\u2139\ufe0f Kennzahlen-Erkl\u00e4rung'):
        st.markdown("""
| Kennzahl | Erkl\u00e4rung |
|---|---|
| **Endkapital** | Gesamtkapital am Ende des Backtest-Zeitraums |
| **CAGR** | J\u00e4hrliche Wachstumsrate |
| **Max. Drawdown** | Gr\u00f6\u00dfter prozentualer R\u00fcckgang |
| **Volatilit\u00e4t** | Annualisierte Standardabweichung der t\u00e4glichen Renditen |
| **Trades** | Gesamtanzahl abgeschlossener Trades |
| **Profit Factor** | Gesamtgewinn / Gesamtverlust (>1 = profitabel) |
| **Win Rate** | Anteil gewinnbringender Trades |
| **Avg Win** | Durchschnittlicher Gewinn je Trade (\u20ac) |
| **Avg Loss** | Durchschnittlicher Verlust je Trade (\u20ac) |
| **Expectancy (R)** | Erwartungswert je Trade in R; >0 = positiv |
""")

    # Charts
    if not equity_df.empty:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(build_equity_chart(equity_df, f'{prefix}Equity Curve'),
                            use_container_width=True)
        with c2:
            st.plotly_chart(build_drawdown_chart(equity_df, f'{prefix}Drawdown'),
                            use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(build_monthly_returns_heatmap(equity_df),
                            use_container_width=True)
        with c4:
            if not trades_df.empty:
                c4a, c4b = st.columns(2)
                with c4a:
                    st.plotly_chart(build_trade_distribution(trades_df),
                                   use_container_width=True)
                with c4b:
                    st.plotly_chart(build_pnl_histogram(trades_df),
                                   use_container_width=True)

    # Downloads
    st.markdown(f'<div class="section-header">{prefix}Downloads</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(f'\u2b07 Download {prefix.strip().lower() or ""}trades.csv',
                           data=trades_df.to_csv(index=False).encode('utf-8'),
                           file_name=f'{prefix.strip().lower()}_trades.csv' if prefix else 'trades.csv',
                           mime='text/csv')
    with dc2:
        st.download_button(f'\u2b07 Download {prefix.strip().lower() or ""}equity.csv',
                           data=equity_df.reset_index().to_csv(index=False).encode('utf-8'),
                           file_name=f'{prefix.strip().lower()}_equity.csv' if prefix else 'equity.csv',
                           mime='text/csv')

    # Breakdown
    st.markdown(f'<div class="section-header">{prefix}Breakdown</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    breakdown_col_cfg = {
        'setup': st.column_config.TextColumn('Setup'),
        'reason': st.column_config.TextColumn('Grund'),
        'Trades': st.column_config.NumberColumn('Trades'),
        'WinRate': st.column_config.NumberColumn('Win Rate', format='%.1f%%'),
        'ProfitFactor': st.column_config.NumberColumn('Profit Factor', format='%.2f'),
        'AvgPnL': st.column_config.NumberColumn('\u00d8 PnL', format='%.2f'),
        'AvgR': st.column_config.NumberColumn('\u00d8 R', format='%.2f'),
    }
    with b1:
        st.caption('Nach Setup')
        st.dataframe(breakdown.get('by_setup', pd.DataFrame()),
                     column_config=breakdown_col_cfg, use_container_width=True)
    with b2:
        st.caption('Nach Exit-Grund')
        st.dataframe(breakdown.get('by_reason', pd.DataFrame()),
                     column_config=breakdown_col_cfg, use_container_width=True)

    # Trades table
    st.markdown(f'<div class="section-header">{prefix}Trades</div>', unsafe_allow_html=True)
    trades_col_cfg = {
        'symbol': st.column_config.TextColumn('Symbol'),
        'side': st.column_config.TextColumn('Richtung'),
        'entry_date': st.column_config.TextColumn('Einstieg'),
        'entry_px': st.column_config.NumberColumn('Einstiegskurs', format='%.2f'),
        'exit_date': st.column_config.TextColumn('Ausstieg'),
        'exit_px': st.column_config.NumberColumn('Ausstiegskurs', format='%.2f'),
        'shares': st.column_config.NumberColumn('St\u00fcck'),
        'pnl': st.column_config.NumberColumn('PnL (\u20ac)', format='%.2f'),
        'reason': st.column_config.TextColumn('Exitgrund'),
        'setup': st.column_config.TextColumn('Setup'),
        'initial_risk_per_share': st.column_config.NumberColumn('Init. Risiko/Aktie', format='%.2f'),
        'R_multiple': st.column_config.NumberColumn('R-Vielfaches', format='%.2f'),
    }
    st.dataframe(trades_df, column_config=trades_col_cfg, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app logic
# ---------------------------------------------------------------------------
if run_btn:
    cfg = json.loads(cfg_text)
    # Always enforce today as end date
    cfg['end'] = _TODAY

    with st.spinner('Universe laden & ggf. aufl\u00f6sen...'):
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

    with st.expander('Symbol\u2011Aufl\u00f6sung', expanded=False):
        st.dataframe(resolve_table, use_container_width=True)

    cfg = dict(cfg)
    cfg['symbols'] = [s for s in symbols
                      if s not in [cfg['regime_symbol']]
                      and s not in cfg.get('inverse_map', {}).values()]

    sector_map = {}
    if universe_mode == 'S&P 500 (Wikipedia)':
        sector_map = load_sp500_sector_map()
    cfg['sector_map'] = sector_map

    needed_daily = sorted(list(set(
        cfg['symbols'] + [cfg['regime_symbol']] + list(cfg.get('inverse_map', {}).values())
    )))

    ui_prog = st.progress(0.0)
    ui_status = st.empty()

    def prog_step(done, total, msg):
        frac = 0.0 if total <= 0 else float(done) / float(total)
        ui_prog.progress(min(1.0, max(0.0, frac)))
        ui_status.caption(msg)

    # -----------------------------------------------------------------------
    # BACKTEST
    # -----------------------------------------------------------------------
    if action == 'Backtest (Daily)':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m),
                           force=force_refresh)
        cfg['symbols'] = [s for s in cfg['symbols'] if s in daily]

        # --- LONG BACKTEST ---
        st.markdown('<div class="section-header">\U0001f4c8 Long Backtest</div>',
                    unsafe_allow_html=True)
        ui_status.caption('Long Backtest l\u00e4uft...')
        bt_prog = st.progress(0.0)
        bt_status = st.empty()

        def bt_step(done, total, date_or_msg):
            frac = 0.0 if total <= 0 else float(done) / float(total)
            bt_prog.progress(min(1.0, max(0.0, frac)))
            if isinstance(date_or_msg, str):
                bt_status.caption(date_or_msg)
            else:
                bt_status.caption(f'Long Backtest: {done}/{total}  ({date_or_msg.date().isoformat()})')

        equity_df, trades_df, summary, breakdown = run_backtest(daily, cfg, progress_cb=bt_step)
        bt_status.caption('Long Backtest: done')

        display_backtest_results(equity_df, trades_df, summary, breakdown, label='\U0001f4c8 Long')

        # --- SHORT BACKTEST ---
        st.markdown('---')
        st.markdown('<div class="section-header">\U0001f4c9 Short Backtest</div>',
                    unsafe_allow_html=True)
        ui_status.caption('Short Backtest l\u00e4uft...')
        sbt_prog = st.progress(0.0)
        sbt_status = st.empty()

        def sbt_step(done, total, date_or_msg):
            frac = 0.0 if total <= 0 else float(done) / float(total)
            sbt_prog.progress(min(1.0, max(0.0, frac)))
            if isinstance(date_or_msg, str):
                sbt_status.caption(date_or_msg)
            else:
                sbt_status.caption(f'Short Backtest: {done}/{total}  ({date_or_msg.date().isoformat()})')

        s_equity_df, s_trades_df, s_summary, s_breakdown = run_short_backtest(
            daily, cfg, progress_cb=sbt_step)
        sbt_status.caption('Short Backtest: done')

        display_backtest_results(s_equity_df, s_trades_df, s_summary, s_breakdown, label='\U0001f4c9 Short')

    # -----------------------------------------------------------------------
    # DAILY LONG SIGNALSCAN
    # -----------------------------------------------------------------------
    elif action == 'Daily Long Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m),
                           force=force_refresh)

        reg = daily[cfg['regime_symbol']].copy()
        reg['Date'] = pd.to_datetime(reg['Date'])
        reg = reg.sort_values('Date').set_index('Date')
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
            df = daily[sym].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').set_index('Date')

            if bl_src == 'high':
                bl_series = df['High'].shift(1).rolling(cfg['breakout_lookback']).max()
            else:
                bl_series = df['Close'].shift(1).rolling(cfg['breakout_lookback']).max()

            high, low, close = df['High'], df['Low'], df['Close']
            tr = pd.concat([(high - low), (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr_v = tr.rolling(cfg['atr_period']).mean().iloc[-1]
            bl = bl_series.iloc[-1]
            if pd.isna(bl) or pd.isna(atr_v) or float(atr_v) <= 0:
                continue

            px = float(df['Close'].iloc[-1])
            if not risk_on or px <= float(bl):
                continue

            if scan_confirm >= 2:
                if len(df) < scan_confirm:
                    continue
                confirmed = all(
                    (not pd.isna(bl_series.iloc[-(k + 1)])) and
                    float(df['Close'].iloc[-(k + 1)]) > float(bl_series.iloc[-(k + 1)])
                    for k in range(scan_confirm)
                )
                if not confirmed:
                    continue

            breakout_strength = (px - float(bl)) / float(atr_v)

            if scan_max_ext < 1e9 and breakout_strength > scan_max_ext:
                continue

            if scan_rsi_p > 0 and scan_rsi_max < 100:
                rsi_v = float(rsi(df['Close'], scan_rsi_p).iloc[-1])
                if pd.isna(rsi_v) or rsi_v > scan_rsi_max:
                    continue

            vol_sma50 = df['Volume'].rolling(50).mean().iloc[-1]
            vol_today = df['Volume'].iloc[-1]

            if scan_min_bvm > 0:
                if (pd.isna(vol_sma50) or float(vol_sma50) <= 0
                        or pd.isna(vol_today)
                        or float(vol_today) < scan_min_bvm * float(vol_sma50)):
                    continue

            vol_ratio = (float(vol_today) / float(vol_sma50)
                         if (not pd.isna(vol_sma50) and float(vol_sma50) > 0
                             and not pd.isna(vol_today)) else 1.0)

            risk_per_share = float(cfg['atr_stop_mult'] * float(atr_v))
            stop_price = float(px - risk_per_share)
            tp_price = float(px + float(cfg.get('take_profit_R', 2.0)) * risk_per_share)
            shares_for_1000eur = int(max(0, (1000.0 * float(cfg['risk_per_trade'])) //
                                         max(1e-9, risk_per_share)))
            rows.append({
                'symbol': sym, 'side': 'LONG', 'price': px,
                'breakout_level': float(bl), 'asof': str(df.index[-1].date()),
                'atr': float(atr_v), 'risk_per_share': risk_per_share,
                'stop_price': stop_price, 'tp_price': tp_price,
                'shares_for_1000eur': shares_for_1000eur,
                'vol_ratio': round(vol_ratio, 2),
                'breakout_strength': round(breakout_strength, 2),
            })

        ui_status.caption('Daily Long scan: done')
        regime_label = '\U0001f7e2 Risk-On' if risk_on else '\U0001f534 Risk-Off'
        st.markdown(f'<div class="section-header">\U0001f4c8 Daily Long Signale ({regime_label})</div>',
                    unsafe_allow_html=True)
        df_sig = pd.DataFrame(rows)
        if not df_sig.empty:
            df_sig['Entry-N\u00e4he \u2605'] = (10.0 / (1.0 + df_sig['breakout_strength'])).round(1)
            df_sig['Follow-Through \u2605'] = (
                df_sig['breakout_strength'] * 2.0 * df_sig['vol_ratio'].pow(0.5)
            ).clip(upper=10.0).round(1)
            df_sig = df_sig.drop(columns=['breakout_strength']).sort_values(
                'Follow-Through \u2605', ascending=False)

            # Render signal summary metrics
            render_metric_cards([
                {'label': 'Signale gefunden', 'value': str(len(df_sig)), 'icon': '\U0001f4e1', 'css_class': 'neutral'},
                {'label': 'Regime', 'value': 'Risk-On' if risk_on else 'Risk-Off',
                 'icon': '\U0001f7e2' if risk_on else '\U0001f534',
                 'css_class': 'positive' if risk_on else 'negative'},
                {'label': 'Beste Entry-N\u00e4he', 'value': f"{df_sig['Entry-N\u00e4he \u2605'].max():.1f} \u2605",
                 'icon': '\u2b50', 'css_class': 'positive'},
                {'label': 'Bester Follow-Through', 'value': f"{df_sig['Follow-Through \u2605'].max():.1f} \u2605",
                 'icon': '\U0001f680', 'css_class': 'positive'},
            ])

            def _color_long(val):
                ratio = min(1.0, float(val) / 10.0)
                r = int(255 * (1.0 - ratio))
                g = int(200 * ratio)
                return f'background-color: rgba({r},{g},80,0.35)'

            styled = df_sig.style.map(_color_long,
                                      subset=['Entry-N\u00e4he \u2605', 'Follow-Through \u2605'])
            long_col_cfg = {
                'symbol': st.column_config.TextColumn('Symbol'),
                'side': st.column_config.TextColumn('Richtung'),
                'price': st.column_config.NumberColumn('Kurs', format='%.2f'),
                'breakout_level': st.column_config.NumberColumn('Ausbruchsniveau', format='%.2f'),
                'asof': st.column_config.TextColumn('Datum'),
                'atr': st.column_config.NumberColumn('ATR', format='%.2f'),
                'risk_per_share': st.column_config.NumberColumn('Risiko/Aktie', format='%.2f'),
                'stop_price': st.column_config.NumberColumn('Stop-Loss', format='%.2f'),
                'tp_price': st.column_config.NumberColumn('Take-Profit', format='%.2f'),
                'shares_for_1000eur': st.column_config.NumberColumn('St\u00fcck/1000\u20ac'),
                'vol_ratio': st.column_config.NumberColumn('Vol-Ratio', format='%.2f'),
                'Entry-N\u00e4he \u2605': st.column_config.NumberColumn('Entry-N\u00e4he \u2605', format='%.1f'),
                'Follow-Through \u2605': st.column_config.NumberColumn('Follow-Through \u2605', format='%.1f'),
            }
            st.dataframe(styled, column_config=long_col_cfg, use_container_width=True)

            # Expandable charts per signal
            st.markdown('<div class="section-header">Signal-Charts</div>', unsafe_allow_html=True)
            for _, row in df_sig.head(10).iterrows():
                sym = row['symbol']
                if sym in daily:
                    with st.expander(f"\U0001f4c8 {sym} \u2014 Kurs: {row['price']:.2f} | "
                                     f"Entry-N\u00e4he: {row['Entry-N\u00e4he \u2605']:.1f}\u2605 | "
                                     f"Follow-Through: {row['Follow-Through \u2605']:.1f}\u2605"):
                        fig = build_signal_chart(daily[sym], sym, row.to_dict(), side='LONG')
                        st.plotly_chart(fig, use_container_width=True)

            st.download_button('\u2b07 Download long_signals.csv',
                               data=df_sig.to_csv(index=False).encode('utf-8'),
                               file_name='long_signals.csv', mime='text/csv')
        else:
            st.info('Keine Long-Signale gefunden.')

    # -----------------------------------------------------------------------
    # DAILY SHORT SIGNALSCAN
    # -----------------------------------------------------------------------
    elif action == 'Daily Short Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m),
                           force=force_refresh)

        reg = daily[cfg['regime_symbol']].copy()
        reg['Date'] = pd.to_datetime(reg['Date'])
        reg = reg.sort_values('Date').set_index('Date')
        risk_on = bool(reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])
        regime_label = '\U0001f7e2 Risk-On' if risk_on else '\U0001f534 Risk-Off'

        ui_status.caption('Short-Scan l\u00e4uft...')
        df_short = run_short_scan(daily, cfg)
        ui_status.caption('Daily Short scan: done')

        st.markdown(f'<div class="section-header">\U0001f4c9 Daily Short Signale '
                    f'(Regime: {regime_label} \u2013 Short-Scan unabh\u00e4ngig vom Regime)</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="glass-panel">'
            '\U0001f6d1 <strong>Short-Kandidaten</strong>: \u00dcberhitzte Aktien mit '
            f'RSI > {int(cfg.get("short_rsi_min", 75))}, '
            f'EMA20-Abstand > {int(float(cfg.get("short_ema20_dist_min", 0.12)) * 100)}\u00a0%, '
            f'5d-Perf > {int(float(cfg.get("short_5d_perf_min", 0.10)) * 100)}\u00a0%, '
            f'Vol-Ratio > {cfg.get("short_vol_mult_min", 1.0)}x. '
            'Sortiert nach <strong>\u00dcberhitzungs-Score</strong> (h\u00f6her = hei\u00dfer). '
            'TP-Zonen: EMA20, letztes Breakout-Level, Fib 38\u00a0%.'
            '</div>',
            unsafe_allow_html=True,
        )

        if not df_short.empty:
            # Summary metric cards
            render_metric_cards([
                {'label': 'Short-Signale', 'value': str(len(df_short)), 'icon': '\U0001f525', 'css_class': 'negative'},
                {'label': 'Regime', 'value': 'Risk-On' if risk_on else 'Risk-Off',
                 'icon': '\U0001f7e2' if risk_on else '\U0001f534',
                 'css_class': 'positive' if risk_on else 'negative'},
                {'label': 'Max \u00dcberhitzung', 'value': f"{df_short['ueberhitzung_score'].max():.1f} \u2605",
                 'icon': '\U0001f321\ufe0f', 'css_class': 'negative'},
                {'label': 'Avg RSI', 'value': f"{df_short['rsi'].mean():.1f}",
                 'icon': '\U0001f4ca', 'css_class': 'neutral'},
            ])

            def _color_short(val):
                ratio = min(1.0, float(val) / 10.0)
                r = int(200 * ratio)
                g = int(60 * (1.0 - ratio))
                return f'background-color: rgba({r},{g},60,0.40)'

            styled_short = df_short.style.map(_color_short, subset=['ueberhitzung_score'])

            short_col_cfg = {
                'symbol': st.column_config.TextColumn('Symbol', help='Ticker'),
                'side': st.column_config.TextColumn('Richtung', help='SHORT'),
                'price': st.column_config.NumberColumn('Kurs', format='%.2f'),
                'asof': st.column_config.TextColumn('Datum'),
                'rsi': st.column_config.NumberColumn('RSI(14)', format='%.1f',
                    help='RSI(14) \u2013 \u00fcber 75 = \u00fcberkauft'),
                'ema20_dist_%': st.column_config.NumberColumn('EMA20-Abstand %', format='%.1f',
                    help='(Close/EMA20 - 1) in %'),
                '5d_perf_%': st.column_config.NumberColumn('5d-Perf %', format='%.1f',
                    help='5-Tage-Performance in %'),
                'vol_ratio': st.column_config.NumberColumn('Vol-Ratio', format='%.2f',
                    help='Volumen / VolSMA50'),
                'atr': st.column_config.NumberColumn('ATR', format='%.2f'),
                'short_stop': st.column_config.NumberColumn('Short Stop-Loss', format='%.2f',
                    help='Stop-Loss \u00fcber letztem 20-Bar-Hoch + ATR'),
                'tp_ema20': st.column_config.NumberColumn('TP: EMA20', format='%.2f',
                    help='Take-Profit Zone 1: 20-Tage EMA'),
                'tp_breakout_level': st.column_config.NumberColumn('TP: Breakout-Level', format='%.2f',
                    help='Take-Profit Zone 2: letztes 55-Tage-Hoch (Breakout-Level)'),
                'tp_fib38': st.column_config.NumberColumn('TP: Fib 38%', format='%.2f',
                    help='Take-Profit Zone 3: 38.2\u00a0% Fibonacci-Retracement des 20-Bar-Swing'),
                'ueberhitzung_score': st.column_config.NumberColumn('\U0001f525 \u00dcberhitzung \u2605',
                    format='%.1f',
                    help='Score 0\u201310: je h\u00f6her, desto \u00fcberhitzter \u2013 besser f\u00fcr Short-Einstieg'),
            }
            st.dataframe(styled_short, column_config=short_col_cfg, use_container_width=True)

            # Expandable signal charts
            st.markdown('<div class="section-header">Signal-Charts</div>', unsafe_allow_html=True)
            for _, row in df_short.head(10).iterrows():
                sym = row['symbol']
                if sym in daily:
                    with st.expander(f"\U0001f525 {sym} \u2014 Kurs: {row['price']:.2f} | "
                                     f"RSI: {row['rsi']:.1f} | "
                                     f"\u00dcberhitzung: {row['ueberhitzung_score']:.1f}\u2605"):
                        fig = build_signal_chart(daily[sym], sym, row.to_dict(), side='SHORT')
                        st.plotly_chart(fig, use_container_width=True)

                        # Inline metrics as badges
                        st.markdown(f'''
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
                            <span class="signal-badge badge-short">RSI: {row["rsi"]:.1f}</span>
                            <span class="signal-badge badge-short">EMA20-Dist: {row["ema20_dist_%"]:.1f}%</span>
                            <span class="signal-badge badge-short">5d-Perf: {row["5d_perf_%"]:.1f}%</span>
                            <span class="signal-badge badge-neutral">Vol: {row["vol_ratio"]:.2f}x</span>
                            <span class="signal-badge badge-neutral">Stop: {row["short_stop"]:.2f}</span>
                            <span class="signal-badge badge-long">TP EMA20: {row["tp_ema20"]:.2f}</span>
                            <span class="signal-badge badge-long">TP BL: {row["tp_breakout_level"]:.2f}</span>
                            <span class="signal-badge badge-long">TP Fib38: {row["tp_fib38"]:.2f}</span>
                        </div>
                        ''', unsafe_allow_html=True)

            st.download_button('\u2b07 Download short_signals.csv',
                               data=df_short.to_csv(index=False).encode('utf-8'),
                               file_name='short_signals.csv', mime='text/csv')
        else:
            st.info('Keine Short-Signale gefunden \u2013 aktuell erf\u00fcllt keine Aktie alle Filter.')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(f'''
<div class="app-footer">
    3S \u2013 Stock Signal Scanner by AF &emsp;|&emsp;
    Letzte Aktualisierung: {_TODAY} &emsp;|&emsp;
    Daten: Yahoo Finance
</div>
''', unsafe_allow_html=True)
