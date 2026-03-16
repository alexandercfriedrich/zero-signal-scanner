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

# ---------------------------------------------------------------------------
# Page config & theme constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title='3S – Stock Signal Scanner by AF', layout='wide',
                   initial_sidebar_state='expanded')

# Color palette
GOLD = '#F59E0B'
EMERALD = '#10B981'
RUBY = '#EF4444'
SAPPHIRE = '#3B82F6'
AMETHYST = '#8B5CF6'
BG_DARK = '#0B0F14'
BG_CARD = '#0F1621'
TEXT_LIGHT = '#E6EAF2'
TEXT_DIM = '#8B95A5'

CACHE_DIR = Path.home() / '.zero_swing_cache'
CACHE_DIR.mkdir(exist_ok=True)

_TODAY = datetime.date.today().isoformat()
_NOW = datetime.datetime.now()

# ---------------------------------------------------------------------------
# CSS injection – glass-morphism, glow, animated borders, custom scrollbars
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown("""
<style>
/* ── Import professional font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --gold: #F59E0B;
    --emerald: #10B981;
    --ruby: #EF4444;
    --sapphire: #3B82F6;
    --amethyst: #8B5CF6;
    --bg-dark: #0B0F14;
    --bg-card: #0F1621;
    --bg-glass: rgba(15, 22, 33, 0.65);
    --border-glass: rgba(245, 158, 11, 0.15);
    --text-light: #E6EAF2;
    --text-dim: #8B95A5;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(160deg, #0B0F14 0%, #111827 40%, #0B0F14 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0B0F14; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #F59E0B 0%, #8B5CF6 100%);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #F59E0B; }

/* ── Glass card ── */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.12);
}

/* ── Animated gradient border ── */
@keyframes borderGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.glow-border {
    position: relative;
    border-radius: 16px;
    padding: 2px;
    background: linear-gradient(135deg, #F59E0B, #8B5CF6, #3B82F6, #10B981, #F59E0B);
    background-size: 300% 300%;
    animation: borderGlow 6s ease infinite;
}
.glow-border-inner {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1.5rem;
}

/* ── Metric card ── */
.metric-card {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.15);
}
.metric-card .metric-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.15rem;
}
.metric-card .metric-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-light);
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 0.8rem 0;
}
.section-header .icon { font-size: 1.3rem; }
.section-header .title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-light);
}
.section-header .badge {
    background: rgba(245, 158, 11, 0.15);
    color: var(--gold);
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* ── Premium header ── */
.premium-header {
    background: linear-gradient(135deg, rgba(15,22,33,0.9) 0%, rgba(15,22,33,0.7) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.premium-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #F59E0B, #FBBF24, #F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.premium-header p {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin: 0;
}

/* ── Footer ── */
.premium-footer {
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(245, 158, 11, 0.1);
    color: var(--text-dim);
    font-size: 0.72rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] { display: none; }
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(15, 22, 33, 0.5);
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: var(--text-dim);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(245, 158, 11, 0.12) !important;
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}
div[data-testid="stExpander"] {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
}

/* ── Dataframe styling ── */
div[data-testid="stDataFrame"] table {
    border-collapse: separate;
    border-spacing: 0;
}
div[data-testid="stDataFrame"] th {
    background: rgba(245, 158, 11, 0.08) !important;
    color: var(--gold) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}
div[data-testid="stDataFrame"] td {
    font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(245,158,11,0.06) !important;
}
div[data-testid="stDataFrame"] tr:hover td {
    background: rgba(245, 158, 11, 0.04) !important;
}

/* ── Comparison highlight ── */
.metric-better { background: rgba(16,185,129,0.18) !important; border-radius: 6px; }
.metric-worse  { background: rgba(239,68,68,0.12) !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


inject_css()

# ---------------------------------------------------------------------------
# Premium header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="premium-header">
    <h1>3S – Stock Signal Scanner</h1>
    <p>Daily Close-Breakout &amp; Short-Overheat Scanner &bull; Backtesting Engine &bull; by AF</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Was macht dieser Scanner?**
Der 3S Stock Signal Scanner durchsucht täglich Aktien aus verschiedenen Indizes nach technischen Signalen auf Tagesbasis.

**Long-Signale:** Daily Close-Breakout über das 55-Tage-Hoch
**Short-Signale:** Überhitzte Aktien mit RSI > 75, Abstand zur EMA20 > 12 %, 5-Tage-Performance > 10 %, hohes Volumen
""")


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
    "short_rsi_min": 80,
    "short_ema20_dist_min": 0.15,
    "short_5d_perf_min": 0.12,
    "short_vol_mult_min": 1.0,
    "short_min_position": 300,
    "short_max_position": 1000,
    "short_cooldown_days": 20,
    "short_regime_filter": "risk_off_only",
    "short_max_holding_days": 30,
    "short_atr_stop_mult": 3.0,
    "short_trail_activation_pct": 0.03,
    "short_min_rr": 1.0,
    "short_min_tp_distance": 0.05,
    "short_require_red_day": True,
}

# Presets (loaded from JSON files in app/)
PRESETS = {
    'Swing (Top-5, Risk-On only)': None,
    'Best (2011-2026)': 'config_best_2011_2026.json',
    'Best (2011-2026) – mehr Signale': 'config_best_2011_2026_mehr_signale.json',
    'Best (2011-2026) – höhere Trefferquote': 'config_best_2011_2026_hoehere_trefferquote.json',
}


def _merge_cfg(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if k == 'inverse_map' and isinstance(v, dict):
            out[k] = dict(out.get(k, {}) or {})
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_preset_cfg(preset_name: str) -> dict:
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
# Helper: Plotly dark premium template
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,22,33,0.4)',
    font=dict(family='Inter, sans-serif', color=TEXT_LIGHT, size=12),
    xaxis=dict(gridcolor='rgba(245,158,11,0.06)', zeroline=False),
    yaxis=dict(gridcolor='rgba(245,158,11,0.06)', zeroline=False),
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    hovermode='x unified',
)


def _apply_dark_layout(fig, title='', height=420):
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=GOLD)),
                      height=height)
    return fig


# ---------------------------------------------------------------------------
# Helper: Metric card HTML
# ---------------------------------------------------------------------------
def metric_card(icon, label, value, color=GOLD):
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>
    """


def section_header(icon, title, badge=''):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <span class="title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Short signal candlestick chart
# ---------------------------------------------------------------------------
def make_short_signal_chart(sym, df_raw, tp_ema20, tp_bl, tp_fib38, short_stop, entry_price):
    """Create a 3-row subplot: candlestick+EMA20+markers, RSI, volume."""
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    # Use last 60 bars
    df = df.iloc[-60:]
    if len(df) < 5:
        return None

    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    rsi_s = rsi(df['Close'], 14)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.03,
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color=EMERALD, decreasing_line_color=RUBY,
        increasing_fillcolor=EMERALD, decreasing_fillcolor=RUBY,
        name='OHLC', showlegend=False,
    ), row=1, col=1)

    # EMA20 line
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20, mode='lines',
        line=dict(color=SAPPHIRE, width=1.5, dash='dot'),
        name='EMA20',
    ), row=1, col=1)

    # TP zone lines
    for tp_val, tp_name, tp_color in [
        (tp_ema20, 'TP: EMA20', SAPPHIRE),
        (tp_bl, 'TP: Breakout', AMETHYST),
        (tp_fib38, 'TP: Fib38%', GOLD),
    ]:
        if tp_val is not None and np.isfinite(tp_val):
            fig.add_hline(y=tp_val, line_dash='dash', line_color=tp_color,
                          line_width=1, annotation_text=tp_name,
                          annotation_font_color=tp_color,
                          annotation_font_size=10, row=1, col=1)

    # Stop line
    if short_stop is not None and np.isfinite(short_stop):
        fig.add_hline(y=short_stop, line_dash='dot', line_color=RUBY,
                      line_width=1.5, annotation_text='Stop-Loss',
                      annotation_font_color=RUBY,
                      annotation_font_size=10, row=1, col=1)

    # Entry marker
    if entry_price is not None and np.isfinite(entry_price):
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[entry_price], mode='markers',
            marker=dict(symbol='triangle-down', size=14, color=RUBY,
                        line=dict(color='white', width=1)),
            name='Short Entry', showlegend=True,
        ), row=1, col=1)

    # Row 2: RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_s, mode='lines',
        line=dict(color=AMETHYST, width=1.5),
        name='RSI(14)', showlegend=True,
    ), row=2, col=1)
    fig.add_hline(y=75, line_dash='dash', line_color=RUBY, line_width=0.8, row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color=TEXT_DIM, line_width=0.5, row=2, col=1)
    fig.add_hline(y=25, line_dash='dash', line_color=EMERALD, line_width=0.8, row=2, col=1)

    # Row 3: Volume
    colors = [EMERALD if c >= o else RUBY for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], marker_color=colors,
        opacity=0.6, name='Volumen', showlegend=False,
    ), row=3, col=1)

    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_layout(
        title=dict(text=f'{sym} – Short-Signal Detail', font=dict(size=14, color=GOLD)),
        height=520,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center',
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=TEXT_DIM)),
    )
    fig.update_yaxes(title_text='Kurs', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text='Vol', row=3, col=1)
    for row in [1, 2, 3]:
        fig.update_xaxes(gridcolor='rgba(245,158,11,0.06)', row=row, col=1)
        fig.update_yaxes(gridcolor='rgba(245,158,11,0.06)', row=row, col=1)

    return fig


# ---------------------------------------------------------------------------
# Helper: Monthly returns heatmap
# ---------------------------------------------------------------------------
def make_monthly_heatmap(equity_df):
    eq = equity_df['Equity'].copy()
    monthly = eq.resample('ME').last()
    monthly_ret = monthly.pct_change().dropna()
    if monthly_ret.empty:
        return None

    tbl = pd.DataFrame({
        'year': monthly_ret.index.year,
        'month': monthly_ret.index.month,
        'ret': monthly_ret.values * 100,
    })
    pivot = tbl.pivot_table(index='year', columns='month', values='ret', aggfunc='sum')
    month_names = ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    pivot.columns = [month_names[c - 1] for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0, RUBY], [0.5, BG_DARK], [1, EMERALD]],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=10, color=TEXT_LIGHT),
        hovertemplate='%{y} %{x}: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='%', tickfont=dict(color=TEXT_DIM)),
    ))
    _apply_dark_layout(fig, 'Monatliche Rendite (%)', height=max(250, len(pivot) * 35 + 100))
    return fig


# ---------------------------------------------------------------------------
# Helper: Trade distribution donut
# ---------------------------------------------------------------------------
def make_trade_distribution(trades_df):
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return None
    wins = int((trades_df['pnl'] > 0).sum())
    losses = int((trades_df['pnl'] < 0).sum())
    breakeven = int((trades_df['pnl'] == 0).sum())

    labels = ['Gewinner', 'Verlierer', 'Breakeven']
    values = [wins, losses, breakeven]
    colors = [EMERALD, RUBY, TEXT_DIM]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.55, marker=dict(colors=colors, line=dict(color=BG_DARK, width=2)),
        textinfo='label+percent', textfont=dict(size=11, color=TEXT_LIGHT),
        hovertemplate='%{label}: %{value} Trades (%{percent})<extra></extra>',
    )])
    _apply_dark_layout(fig, 'Trade-Verteilung', height=340)
    return fig


# ---------------------------------------------------------------------------
# Feature 1: Korrelationsmatrix-Heatmap
# ---------------------------------------------------------------------------
def make_correlation_heatmap(daily: dict, signal_symbols: list[str]):
    """Pairwise Pearson correlation of 60-day daily returns for signal stocks."""
    if len(signal_symbols) < 2:
        return None
    returns = {}
    for sym in signal_symbols:
        if sym not in daily:
            continue
        df = daily[sym].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        close = df['Close'].dropna()
        if len(close) >= 60:
            returns[sym] = close.iloc[-60:].pct_change().dropna()
    if len(returns) < 2:
        return None
    ret_df = pd.DataFrame(returns).dropna()
    if len(ret_df) < 10:
        return None
    corr = ret_df.corr()
    # Custom color scale: Emerald(low) -> Gold(mid) -> Ruby(high)
    colorscale = [
        [0.0, EMERALD], [0.4, EMERALD],
        [0.4, GOLD], [0.7, GOLD],
        [0.7, RUBY], [1.0, RUBY],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=colorscale,
        zmin=0, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, color=TEXT_LIGHT),
        hovertemplate='%{x} / %{y}: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Korr.', tickfont=dict(color=TEXT_DIM)),
    ))
    _apply_dark_layout(fig, '', height=max(300, len(corr) * 40 + 100))
    return fig


# ---------------------------------------------------------------------------
# Feature 2: Risk/Reward Gauge
# ---------------------------------------------------------------------------
def make_rr_gauge(rr_ratio: float):
    """Compact Risk/Reward gauge indicator."""
    display_val = min(5.0, max(0.0, rr_ratio))
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=display_val,
        number=dict(suffix='R', font=dict(size=20, color=TEXT_LIGHT)),
        gauge=dict(
            axis=dict(range=[0, 5], tickfont=dict(color=TEXT_DIM, size=9)),
            bar=dict(color=GOLD),
            bgcolor='rgba(15,22,33,0.4)',
            borderwidth=0,
            steps=[
                dict(range=[0, 1], color='rgba(239,68,68,0.3)'),
                dict(range=[1, 2], color='rgba(245,158,11,0.3)'),
                dict(range=[2, 3], color='rgba(16,185,129,0.2)'),
                dict(range=[3, 5], color='rgba(16,185,129,0.35)'),
            ],
        ),
        title=dict(text='Risk/Reward', font=dict(size=12, color=TEXT_DIM)),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=TEXT_LIGHT),
        height=160, width=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature 3: Sector Exposure Sunburst
# ---------------------------------------------------------------------------
def make_sector_sunburst(signal_symbols: list[str], daily: dict):
    """Sunburst chart of sector exposure for signal stocks."""
    SECTOR_COLORS = [GOLD, EMERALD, SAPPHIRE, AMETHYST, RUBY,
                     '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1']
    sector_map = {}
    for sym in signal_symbols:
        if sym not in daily:
            continue
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            sector_map[sym] = info.get('sector', 'Unbekannt')
        except Exception:
            sector_map[sym] = 'Unbekannt'

    if not sector_map:
        return None

    from collections import Counter
    sector_counts = Counter(sector_map.values())
    total = sum(sector_counts.values())

    ids, labels, parents, values, colors = [], [], [], [], []
    for i, (sector, count) in enumerate(sector_counts.most_common()):
        ids.append(sector)
        labels.append(f"{sector}\n({count})")
        parents.append('')
        values.append(count)
        colors.append(SECTOR_COLORS[i % len(SECTOR_COLORS)])
        for sym, sec in sector_map.items():
            if sec == sector:
                ids.append(f"{sector}-{sym}")
                labels.append(sym)
                parents.append(sector)
                values.append(1)
                colors.append(SECTOR_COLORS[i % len(SECTOR_COLORS)])

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color=BG_DARK, width=1)),
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>Anzahl: %{value}<extra></extra>',
        textfont=dict(size=11, color=TEXT_LIGHT),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color=TEXT_LIGHT),
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature 4: Regime-Timeline
# ---------------------------------------------------------------------------
def make_regime_timeline(daily: dict, cfg: dict, trades_df=None):
    """SPY regime timeline with trade markers."""
    regime_sym = cfg.get('regime_symbol', 'SPY')
    if regime_sym not in daily:
        return None
    df = daily[regime_sym].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    sma_n = int(cfg.get('sma_regime', 200))
    df['SMA200'] = df['Close'].rolling(sma_n, min_periods=sma_n).mean()
    df = df.dropna(subset=['SMA200'])
    if df.empty:
        return None

    fig = go.Figure()
    # Risk-On/Off background zones
    risk_on = df['Close'] > df['SMA200']
    zone_start = df.index[0]
    current_state = bool(risk_on.iloc[0])
    for i in range(1, len(df)):
        new_state = bool(risk_on.iloc[i])
        if new_state != current_state or i == len(df) - 1:
            zone_end = df.index[i]
            color = 'rgba(16,185,129,0.08)' if current_state else 'rgba(239,68,68,0.08)'
            fig.add_vrect(x0=zone_start, x1=zone_end, fillcolor=color,
                          layer='below', line_width=0)
            zone_start = zone_end
            current_state = new_state

    # SPY price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], mode='lines',
        line=dict(color=GOLD, width=1.5), name=regime_sym,
    ))
    # SMA200 line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA200'], mode='lines',
        line=dict(color=TEXT_DIM, width=1, dash='dash'), name=f'SMA{sma_n}',
    ))
    # Trade markers
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            entry_d = pd.Timestamp(t['entry_date'])
            exit_d = pd.Timestamp(t['exit_date'])
            side = t.get('side', 'LONG')
            # Entry marker
            if entry_d in df.index:
                y_val = float(df.loc[entry_d, 'Close'])
            else:
                y_val = float(t['entry_px'])
            fig.add_trace(go.Scatter(
                x=[entry_d], y=[y_val], mode='markers',
                marker=dict(
                    symbol='triangle-up' if side == 'LONG' else 'triangle-down',
                    size=8, color=EMERALD if side == 'LONG' else RUBY,
                    line=dict(color='white', width=0.5),
                ),
                name=f'{side} Entry', showlegend=False,
                hovertemplate=f'{t["symbol"]} {side} Entry<br>%{{x}}<extra></extra>',
            ))
            # Exit marker
            if exit_d in df.index:
                y_val_exit = float(df.loc[exit_d, 'Close'])
            else:
                y_val_exit = float(t['exit_px'])
            fig.add_trace(go.Scatter(
                x=[exit_d], y=[y_val_exit], mode='markers',
                marker=dict(symbol='x', size=6, color=TEXT_DIM,
                            line=dict(color=TEXT_DIM, width=1)),
                name='Exit', showlegend=False,
                hovertemplate=f'{t["symbol"]} Exit ({t.get("reason", "")})<br>%{{x}}<extra></extra>',
            ))

    _apply_dark_layout(fig, '', height=300)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center',
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=TEXT_DIM)),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature 5: Performance-Attribution
# ---------------------------------------------------------------------------
def make_performance_attribution(trades_df, daily: dict, cfg: dict):
    """Stacked bar charts: PnL by setup/regime, win rate, avg R."""
    if trades_df is None or trades_df.empty:
        return None, None, None

    regime_sym = cfg.get('regime_symbol', 'SPY')
    sma_n = int(cfg.get('sma_regime', 200))

    # Determine regime at entry for each trade
    reg_df = None
    if regime_sym in daily:
        reg_df = daily[regime_sym].copy()
        reg_df['Date'] = pd.to_datetime(reg_df['Date'])
        reg_df = reg_df.sort_values('Date').set_index('Date')
        reg_df['SMA200'] = reg_df['Close'].rolling(sma_n, min_periods=sma_n).mean()

    tdf = trades_df.copy()
    regimes = []
    for _, row in tdf.iterrows():
        entry_d = pd.Timestamp(row['entry_date'])
        if reg_df is not None and entry_d in reg_df.index:
            sma_val = reg_df.loc[entry_d, 'SMA200']
            close_val = reg_df.loc[entry_d, 'Close']
            if not pd.isna(sma_val) and not pd.isna(close_val):
                regimes.append('Risk-On' if close_val > sma_val else 'Risk-Off')
            else:
                regimes.append('Unbekannt')
        else:
            regimes.append('Unbekannt')
    tdf['regime'] = regimes

    setups = sorted(tdf['setup'].dropna().unique())
    regime_vals = ['Risk-On', 'Risk-Off']

    # Chart 1: Total PnL by setup (stacked by regime)
    fig1 = go.Figure()
    for regime, color in [('Risk-On', EMERALD), ('Risk-Off', RUBY)]:
        pnl_vals = []
        for setup in setups:
            mask = (tdf['setup'] == setup) & (tdf['regime'] == regime)
            pnl_vals.append(float(tdf.loc[mask, 'pnl'].sum()))
        fig1.add_trace(go.Bar(
            name=regime, x=setups, y=pnl_vals,
            marker_color=color, opacity=0.85,
        ))
    fig1.update_layout(barmode='stack')
    _apply_dark_layout(fig1, 'Gesamt-PnL nach Setup', height=350)

    # Chart 2: Win rate by setup (grouped bars)
    fig2 = go.Figure()
    for regime, color in [('Risk-On', EMERALD), ('Risk-Off', RUBY)]:
        wr_vals = []
        for setup in setups:
            mask = (tdf['setup'] == setup) & (tdf['regime'] == regime)
            sub = tdf.loc[mask]
            if len(sub) > 0:
                wr_vals.append(float((sub['pnl'] > 0).sum() / len(sub)) * 100)
            else:
                wr_vals.append(0)
        fig2.add_trace(go.Bar(
            name=regime, x=setups, y=wr_vals,
            marker_color=color, opacity=0.85,
        ))
    fig2.update_layout(barmode='group')
    _apply_dark_layout(fig2, 'Win Rate nach Setup (%)', height=350)
    fig2.update_yaxes(ticksuffix='%')

    # Chart 3: Average R-multiple by setup
    fig3 = None
    if 'R_multiple' in tdf.columns:
        fig3 = go.Figure()
        avg_r = []
        for setup in setups:
            mask = tdf['setup'] == setup
            val = tdf.loc[mask, 'R_multiple'].mean()
            avg_r.append(float(val) if np.isfinite(val) else 0)
        colors_r = [EMERALD if v >= 0 else RUBY for v in avg_r]
        fig3.add_trace(go.Bar(
            x=setups, y=avg_r,
            marker_color=colors_r, opacity=0.85,
            name='Ø R-Vielfaches',
        ))
        _apply_dark_layout(fig3, 'Ø R-Vielfaches nach Setup', height=350)

    return fig1, fig2, fig3


# ---------------------------------------------------------------------------
# Feature 7: Monte-Carlo Simulation
# ---------------------------------------------------------------------------
def run_monte_carlo(trades_df, initial_cash: float, n_sims: int = 1000):
    """Run Monte-Carlo permutation of trade sequence."""
    if trades_df is None or trades_df.empty or len(trades_df) < 3:
        return None
    pnls = trades_df['pnl'].values.copy()
    n_trades = len(pnls)
    curves = np.zeros((n_sims, n_trades + 1))
    curves[:, 0] = initial_cash
    rng = np.random.default_rng(42)
    for i in range(n_sims):
        shuffled = rng.permutation(pnls)
        curves[i, 1:] = initial_cash + np.cumsum(shuffled)

    # Stats
    final_eq = curves[:, -1]
    n_days_approx = 252  # approximate for CAGR
    cagrs = (final_eq / initial_cash) ** (252 / max(1, n_trades)) - 1

    # Max drawdown per simulation
    max_dds = np.zeros(n_sims)
    for i in range(n_sims):
        eq = curves[i]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dds[i] = dd.min()

    stats = {
        'median_cagr': float(np.median(cagrs)),
        'p5_cagr': float(np.percentile(cagrs, 5)),
        'p95_cagr': float(np.percentile(cagrs, 95)),
        'median_maxdd': float(np.median(max_dds)),
        'p5_maxdd': float(np.percentile(max_dds, 5)),
        'p95_maxdd': float(np.percentile(max_dds, 95)),
        'median_final': float(np.median(final_eq)),
        'p5_final': float(np.percentile(final_eq, 5)),
        'p95_final': float(np.percentile(final_eq, 95)),
        'ruin_prob': float((max_dds < -0.5).sum() / n_sims * 100),
    }

    # Build figure
    fig = go.Figure()
    x = list(range(n_trades + 1))

    # All curves as thin transparent lines (sample 200 for performance)
    sample_idx = rng.choice(n_sims, size=min(200, n_sims), replace=False)
    for idx in sample_idx:
        fig.add_trace(go.Scatter(
            x=x, y=curves[idx], mode='lines',
            line=dict(color=GOLD, width=0.3), opacity=0.08,
            showlegend=False, hoverinfo='skip',
        ))

    # Percentile bands
    p5 = np.percentile(curves, 5, axis=0)
    p95 = np.percentile(curves, 95, axis=0)
    median = np.median(curves, axis=0)

    fig.add_trace(go.Scatter(
        x=x, y=p95, mode='lines',
        line=dict(color=EMERALD, width=1, dash='dash'),
        name='95. Perzentil', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5, mode='lines',
        line=dict(color=RUBY, width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(245,158,11,0.06)',
        name='5. Perzentil', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=median, mode='lines',
        line=dict(color=GOLD, width=2.5),
        name='Median', showlegend=True,
    ))

    _apply_dark_layout(fig, '', height=420)
    fig.update_layout(
        xaxis_title='Trade #',
        yaxis_title='Equity (€)',
        showlegend=True,
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center',
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=TEXT_DIM)),
    )
    return fig, stats


# ---------------------------------------------------------------------------
# Feature 6: Watchlist persistence
# ---------------------------------------------------------------------------
WATCHLIST_PATH = Path.home() / '.zero_swing_watchlist.json'


def load_watchlist() -> list[dict]:
    if WATCHLIST_PATH.exists():
        try:
            return json.loads(WATCHLIST_PATH.read_text())
        except Exception:
            return []
    return []


def save_watchlist(wl: list[dict]):
    WATCHLIST_PATH.write_text(json.dumps(wl, indent=2, default=str))


def add_to_watchlist(entry: dict):
    wl = load_watchlist()
    # Avoid duplicates by symbol+side
    for existing in wl:
        if existing['symbol'] == entry['symbol'] and existing['side'] == entry['side']:
            return
    wl.append(entry)
    save_watchlist(wl)


def remove_from_watchlist(symbol: str, side: str):
    wl = load_watchlist()
    wl = [e for e in wl if not (e['symbol'] == symbol and e['side'] == side)]
    save_watchlist(wl)


def send_telegram_alert(token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram notification."""
    try:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        resp = requests.post(url, json={'chat_id': chat_id, 'text': message,
                                         'parse_mode': 'HTML'}, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Smart caching: trading-day-aware freshness
# ---------------------------------------------------------------------------
def _is_cache_fresh(cache_end_ts: pd.Timestamp) -> bool:
    """Cache is fresh if it includes data through the last completed trading day."""
    now = datetime.datetime.now()
    today = now.date()
    weekday = today.weekday()  # 0=Mon .. 6=Sun

    # Determine last expected trading day
    if weekday == 0:
        # Monday: last trading day was Friday
        last_trading_day = today - datetime.timedelta(days=3)
    elif weekday == 6:
        # Sunday: last trading day was Friday
        last_trading_day = today - datetime.timedelta(days=2)
    elif weekday == 5:
        # Saturday: last trading day was Friday
        last_trading_day = today - datetime.timedelta(days=1)
    else:
        # Tue-Fri: if market is open (before 22:00 UTC ~ US close),
        # last completed day is yesterday; otherwise today
        if now.hour >= 22:
            last_trading_day = today
        else:
            last_trading_day = today - datetime.timedelta(days=1)

    return cache_end_ts.date() >= last_trading_day


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header('Universe')
    universe_mode = st.radio(
        'Quelle',
        ['S&P 500 (Wikipedia)', 'Nasdaq 100 (Wikipedia)', 'DAX 40 (Wikipedia)',
         'ATX (Wikipedia)', 'Custom'],
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
                'der Resolver versucht diese automatisch in Yahoo-Ticker umzuwandeln.'
            ),
        )

    st.header('Aktion')
    action = st.radio(
        'Modus',
        ['Daily Long Signalscan', 'Daily Short Signalscan', 'Backtest (Daily)',
         '📋 Watchlist', '⚖️ Vergleich'],
        index=0,
    )

    st.header('Resolver Präferenzen')
    prefer_regions = st.multiselect('Yahoo region Reihenfolge', ['DE', 'AT', 'US', 'GB'],
                                    default=['DE', 'AT', 'US'])
    throttle_s = st.slider('Resolver Throttle (Sek.)', 0.0, 1.0, 0.2, 0.05)

    st.header('Konfiguration')
    preset = st.selectbox('Preset', list(PRESETS.keys()), index=1)

    if 'cfg_text' not in st.session_state:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)

    prev_preset = st.session_state.get('preset_name')
    if prev_preset != preset:
        st.session_state['cfg_text'] = json.dumps(load_preset_cfg(preset), indent=2)
        st.session_state['preset_name'] = preset

    cfg_text = st.text_area('config.json (ohne symbols)', value=st.session_state['cfg_text'],
                            height=420)

    # --- Short-Parameter Sidebar (shown for Short Signalscan & Backtest) ---
    if action in ('Daily Short Signalscan', 'Backtest (Daily)'):
        with st.expander('📉 Short-Parameter', expanded=False):
            sb_short_rsi_min = st.number_input(
                'RSI Mindestfilter', min_value=50, max_value=99,
                value=int(DEFAULT_CFG.get('short_rsi_min', 80)),
                help='Nur Aktien mit RSI ≥ diesem Wert werden geshorted',
                key='sb_short_rsi_min')
            sb_short_ema20_dist = st.number_input(
                'EMA20 Distanz Mindestfilter (%)', min_value=0.0, max_value=1.0,
                value=float(DEFAULT_CFG.get('short_ema20_dist_min', 0.15)),
                step=0.01, format='%.2f',
                help='Min. Abstand Close/EMA20 - 1 (z.B. 0.15 = 15 %)',
                key='sb_short_ema20_dist')
            sb_short_5d_perf = st.number_input(
                '5-Tage Perf. Mindestfilter (%)', min_value=0.0, max_value=1.0,
                value=float(DEFAULT_CFG.get('short_5d_perf_min', 0.12)),
                step=0.01, format='%.2f',
                help='Min. 5-Tage-Performance (z.B. 0.12 = 12 %)',
                key='sb_short_5d_perf')
            sb_short_min_pos = st.number_input(
                'Mindest-Positionsgröße (€)', min_value=0, max_value=10000,
                value=int(DEFAULT_CFG.get('short_min_position', 300)),
                step=50,
                help='Minimale Positionsgröße pro Short-Trade in €',
                key='sb_short_min_pos')
            sb_short_max_pos = st.number_input(
                'Maximale Positionsgröße (€)', min_value=100, max_value=50000,
                value=int(DEFAULT_CFG.get('short_max_position', 1000)),
                step=100,
                help='Maximale Positionsgröße pro Short-Trade in €',
                key='sb_short_max_pos')
            sb_short_cooldown = st.number_input(
                'Cooldown pro Symbol (Tage)', min_value=0, max_value=120,
                value=int(DEFAULT_CFG.get('short_cooldown_days', 20)),
                step=5,
                help='Nach Exit: so viele Tage kein Neueinstieg im selben Symbol',
                key='sb_short_cooldown')
            sb_short_regime = st.selectbox(
                'Regime-Filter',
                options=['any', 'risk_off_only', 'weak_only'],
                index=['any', 'risk_off_only', 'weak_only'].index(
                    DEFAULT_CFG.get('short_regime_filter', 'risk_off_only')),
                help='any = immer shorten, risk_off_only = nur wenn SPY < SMA200, weak_only = nur wenn SPY < SMA50',
                key='sb_short_regime')
            sb_short_atr_stop = st.number_input(
                'Short ATR-Stop Multiplikator', min_value=1.5, max_value=5.0,
                value=float(DEFAULT_CFG.get('short_atr_stop_mult', 3.0)),
                step=0.5, format='%.1f',
                help='ATR-Multiplikator für Short-Stops (breiter = weniger Stopouts)',
                key='sb_short_atr_stop')
            sb_short_trail_act = st.number_input(
                'Trail-Aktivierung (% Profit)', min_value=0.01, max_value=0.10,
                value=float(DEFAULT_CFG.get('short_trail_activation_pct', 0.03)),
                step=0.01, format='%.2f',
                help='Trailing-Stop erst aktivieren, wenn Position ≥ diesen %-Gewinn hat (z.B. 0.03 = 3 %)',
                key='sb_short_trail_act')
            sb_short_min_rr = st.number_input(
                'Min. Reward:Risk Ratio', min_value=0.5, max_value=3.0,
                value=float(DEFAULT_CFG.get('short_min_rr', 1.0)),
                step=0.1, format='%.1f',
                help='Mindest-R:R — Trade wird übersprungen wenn Reward/Risk < diesem Wert',
                key='sb_short_min_rr')
            sb_short_min_tp_dist = st.number_input(
                'Min. TP-Abstand (%)', min_value=0.02, max_value=0.20,
                value=float(DEFAULT_CFG.get('short_min_tp_distance', 0.05)),
                step=0.01, format='%.2f',
                help='TPs näher als dieser %-Abstand zum Entry werden ignoriert (z.B. 0.05 = 5 %)',
                key='sb_short_min_tp_dist')
            sb_short_red_day = st.checkbox(
                'Red-Day Bestätigung',
                value=bool(DEFAULT_CFG.get('short_require_red_day', True)),
                help='Nur einsteigen wenn heutiger Close < gestriger Close (erster roter Tag)',
                key='sb_short_red_day')
            sb_short_max_hold = st.number_input(
                'Max. Haltedauer Shorts (Tage)', min_value=5, max_value=365,
                value=int(DEFAULT_CFG.get('short_max_holding_days', 30)),
                step=5,
                help='Maximale Haltedauer für Short-Positionen',
                key='sb_short_max_hold')

    with st.expander('ℹ️ Konfigurations-Parameter'):
        st.markdown("""
| Parameter | Standard | Beschreibung |
|---|---|---|
| `start` | `"2021-01-01"` | Startdatum (YYYY-MM-DD) |
| `end` | *(heute)* | Enddatum – wird automatisch auf heute gesetzt |
| `regime_symbol` | `"SPY"` | Regime/Benchmark-Symbol |
| `inverse_map` | `{ "SPY": "SH" }` | Mapping Regime→Inverse-ETF |
| `hard_risk_on` | `true` | Keine neuen Long-Trades bei Risk-Off |
| `max_new_trades_per_day` | `2` | Max. neue Einstiege pro Tag |
| `max_positions` | `5` | Max. gleichzeitige Positionen |
| `weekly_rerank` | `true` | Wöchentlich neu ranken |
| `weekly_rebalance_weekday` | `0` | Wochentag fürs Rerank (0=Mo … 4=Fr) |
| `risk_per_trade` | `0.01` | Risiko pro Trade (Anteil vom Equity) |
| `atr_period` | `14` | ATR-Periode |
| `atr_stop_mult` | `2.0` | Initialer Stop = ATR × Mult |
| `use_trailing_stop` | `true` | Trailing-Stop aktiv |
| `atr_trail_mult` | `2.5` | Trailing-Stop = ATR × Mult |
| `trailing_reference` | `"close"` | `"high"` = Chandelier, `"close"` = Schlusskurs |
| `take_profit_R` | `2.0` | Take-Profit in R (nur ohne Trailing) |
| `breakout_lookback` | `55` | Lookback für Breakout-Level |
| `breakout_level_source` | `"close"` | `"close"` oder `"high"` |
| `breakout_confirm_closes` | `1` | Anzahl Closes über Level |
| `sma_regime` | `200` | SMA-Länge für Regime-Filter |
| `max_holding_days` | `30` | Max. Haltedauer (Tage) |
| `min_breakout_vol_mult` | `0.0` | Vol ≥ N × VolSMA50 (0 = aus) |
| `rsi_period` | `0` | RSI-Periode für Long-Filter (0 = aus) |
| `rsi_max` | `100` | Max. RSI beim Long-Einstieg |
| `max_breakout_extension_atr` | `1e9` | Max. Ausdehnung über Level in ATR |
| `mom_lookback` | `126` | Momentum-Lookback (Tage) |
| `enable_cwh` | `true` | Cup-with-Handle aktivieren |
| `cwh_cup_min_bars` | `30` | CWH: min. Cup-Länge |
| `cwh_cup_max_bars` | `130` | CWH: max. Cup-Länge |
| `cwh_handle_min_bars` | `5` | CWH: min. Handle-Länge |
| `cwh_handle_max_bars` | `20` | CWH: max. Handle-Länge |
| `cwh_max_cup_depth` | `0.35` | CWH: max. Cup-Tiefe |
| `cwh_max_handle_depth` | `0.15` | CWH: max. Handle-Tiefe |
| `cwh_trend_sma` | `50` | CWH: Trendfilter SMA |
| `cwh_vol_bonus` | `0.3` | CWH: Volumen-Qualitäts-Bonus |
| `corr_lookback_days` | `60` | Lookback für Korrelationsfilter |
| `max_pair_corr` | `1.0` | Max. Korrelation (1.0 = aus) |
| `max_positions_per_sector` | `999` | Max. Positionen pro Sektor |
| `spread_bps_per_side` | `8` | Spread in Basispunkten |
| `min_price` | `2.0` | Mindestkurs |
| `min_dollar_volume` | `2000000` | Mindest-Dollar-Volume |
| `initial_cash` | `5000` | Startkapital |
| `short_rsi_min` | `80` | **Short-Scan**: Min. RSI |
| `short_ema20_dist_min` | `0.15` | **Short-Scan**: Min. Abstand zur EMA20 (z.B. 0.15 = 15 %) |
| `short_5d_perf_min` | `0.12` | **Short-Scan**: Min. 5-Tage-Performance |
| `short_vol_mult_min` | `1.0` | **Short-Scan**: Min. Vol-Ratio (Vol / VolSMA50) |
| `short_min_position` | `300` | **Short**: Mindest-Positionsgröße in € |
| `short_max_position` | `1000` | **Short**: Maximale Positionsgröße in € |
| `short_cooldown_days` | `20` | **Short**: Tage Pause pro Symbol nach Exit |
| `short_regime_filter` | `"risk_off_only"` | **Short**: Regime-Filter (`any`/`risk_off_only`/`weak_only`) |
| `short_atr_stop_mult` | `3.0` | **Short**: ATR-Multiplikator für Stops (breiter als Long) |
| `short_trail_activation_pct` | `0.03` | **Short**: Trailing erst ab diesem %-Gewinn aktivieren |
| `short_min_rr` | `1.0` | **Short**: Mindest-Reward:Risk Ratio |
| `short_min_tp_distance` | `0.05` | **Short**: Min. TP-Abstand zum Entry (5 %) |
| `short_require_red_day` | `true` | **Short**: Nur einsteigen bei Close < Vortages-Close |
| `short_max_holding_days` | `30` | **Short**: Max. Haltedauer für Shorts (Tage) |
""")

    st.divider()
    st.markdown('##### Cache-Verwaltung')
    force_refresh = st.button('🔄 Cache leeren (Force Refresh)')
    if force_refresh:
        import shutil
        parquet_files = list(CACHE_DIR.glob('*.parquet'))
        for f in parquet_files:
            f.unlink()
        st.success(f'{len(parquet_files)} Cache-Dateien gelöscht.')
        st.cache_data.clear()

    run_btn = st.button('Start', type='primary')


# ---------------------------------------------------------------------------
# Symbol loaders (unchanged logic)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_sp500_symbols() -> list[str]:
    wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        r = requests.get(wiki_url, timeout=25,
                         headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        df = tables[0]
        syms = df['Symbol'].astype(str).str.upper().tolist()
        return [s.replace('.', '-') for s in syms]
    except Exception:
        pass
    raw_csv = ('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/'
               'master/data/constituents.csv')
    r = requests.get(raw_csv, timeout=25,
                     headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
    r.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(r.text))
    syms = df['Symbol'].astype(str).str.upper().tolist()
    return [s.replace('.', '-') for s in syms]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_nasdaq100_symbols() -> list[str]:
    wiki_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    try:
        r = requests.get(wiki_url, timeout=25,
                         headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns
                           if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
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
        r = requests.get(wiki_url, timeout=25,
                         headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns
                           if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
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
        r = requests.get(wiki_url, timeout=25,
                         headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in df.columns
                           if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
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
        r = requests.get(wiki_url, timeout=25,
                         headers={'User-Agent': 'Mozilla/5.0 (zero-signal-scanner)'})
        r.raise_for_status()
        tables = pd.read_html(r.text)
        df = tables[0]
        cols_lower = {str(c).lower(): c for c in df.columns}
        sym_col = next((cols_lower[k] for k in cols_lower
                        if 'symbol' in k or 'ticker' in k), None)
        sec_col = next((cols_lower[k] for k in cols_lower
                        if 'sector' in k or 'gics' in k), None)
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
# Resolver helpers (unchanged)
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
# Data loading with smart caching
# ---------------------------------------------------------------------------
def cache_path(sym: str, kind: str) -> Path:
    safe = sym.replace('^', '_').replace('/', '_')
    return CACHE_DIR / f'{safe}.{kind}.parquet'


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


def load_daily(symbols: list[str], start: str, end: str, progress_cb=None) -> dict[str, pd.DataFrame]:
    out = {}
    need = []
    for s in symbols:
        p = cache_path(s, '1d')
        if p.exists():
            try:
                d = pd.read_parquet(p)
                d['Date'] = pd.to_datetime(d['Date'])
                cache_end = d['Date'].max()
                if (d['Date'].min() <= pd.Timestamp(start) and _is_cache_fresh(cache_end)):
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

        ema20 = close.ewm(span=20, adjust=False).mean()
        rsi_series = rsi(close, 14)
        vol_sma50 = vol.rolling(50).mean()

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(),
                         (low - prev_close).abs()], axis=1).max(axis=1)
        atr_series = tr.rolling(atr_p).mean()

        if bl_src == 'high':
            bl_series = high.shift(1).rolling(bl_lookback).max()
        else:
            bl_series = close.shift(1).rolling(bl_lookback).max()

        px = float(close.iloc[-1])
        ema20_v = float(ema20.iloc[-1])
        rsi_v = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else np.nan
        vol_today = float(vol.iloc[-1])
        vol_sma50_v = float(vol_sma50.iloc[-1]) if not pd.isna(vol_sma50.iloc[-1]) else np.nan
        atr_v = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else np.nan
        bl_v = float(bl_series.iloc[-1]) if not pd.isna(bl_series.iloc[-1]) else np.nan

        if len(close) < 6:
            continue
        perf5 = float(close.iloc[-1] / close.iloc[-6] - 1)

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

        tp_ema20 = round(ema20_v, 2)
        tp_breakout = round(bl_v, 2) if not pd.isna(bl_v) else np.nan

        recent_high = float(high.iloc[-20:].max())
        recent_low = float(low.iloc[-20:].min())
        fib38 = round(recent_high - 0.382 * (recent_high - recent_low), 2)

        short_stop = round(recent_high + atr_v, 2)

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
# Display helpers: backtest metrics as glass cards
# ---------------------------------------------------------------------------
def display_backtest_metrics(summary, prefix=''):
    """Render two rows of glass-morphism metric cards for backtest results."""
    final_eq = summary['final_equity']
    initial = summary['initial_cash']

    row1 = st.columns(5)
    cards_row1 = [
        ('💰', 'Endkapital', f"€{final_eq:,.2f}", GOLD),
        ('📈', 'CAGR', f"{summary['CAGR'] * 100:.2f}%",
         EMERALD if summary['CAGR'] >= 0 else RUBY),
        ('📉', 'Max. Drawdown', f"{summary['MaxDrawdown'] * 100:.2f}%", RUBY),
        ('📊', 'Volatilität', f"{summary['Volatility'] * 100:.2f}%", SAPPHIRE),
        ('🔢', 'Trades', str(summary['Trades']), AMETHYST),
    ]
    for col, (icon, label, value, color) in zip(row1, cards_row1):
        col.markdown(metric_card(icon, label, value, color), unsafe_allow_html=True)

    row2 = st.columns(5)
    pf = summary.get('ProfitFactor', np.nan)
    wr = summary.get('WinRate', np.nan)
    aw = summary.get('AvgWin', np.nan)
    al = summary.get('AvgLoss', np.nan)
    er = summary.get('Expectancy_R', np.nan)
    cards_row2 = [
        ('⚖️', 'Profit Factor',
         '-' if pd.isna(pf) else f"{pf:.2f}",
         EMERALD if (not pd.isna(pf) and pf >= 1) else RUBY),
        ('🎯', 'Win Rate',
         '-' if pd.isna(wr) else f"{wr * 100:.1f}%",
         EMERALD if (not pd.isna(wr) and wr >= 0.5) else RUBY),
        ('✅', 'Ø Gewinn',
         '-' if pd.isna(aw) else f"€{aw:.2f}", EMERALD),
        ('❌', 'Ø Verlust',
         '-' if pd.isna(al) else f"€{al:.2f}", RUBY),
        ('🧮', 'Expectancy (R)',
         '-' if pd.isna(er) else f"{er:.2f}",
         EMERALD if (not pd.isna(er) and er >= 0) else RUBY),
    ]
    for col, (icon, label, value, color) in zip(row2, cards_row2):
        col.markdown(metric_card(icon, label, value, color), unsafe_allow_html=True)


def display_equity_drawdown(equity_df, title_suffix='', prefix='long'):
    """Render premium equity curve and drawdown charts side by side."""
    c1, c2 = st.columns([1.4, 1])
    with c1:
        eq_reset = equity_df.reset_index()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_reset['Date'], y=eq_reset['Equity'],
            mode='lines',
            line=dict(color=GOLD, width=2),
            fill='tozeroy',
            fillcolor='rgba(245,158,11,0.08)',
            name='Equity',
        ))
        _apply_dark_layout(fig_eq, f'Equity Curve{title_suffix}', height=380)
        st.plotly_chart(fig_eq, use_container_width=True, key=f'{prefix}_equity_curve')

    with c2:
        eq = equity_df['Equity']
        dd = (eq / eq.cummax() - 1)
        dd_reset = dd.reset_index()
        dd_reset.columns = ['Date', 'Drawdown']
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_reset['Date'], y=dd_reset['Drawdown'],
            mode='lines',
            line=dict(color=RUBY, width=1.5),
            fill='tozeroy',
            fillcolor='rgba(239,68,68,0.12)',
            name='Drawdown',
        ))
        _apply_dark_layout(fig_dd, f'Drawdown{title_suffix}', height=380)
        fig_dd.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig_dd, use_container_width=True, key=f'{prefix}_drawdown')


def display_advanced_charts(equity_df, trades_df, prefix='long'):
    """Render monthly heatmap full-width, then trade distribution below."""
    fig_heat = make_monthly_heatmap(equity_df)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True, key=f'{prefix}_heatmap')
    fig_dist = make_trade_distribution(trades_df)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True, key=f'{prefix}_distribution')


def display_breakdown_tables(breakdown, prefix='long'):
    """Render breakdown tables with styled formatting."""
    breakdown_col_cfg = {
        'setup': st.column_config.TextColumn('Setup'),
        'reason': st.column_config.TextColumn('Grund'),
        'Trades': st.column_config.NumberColumn('Trades'),
        'WinRate': st.column_config.NumberColumn('Win Rate', format='%.1f%%'),
        'ProfitFactor': st.column_config.NumberColumn('Profit Factor', format='%.2f'),
        'AvgPnL': st.column_config.NumberColumn('Ø PnL', format='%.2f'),
        'AvgR': st.column_config.NumberColumn('Ø R', format='%.2f'),
    }
    b1, b2 = st.columns(2)
    with b1:
        st.caption('Nach Setup')
        st.dataframe(breakdown.get('by_setup', pd.DataFrame()),
                     column_config=breakdown_col_cfg, use_container_width=True,
                     key=f'{prefix}_breakdown_setup')
    with b2:
        st.caption('Nach Exit-Grund')
        st.dataframe(breakdown.get('by_reason', pd.DataFrame()),
                     column_config=breakdown_col_cfg, use_container_width=True,
                     key=f'{prefix}_breakdown_reason')


def display_trades_table(trades_df, prefix='long'):
    """Render styled trades table."""
    trades_col_cfg = {
        'symbol': st.column_config.TextColumn('Symbol'),
        'side': st.column_config.TextColumn('Richtung'),
        'entry_date': st.column_config.TextColumn('Einstieg'),
        'entry_px': st.column_config.NumberColumn('Einstiegskurs', format='%.2f'),
        'exit_date': st.column_config.TextColumn('Ausstieg'),
        'exit_px': st.column_config.NumberColumn('Ausstiegskurs', format='%.2f'),
        'shares': st.column_config.NumberColumn('Stück'),
        'pnl': st.column_config.NumberColumn('PnL (€)', format='%.2f'),
        'reason': st.column_config.TextColumn('Exitgrund'),
        'setup': st.column_config.TextColumn('Setup'),
        'initial_risk_per_share': st.column_config.NumberColumn('Init. Risiko/Aktie', format='%.2f'),
        'R_multiple': st.column_config.NumberColumn('R-Vielfaches', format='%.2f'),
    }
    st.dataframe(trades_df, column_config=trades_col_cfg, use_container_width=True,
                 key=f'{prefix}_trades_table')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP LOGIC
# ═══════════════════════════════════════════════════════════════════════════
if run_btn:
    cfg = json.loads(cfg_text)
    cfg['end'] = _TODAY

    # Merge sidebar short params if available (Short Signalscan or Backtest mode)
    if action in ('Daily Short Signalscan', 'Backtest (Daily)'):
        cfg['short_rsi_min'] = st.session_state.get('sb_short_rsi_min', cfg.get('short_rsi_min', 80))
        cfg['short_ema20_dist_min'] = st.session_state.get('sb_short_ema20_dist', cfg.get('short_ema20_dist_min', 0.15))
        cfg['short_5d_perf_min'] = st.session_state.get('sb_short_5d_perf', cfg.get('short_5d_perf_min', 0.12))
        cfg['short_min_position'] = st.session_state.get('sb_short_min_pos', cfg.get('short_min_position', 300))
        cfg['short_max_position'] = st.session_state.get('sb_short_max_pos', cfg.get('short_max_position', 1000))
        cfg['short_cooldown_days'] = st.session_state.get('sb_short_cooldown', cfg.get('short_cooldown_days', 20))
        cfg['short_regime_filter'] = st.session_state.get('sb_short_regime', cfg.get('short_regime_filter', 'risk_off_only'))
        cfg['short_atr_stop_mult'] = st.session_state.get('sb_short_atr_stop', cfg.get('short_atr_stop_mult', 3.0))
        cfg['short_trail_activation_pct'] = st.session_state.get('sb_short_trail_act', cfg.get('short_trail_activation_pct', 0.03))
        cfg['short_min_rr'] = st.session_state.get('sb_short_min_rr', cfg.get('short_min_rr', 1.0))
        cfg['short_min_tp_distance'] = st.session_state.get('sb_short_min_tp_dist', cfg.get('short_min_tp_distance', 0.05))
        cfg['short_require_red_day'] = st.session_state.get('sb_short_red_day', cfg.get('short_require_red_day', True))
        cfg['short_max_holding_days'] = st.session_state.get('sb_short_max_hold', cfg.get('short_max_holding_days', 30))

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

    with st.expander('Symbol‑Auflösung', expanded=False):
        st.dataframe(resolve_table, use_container_width=True, key='resolve_table')

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

    # ===================================================================
    # BACKTEST
    # ===================================================================
    if action == 'Backtest (Daily)':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m))
        cfg['symbols'] = [s for s in cfg['symbols'] if s in daily]

        # ── Regime Timeline (Feature 4) ──
        section_header('📈', 'Markt-Regime Timeline', '')
        st.caption('SPY vs SMA200 – Grün = Risk-On, Rot = Risk-Off. Dreiecke = Trades.')
        regime_fig = make_regime_timeline(daily, cfg)
        if regime_fig:
            st.plotly_chart(regime_fig, use_container_width=True, key='regime_timeline_initial')

        # ── Long Backtest ──
        section_header('📈', 'Long Backtest', 'LONG')

        ui_status.caption('Long-Backtest läuft...')
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
        bt_status.caption('Long-Backtest: done')

        display_backtest_metrics(summary)

        with st.expander('ℹ️ Kennzahlen-Erklärung'):
            st.markdown("""
| Kennzahl | Erklärung |
|---|---|
| **Endkapital** | Gesamtkapital am Ende des Backtest-Zeitraums |
| **CAGR** | Jährliche Wachstumsrate |
| **Max. Drawdown** | Größter prozentualer Rückgang |
| **Volatilität** | Annualisierte Standardabweichung der täglichen Renditen |
| **Trades** | Gesamtanzahl abgeschlossener Trades |
| **Profit Factor** | Gesamtgewinn / Gesamtverlust (>1 = profitabel) |
| **Win Rate** | Anteil gewinnbringender Trades |
| **Ø Gewinn** | Durchschnittlicher Gewinn je Trade (€) |
| **Ø Verlust** | Durchschnittlicher Verlust je Trade (€) |
| **Expectancy (R)** | Erwartungswert je Trade in R; >0 = positiv |
""")

        display_equity_drawdown(equity_df, ' (Long)', prefix='long')
        display_advanced_charts(equity_df, trades_df, prefix='long')

        st.subheader('Downloads (Long)')
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button('⬇ Download trades_long.csv',
                               data=trades_df.to_csv(index=False).encode('utf-8'),
                               file_name='trades_long.csv', mime='text/csv',
                               key='dl_trades_long')
        with dl2:
            st.download_button('⬇ Download equity_long.csv',
                               data=equity_df.reset_index().to_csv(index=False).encode('utf-8'),
                               file_name='equity_long.csv', mime='text/csv',
                               key='dl_equity_long')

        with st.expander('Breakdown (Long)', expanded=False):
            display_breakdown_tables(breakdown, prefix='long')

        with st.expander('Alle Trades (Long)', expanded=False):
            display_trades_table(trades_df, prefix='long')

        # ── Short Backtest ──
        st.divider()
        section_header('📉', 'Short Backtest', 'SHORT')

        sbt_prog = st.progress(0.0)
        sbt_status = st.empty()

        def sbt_step(done, total, date_or_msg):
            frac = 0.0 if total <= 0 else float(done) / float(total)
            sbt_prog.progress(min(1.0, max(0.0, frac)))
            if isinstance(date_or_msg, str):
                sbt_status.caption(date_or_msg)
            else:
                sbt_status.caption(
                    f'Short-Backtest: {done}/{total}  ({date_or_msg.date().isoformat()})')

        s_equity_df, s_trades_df, s_summary, s_breakdown = run_short_backtest(
            daily, cfg, progress_cb=sbt_step)
        sbt_status.caption('Short-Backtest: done')

        display_backtest_metrics(s_summary)
        display_equity_drawdown(s_equity_df, ' (Short)', prefix='short')
        display_advanced_charts(s_equity_df, s_trades_df, prefix='short')

        st.subheader('Downloads (Short)')
        dl3, dl4 = st.columns(2)
        with dl3:
            st.download_button('⬇ Download trades_short.csv',
                               data=s_trades_df.to_csv(index=False).encode('utf-8'),
                               file_name='trades_short.csv', mime='text/csv',
                               key='dl_trades_short')
        with dl4:
            st.download_button('⬇ Download equity_short.csv',
                               data=s_equity_df.reset_index().to_csv(index=False).encode('utf-8'),
                               file_name='equity_short.csv', mime='text/csv',
                               key='dl_equity_short')

        with st.expander('Breakdown (Short)', expanded=False):
            display_breakdown_tables(s_breakdown, prefix='short')

        with st.expander('Alle Trades (Short)', expanded=False):
            display_trades_table(s_trades_df, prefix='short')

        # ── Update Regime Timeline with all trades ──
        all_trades = pd.concat([trades_df, s_trades_df], ignore_index=True)
        if not all_trades.empty:
            if 'side' not in trades_df.columns:
                trades_df = trades_df.copy()
                trades_df['side'] = 'LONG'
            if 'side' not in s_trades_df.columns:
                s_trades_df_copy = s_trades_df.copy()
                s_trades_df_copy['side'] = 'SHORT'
                all_trades = pd.concat([trades_df, s_trades_df_copy], ignore_index=True)
            regime_fig_full = make_regime_timeline(daily, cfg, all_trades)
            if regime_fig_full:
                # Re-render with trade markers
                st.divider()
                section_header('📈', 'Markt-Regime Timeline (mit Trades)', '')
                st.plotly_chart(regime_fig_full, use_container_width=True, key='regime_timeline_full')

        # ── Performance Attribution (Feature 5) ──
        st.divider()
        section_header('🎯', 'Performance-Attribution', '')
        all_bt_trades = pd.concat([trades_df, s_trades_df], ignore_index=True)
        if not all_bt_trades.empty:
            fig_pnl, fig_wr, fig_avgr = make_performance_attribution(all_bt_trades, daily, cfg)
            pa_c1, pa_c2 = st.columns(2)
            with pa_c1:
                if fig_pnl:
                    st.plotly_chart(fig_pnl, use_container_width=True, key='pa_pnl')
                if fig_avgr:
                    st.plotly_chart(fig_avgr, use_container_width=True, key='pa_avgr')
            with pa_c2:
                if fig_wr:
                    st.plotly_chart(fig_wr, use_container_width=True, key='pa_wr')
        else:
            st.info('Keine Trades für Performance-Attribution vorhanden.')

        # ── Monte-Carlo Simulation (Feature 7) ──
        st.divider()
        section_header('🎲', 'Monte-Carlo Simulation (1.000 Permutationen)', '')
        if st.button('Monte-Carlo starten', key='mc_start'):
            if not all_bt_trades.empty and len(all_bt_trades) >= 3:
                with st.spinner('Monte-Carlo läuft (1.000 Permutationen)...'):
                    mc_result = run_monte_carlo(all_bt_trades, float(cfg.get('initial_cash', 5000)))
                if mc_result is not None:
                    mc_fig, mc_stats = mc_result
                    st.plotly_chart(mc_fig, use_container_width=True, key='mc_chart')
                    # Stats table
                    mc_df = pd.DataFrame({
                        'Kennzahl': ['Median CAGR', '5. Perzentil CAGR', '95. Perzentil CAGR',
                                     'Median Max DD', '5. Perzentil Max DD', '95. Perzentil Max DD',
                                     'Median Endkapital', '5. Perz. Endkapital', '95. Perz. Endkapital',
                                     'Ruin-Wahrscheinlichkeit (>50% DD)'],
                        'Wert': [
                            f"{mc_stats['median_cagr']*100:.2f}%",
                            f"{mc_stats['p5_cagr']*100:.2f}%",
                            f"{mc_stats['p95_cagr']*100:.2f}%",
                            f"{mc_stats['median_maxdd']*100:.2f}%",
                            f"{mc_stats['p5_maxdd']*100:.2f}%",
                            f"{mc_stats['p95_maxdd']*100:.2f}%",
                            f"€{mc_stats['median_final']:,.2f}",
                            f"€{mc_stats['p5_final']:,.2f}",
                            f"€{mc_stats['p95_final']:,.2f}",
                            f"{mc_stats['ruin_prob']:.1f}%",
                        ],
                    })
                    st.dataframe(mc_df, use_container_width=True, hide_index=True, key='mc_stats_df')
                else:
                    st.warning('Zu wenige Trades für Monte-Carlo-Simulation.')
            else:
                st.warning('Mindestens 3 Trades nötig für Monte-Carlo.')

    # ===================================================================
    # DAILY LONG SIGNALSCAN
    # ===================================================================
    elif action == 'Daily Long Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m))

        reg = daily[cfg['regime_symbol']].copy()
        reg['Date'] = pd.to_datetime(reg['Date'])
        reg = reg.sort_values('Date').set_index('Date')
        risk_on = bool(
            reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])

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
        regime_label = '🟢 Risk-On' if risk_on else '🔴 Risk-Off'
        section_header('📈', f'Daily Long Signale ({regime_label})', f'{len(rows)} Signale')

        df_sig = pd.DataFrame(rows)
        if not df_sig.empty:
            df_sig['Entry-Nähe ★'] = (10.0 / (1.0 + df_sig['breakout_strength'])).round(1)
            df_sig['Follow-Through ★'] = (
                df_sig['breakout_strength'] * 2.0 * df_sig['vol_ratio'].pow(0.5)
            ).clip(upper=10.0).round(1)
            df_sig = df_sig.drop(columns=['breakout_strength']).sort_values(
                'Follow-Through ★', ascending=False)

            def _color_long(val):
                ratio = min(1.0, float(val) / 10.0)
                r = int(255 * (1.0 - ratio))
                g = int(200 * ratio)
                return f'background-color: rgba({r},{g},80,0.35)'

            styled = df_sig.style.map(_color_long,
                                      subset=['Entry-Nähe ★', 'Follow-Through ★'])
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
                'shares_for_1000eur': st.column_config.NumberColumn('Stück/1000€'),
                'vol_ratio': st.column_config.NumberColumn('Vol-Ratio', format='%.2f'),
                'Entry-Nähe ★': st.column_config.NumberColumn('Entry-Nähe ★', format='%.1f'),
                'Follow-Through ★': st.column_config.NumberColumn('Follow-Through ★',
                                                                   format='%.1f'),
            }
            st.dataframe(styled, column_config=long_col_cfg, use_container_width=True, key='long_signals_df')
            st.download_button('⬇ Download long_signals.csv',
                               data=df_sig.to_csv(index=False).encode('utf-8'),
                               file_name='long_signals.csv', mime='text/csv', key='dl_long_signals')

            # ── Signal Detail with R/R Gauge + Watchlist (Features 2 & 6) ──
            signal_syms = df_sig['symbol'].tolist()
            section_header('📊', 'Signal-Detail', 'Interaktiv')
            for _, row in df_sig.iterrows():
                sym = row['symbol']
                if sym not in daily:
                    continue
                with st.expander(f"📊 {sym} – Kurs: {row['price']:.2f} | "
                                 f"Stop: {row['stop_price']:.2f} | TP: {row['tp_price']:.2f}",
                                 expanded=False):
                    # R/R gauge
                    reward = row['tp_price'] - row['price']
                    risk = row['price'] - row['stop_price']
                    rr = reward / risk if risk > 0 else 0
                    g_col, info_col = st.columns([1, 3])
                    with g_col:
                        rr_fig = make_rr_gauge(rr)
                        st.plotly_chart(rr_fig, use_container_width=True, key=f'rr_long_{sym}')
                    with info_col:
                        st.markdown(f"""
**{sym}** – Long Breakout Signal
- **Entry:** {row['price']:.2f} | **Stop:** {row['stop_price']:.2f} | **TP:** {row['tp_price']:.2f}
- **ATR:** {row['atr']:.2f} | **Vol-Ratio:** {row['vol_ratio']:.2f}
- **R:R = {rr:.2f}**
""")
                    # Watchlist button
                    if st.button(f'➕ Watchlist', key=f'wl_long_{sym}'):
                        add_to_watchlist({
                            'symbol': sym, 'side': 'LONG',
                            'entry': row['price'], 'stop': row['stop_price'],
                            'tp': row['tp_price'], 'date_added': _TODAY,
                        })
                        st.success(f'{sym} zur Watchlist hinzugefügt!')

            # ── Korrelationsmatrix (Feature 1) ──
            if len(signal_syms) >= 2:
                st.divider()
                section_header('📊', 'Korrelationsmatrix', '')
                st.caption('Zeigt die Korrelation der täglichen Renditen (60 Tage) zwischen den '
                           'Signalen. Hohe Korrelation (rot) = ähnliches Risiko.')
                corr_fig = make_correlation_heatmap(daily, signal_syms)
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True, key='corr_long')

            # ── Sektor-Verteilung (Feature 3) ──
            if len(signal_syms) >= 1:
                st.divider()
                section_header('🏢', 'Sektor-Verteilung', '')
                with st.spinner('Sektordaten laden...'):
                    sunburst_fig = make_sector_sunburst(signal_syms, daily)
                if sunburst_fig:
                    st.plotly_chart(sunburst_fig, use_container_width=True, key='sunburst_long')
        else:
            st.info('Keine Long-Signale gefunden.')

    # ===================================================================
    # DAILY SHORT SIGNALSCAN
    # ===================================================================
    elif action == 'Daily Short Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'],
                           progress_cb=lambda d, t, m: prog_step(d, t, m))

        reg = daily[cfg['regime_symbol']].copy()
        reg['Date'] = pd.to_datetime(reg['Date'])
        reg = reg.sort_values('Date').set_index('Date')
        risk_on = bool(
            reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])
        regime_label = '🟢 Risk-On' if risk_on else '🔴 Risk-Off'

        ui_status.caption('Short-Scan läuft...')
        df_short = run_short_scan(daily, cfg)
        ui_status.caption('Daily Short scan: done')

        section_header('📉', f'Daily Short Signale (Regime: {regime_label})',
                       f'{len(df_short)} Signale')
        st.info(
            '🛑 **Short-Kandidaten**: Überhitzte Aktien mit RSI > {rsi}, '
            'EMA20-Abstand > {ema} %, 5d-Perf > {p5} %, Vol-Ratio > {vm}x. '
            'Sortiert nach **Überhitzungs-Score** (höher = heißer). '
            'TP-Zonen: EMA20, letztes Breakout-Level, Fib 38 %.'.format(
                rsi=int(cfg.get('short_rsi_min', 75)),
                ema=int(float(cfg.get('short_ema20_dist_min', 0.12)) * 100),
                p5=int(float(cfg.get('short_5d_perf_min', 0.10)) * 100),
                vm=cfg.get('short_vol_mult_min', 1.0),
            )
        )

        if not df_short.empty:
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
                    help='RSI(14) – über 75 = überkauft'),
                'ema20_dist_%': st.column_config.NumberColumn('EMA20-Abstand %', format='%.1f',
                    help='(Close/EMA20 - 1) in %'),
                '5d_perf_%': st.column_config.NumberColumn('5d-Perf %', format='%.1f',
                    help='5-Tage-Performance in %'),
                'vol_ratio': st.column_config.NumberColumn('Vol-Ratio', format='%.2f',
                    help='Volumen / VolSMA50'),
                'atr': st.column_config.NumberColumn('ATR', format='%.2f'),
                'short_stop': st.column_config.NumberColumn('Short Stop-Loss', format='%.2f',
                    help='Stop-Loss über letztem 20-Bar-Hoch + ATR'),
                'tp_ema20': st.column_config.NumberColumn('TP: EMA20', format='%.2f',
                    help='Take-Profit Zone 1: 20-Tage EMA'),
                'tp_breakout_level': st.column_config.NumberColumn('TP: Breakout-Level',
                    format='%.2f',
                    help='Take-Profit Zone 2: letztes 55-Tage-Hoch (Breakout-Level)'),
                'tp_fib38': st.column_config.NumberColumn('TP: Fib 38%', format='%.2f',
                    help='Take-Profit Zone 3: 38.2 % Fibonacci-Retracement des 20-Bar-Swing'),
                'ueberhitzung_score': st.column_config.NumberColumn('🔥 Überhitzung ★',
                    format='%.1f',
                    help='Score 0–10: je höher, desto überhitzter – besser für Short-Einstieg'),
            }
            st.dataframe(styled_short, column_config=short_col_cfg, use_container_width=True, key='short_signals_df')

            # ── Expandable signal detail charts ──
            signal_syms_short = df_short['symbol'].tolist()
            section_header('📊', 'Signal-Detail Charts', 'Interaktiv')
            for _, row in df_short.iterrows():
                sym = row['symbol']
                if sym not in daily:
                    continue
                with st.expander(f"📊 {sym} – Score: {row['ueberhitzung_score']} | "
                                 f"RSI: {row['rsi']} | Kurs: {row['price']}", expanded=False):
                    fig = make_short_signal_chart(
                        sym, daily[sym],
                        tp_ema20=row['tp_ema20'],
                        tp_bl=row['tp_breakout_level'],
                        tp_fib38=row['tp_fib38'],
                        short_stop=row['short_stop'],
                        entry_price=row['price'],
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f'chart_short_{sym}')

                    # R/R gauge (Feature 2) + metric cards
                    reward = row['price'] - row['tp_ema20']
                    risk = row['short_stop'] - row['price']
                    rr = reward / risk if risk > 0 else 0
                    g_col, mc_col = st.columns([1, 3])
                    with g_col:
                        rr_fig = make_rr_gauge(rr)
                        st.plotly_chart(rr_fig, use_container_width=True, key=f'rr_short_{sym}')
                    with mc_col:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.markdown(metric_card('📊', 'RSI(14)', f"{row['rsi']}", AMETHYST),
                                     unsafe_allow_html=True)
                        mc2.markdown(metric_card('📏', 'EMA20-Dist',
                                                 f"{row['ema20_dist_%']}%", SAPPHIRE),
                                     unsafe_allow_html=True)
                        mc3.markdown(metric_card('🚀', '5d-Perf',
                                                 f"{row['5d_perf_%']}%", RUBY),
                                     unsafe_allow_html=True)
                        mc4.markdown(metric_card('🔊', 'Vol-Ratio',
                                                 f"{row['vol_ratio']}x", GOLD),
                                     unsafe_allow_html=True)

                    # Watchlist button (Feature 6)
                    if st.button(f'➕ Watchlist', key=f'wl_short_{sym}'):
                        add_to_watchlist({
                            'symbol': sym, 'side': 'SHORT',
                            'entry': row['price'], 'stop': row['short_stop'],
                            'tp': row['tp_ema20'], 'date_added': _TODAY,
                        })
                        st.success(f'{sym} (Short) zur Watchlist hinzugefügt!')

            st.download_button('⬇ Download short_signals.csv',
                               data=df_short.to_csv(index=False).encode('utf-8'),
                               file_name='short_signals.csv', mime='text/csv', key='dl_short_signals')

            # ── Korrelationsmatrix (Feature 1) ──
            if len(signal_syms_short) >= 2:
                st.divider()
                section_header('📊', 'Korrelationsmatrix', '')
                st.caption('Zeigt die Korrelation der täglichen Renditen (60 Tage) zwischen den '
                           'Signalen. Hohe Korrelation (rot) = ähnliches Risiko.')
                corr_fig_s = make_correlation_heatmap(daily, signal_syms_short)
                if corr_fig_s:
                    st.plotly_chart(corr_fig_s, use_container_width=True, key='corr_short')

            # ── Sektor-Verteilung (Feature 3) ──
            if len(signal_syms_short) >= 1:
                st.divider()
                section_header('🏢', 'Sektor-Verteilung', '')
                with st.spinner('Sektordaten laden...'):
                    sunburst_fig_s = make_sector_sunburst(signal_syms_short, daily)
                if sunburst_fig_s:
                    st.plotly_chart(sunburst_fig_s, use_container_width=True, key='sunburst_short')
        else:
            st.info('Keine Short-Signale gefunden – aktuell erfüllt keine Aktie alle Filter.')

    # ===================================================================
    # COMPARISON MODE (Feature 8)
    # ===================================================================
    elif action == '⚖️ Vergleich':
        section_header('⚖️', 'Backtest-Vergleich', '')
        st.caption('Vergleiche zwei Konfigurationen nebeneinander.')

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader('Konfiguration A')
            preset_a = st.selectbox('Preset A', list(PRESETS.keys()), index=1, key='preset_a')
            cfg_a_text = st.text_area('Config A', value=json.dumps(load_preset_cfg(preset_a), indent=2),
                                      height=300, key='cfg_a_text')
        with col_b:
            st.subheader('Konfiguration B')
            preset_b = st.selectbox('Preset B', list(PRESETS.keys()), index=0, key='preset_b')
            cfg_b_text = st.text_area('Config B', value=json.dumps(load_preset_cfg(preset_b), indent=2),
                                      height=300, key='cfg_b_text')

        if st.button('Backtest vergleichen', type='primary', key='compare_run'):
            cfg_a = json.loads(cfg_a_text)
            cfg_b = json.loads(cfg_b_text)
            cfg_a['end'] = _TODAY
            cfg_b['end'] = _TODAY
            cfg_a['symbols'] = cfg.get('symbols', [])
            cfg_b['symbols'] = cfg.get('symbols', [])

            ui_status.caption('Daily Daten laden...')
            all_syms = sorted(list(set(
                cfg_a['symbols'] + cfg_b['symbols']
                + [cfg_a.get('regime_symbol', 'SPY'), cfg_b.get('regime_symbol', 'SPY')]
                + list(cfg_a.get('inverse_map', {}).values())
                + list(cfg_b.get('inverse_map', {}).values())
            )))
            daily = load_daily(all_syms, min(cfg_a.get('start', '2021-01-01'),
                                              cfg_b.get('start', '2021-01-01')),
                               _TODAY, progress_cb=lambda d, t, m: prog_step(d, t, m))
            cfg_a['symbols'] = [s for s in cfg_a['symbols'] if s in daily]
            cfg_b['symbols'] = [s for s in cfg_b['symbols'] if s in daily]
            cfg_a['sector_map'] = sector_map
            cfg_b['sector_map'] = sector_map

            ui_status.caption('Backtest A läuft...')
            eq_a, tr_a, sum_a, _ = run_backtest(daily, cfg_a, progress_cb=None)
            ui_status.caption('Backtest B läuft...')
            eq_b, tr_b, sum_b, _ = run_backtest(daily, cfg_b, progress_cb=None)
            ui_status.caption('Vergleich berechnet.')

            # Dual equity curves
            st.divider()
            fig_eq_cmp = go.Figure()
            eq_a_r = eq_a.reset_index()
            eq_b_r = eq_b.reset_index()
            fig_eq_cmp.add_trace(go.Scatter(x=eq_a_r['Date'], y=eq_a_r['Equity'],
                                            mode='lines', line=dict(color=GOLD, width=2),
                                            name='Config A'))
            fig_eq_cmp.add_trace(go.Scatter(x=eq_b_r['Date'], y=eq_b_r['Equity'],
                                            mode='lines', line=dict(color=SAPPHIRE, width=2),
                                            name='Config B'))
            _apply_dark_layout(fig_eq_cmp, 'Equity Curves – Vergleich', height=400)
            st.plotly_chart(fig_eq_cmp, use_container_width=True, key='cmp_equity')

            # Dual drawdown curves
            dd_a = (eq_a['Equity'] / eq_a['Equity'].cummax() - 1).reset_index()
            dd_a.columns = ['Date', 'DD']
            dd_b = (eq_b['Equity'] / eq_b['Equity'].cummax() - 1).reset_index()
            dd_b.columns = ['Date', 'DD']
            fig_dd_cmp = go.Figure()
            fig_dd_cmp.add_trace(go.Scatter(x=dd_a['Date'], y=dd_a['DD'],
                                            mode='lines', line=dict(color=GOLD, width=1.5),
                                            name='Config A DD'))
            fig_dd_cmp.add_trace(go.Scatter(x=dd_b['Date'], y=dd_b['DD'],
                                            mode='lines', line=dict(color=SAPPHIRE, width=1.5),
                                            name='Config B DD'))
            _apply_dark_layout(fig_dd_cmp, 'Drawdown – Vergleich', height=300)
            fig_dd_cmp.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig_dd_cmp, use_container_width=True, key='cmp_drawdown')

            # Metrics comparison table
            st.subheader('Kennzahlen-Vergleich')
            metrics_keys = ['final_equity', 'CAGR', 'MaxDrawdown', 'Volatility',
                           'Trades', 'ProfitFactor', 'WinRate', 'AvgWin', 'AvgLoss',
                           'Expectancy_R']
            metrics_labels = ['Endkapital', 'CAGR', 'Max. Drawdown', 'Volatilität',
                             'Trades', 'Profit Factor', 'Win Rate', 'Ø Gewinn',
                             'Ø Verlust', 'Expectancy (R)']
            higher_better = [True, True, False, False, None, True, True, True, False, True]

            rows_cmp = []
            for key, label, hb in zip(metrics_keys, metrics_labels, higher_better):
                va = sum_a.get(key, np.nan)
                vb = sum_b.get(key, np.nan)
                if key in ('CAGR', 'MaxDrawdown', 'Volatility', 'WinRate'):
                    fa = f"{va*100:.2f}%" if not pd.isna(va) else '-'
                    fb = f"{vb*100:.2f}%" if not pd.isna(vb) else '-'
                elif key == 'final_equity':
                    fa = f"€{va:,.2f}" if not pd.isna(va) else '-'
                    fb = f"€{vb:,.2f}" if not pd.isna(vb) else '-'
                elif key in ('AvgWin', 'AvgLoss'):
                    fa = f"€{va:.2f}" if not pd.isna(va) else '-'
                    fb = f"€{vb:.2f}" if not pd.isna(vb) else '-'
                elif key == 'Trades':
                    fa = str(int(va)) if not pd.isna(va) else '-'
                    fb = str(int(vb)) if not pd.isna(vb) else '-'
                else:
                    fa = f"{va:.2f}" if not pd.isna(va) else '-'
                    fb = f"{vb:.2f}" if not pd.isna(vb) else '-'
                winner = ''
                if hb is not None and not pd.isna(va) and not pd.isna(vb):
                    if hb:
                        winner = 'A' if va > vb else ('B' if vb > va else '=')
                    else:
                        winner = 'A' if abs(va) < abs(vb) else ('B' if abs(vb) < abs(va) else '=')
                rows_cmp.append({'Kennzahl': label, 'Config A': fa, 'Config B': fb,
                                'Besser': winner})
            cmp_df = pd.DataFrame(rows_cmp)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True, key='cmp_metrics_df')

            # Monthly heatmaps side by side
            st.subheader('Monatliche Rendite – Vergleich')
            hm_a, hm_b = st.columns(2)
            with hm_a:
                st.caption('Config A')
                fig_hm_a = make_monthly_heatmap(eq_a)
                if fig_hm_a:
                    st.plotly_chart(fig_hm_a, use_container_width=True, key='cmp_hm_a')
            with hm_b:
                st.caption('Config B')
                fig_hm_b = make_monthly_heatmap(eq_b)
                if fig_hm_b:
                    st.plotly_chart(fig_hm_b, use_container_width=True, key='cmp_hm_b')


# ═══════════════════════════════════════════════════════════════════════════
# WATCHLIST MODE (Feature 6) – works without clicking Start
# ═══════════════════════════════════════════════════════════════════════════
if action == '📋 Watchlist':
    section_header('📋', 'Watchlist & Alerts', '')

    wl = load_watchlist()

    if not wl:
        st.info('Watchlist ist leer. Füge Signale über den Long- oder Short-Scan hinzu (➕ Watchlist).')
    else:
        # Build watchlist table with live prices
        wl_rows = []
        for entry in wl:
            sym = entry['symbol']
            side = entry.get('side', 'LONG')
            entry_px = float(entry.get('entry', 0))
            stop_px = float(entry.get('stop', 0))
            tp_px = float(entry.get('tp', 0))
            date_added = entry.get('date_added', '')

            # Fetch current price
            try:
                t = yf.Ticker(sym)
                hist = t.history(period='1d')
                current_px = float(hist['Close'].iloc[-1]) if not hist.empty else np.nan
            except Exception:
                current_px = np.nan

            dist_to_entry = ((current_px / entry_px - 1) * 100) if (entry_px > 0 and np.isfinite(current_px)) else np.nan

            # Status determination
            if np.isnan(current_px):
                status = 'Unbekannt'
            elif side == 'LONG':
                if current_px <= stop_px:
                    status = 'Stop erreicht'
                elif current_px >= entry_px:
                    status = 'Entry erreicht!'
                else:
                    status = 'Warten'
            else:  # SHORT
                if current_px >= stop_px:
                    status = 'Stop erreicht'
                elif current_px <= entry_px:
                    status = 'Entry erreicht!'
                else:
                    status = 'Warten'

            wl_rows.append({
                'Symbol': sym, 'Seite': side, 'Entry': entry_px,
                'Stop': stop_px, 'TP': tp_px, 'Hinzugefügt': date_added,
                'Aktuell': round(current_px, 2) if np.isfinite(current_px) else None,
                'Dist. Entry %': round(dist_to_entry, 2) if not np.isnan(dist_to_entry) else None,
                'Status': status,
            })

        wl_df = pd.DataFrame(wl_rows)
        st.dataframe(wl_df, use_container_width=True, hide_index=True, key='watchlist_df')

        # Remove buttons
        st.caption('Eintrag entfernen:')
        for entry in wl:
            sym = entry['symbol']
            side = entry.get('side', 'LONG')
            if st.button(f'🗑 {sym} ({side}) entfernen', key=f'rm_{sym}_{side}'):
                remove_from_watchlist(sym, side)
                st.success(f'{sym} ({side}) entfernt.')
                st.rerun()

    # Alert configuration
    st.divider()
    section_header('🔔', 'Alert-Konfiguration', '')
    st.caption('Hinweis: Alerts werden nur geprüft, wenn die App geöffnet ist. '
               'Für automatische Alerts: `streamlit run app.py` mit Cron-Job oder separatem Alert-Script.')

    tg_token = st.text_input('Telegram Bot Token', type='password',
                              key='tg_token', value=st.session_state.get('tg_token_val', ''))
    tg_chat = st.text_input('Telegram Chat ID', key='tg_chat',
                             value=st.session_state.get('tg_chat_val', ''))
    st.session_state['tg_token_val'] = tg_token
    st.session_state['tg_chat_val'] = tg_chat

    if st.button('🔔 Alerts prüfen', key='check_alerts'):
        wl = load_watchlist()
        if not wl:
            st.info('Watchlist ist leer.')
        else:
            alerts = []
            for entry in wl:
                sym = entry['symbol']
                side = entry.get('side', 'LONG')
                entry_px = float(entry.get('entry', 0))
                stop_px = float(entry.get('stop', 0))
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period='1d')
                    cpx = float(hist['Close'].iloc[-1]) if not hist.empty else np.nan
                except Exception:
                    cpx = np.nan
                if np.isnan(cpx):
                    continue
                if side == 'LONG':
                    if cpx >= entry_px:
                        alerts.append(f'🟢 {sym} LONG Entry erreicht! Kurs: {cpx:.2f} >= {entry_px:.2f}')
                    elif cpx <= stop_px:
                        alerts.append(f'🔴 {sym} LONG Stop erreicht! Kurs: {cpx:.2f} <= {stop_px:.2f}')
                else:
                    if cpx <= entry_px:
                        alerts.append(f'🟢 {sym} SHORT Entry erreicht! Kurs: {cpx:.2f} <= {entry_px:.2f}')
                    elif cpx >= stop_px:
                        alerts.append(f'🔴 {sym} SHORT Stop erreicht! Kurs: {cpx:.2f} >= {stop_px:.2f}')

            if alerts:
                for a in alerts:
                    st.warning(a)
                # Send Telegram if configured
                if tg_token and tg_chat:
                    msg = '<b>🔔 Zero Signal Scanner – Alerts</b>\n\n' + '\n'.join(alerts)
                    ok = send_telegram_alert(tg_token, tg_chat, msg)
                    if ok:
                        st.success('Telegram-Benachrichtigung gesendet!')
                    else:
                        st.error('Telegram-Versand fehlgeschlagen.')
            else:
                st.success('Keine Alerts – alle Watchlist-Einträge im Wartebereich.')


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="premium-footer">
    3S – Stock Signal Scanner by AF &bull; {_TODAY} &bull;
    Daten: Yahoo Finance &bull; Keine Anlageberatung
</div>
""", unsafe_allow_html=True)
