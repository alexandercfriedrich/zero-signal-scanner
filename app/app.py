import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

from backtest_engine import run_backtest

st.set_page_config(page_title='Signal Scanner (US+EU, Intraday)', layout='wide')

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

  "breakout_lookback": 55,
  "sma_regime": 200,
  "max_holding_days": 30,

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

  "spread_bps_per_side": 8,
  "min_price": 2.0,
  "min_dollar_volume": 2_000_000,
  "initial_cash": 5000
}

# Presets (loaded from JSON files in app/)
PRESETS = {
    'Swing (Top-5, Risk-On only)': None,  # uses DEFAULT_CFG
    'Best (2011-2026)': 'config_best_2011_2026.json',
}


def load_preset_cfg(preset_name: str) -> dict:
    """Load preset config dict. Returns DEFAULT_CFG for the default preset."""
    p = PRESETS.get(preset_name)
    if not p:
        return dict(DEFAULT_CFG)
    preset_path = (Path(__file__).parent / p)
    try:
        return json.loads(preset_path.read_text())
    except Exception:
        # Fall back to default if file is missing or invalid
        return dict(DEFAULT_CFG)


st.title('Signal‑Scanner: S&P 500 + EU (ISIN/WKN/Name) + Intraday Updates')
st.caption('S&P 500 auto, EU Resolver (DE/AT), yfinance Download + Disk‑Cache. Intraday scan compares latest intraday price vs daily breakout levels.')

with st.sidebar:
    st.header('Universe')
    universe_mode = st.radio('Quelle', ['S&P 500 (Wikipedia)', 'Custom (ISIN/WKN/Name/Yahoo Ticker)'], index=0)
    custom_list = ''
    if universe_mode.startswith('Custom'):
        custom_list = st.text_area('Liste (eine Zeile pro Eintrag)', value='AAPL\nMSFT\nVIG.VI\nDB1.DE\nAT0000937503')

    st.header('Aktion')
    action = st.radio('Modus', ['Intraday Signalscan', 'Daily Signalscan', 'Backtest (Daily, 5y)'], index=0)
    intraday_interval = st.selectbox('Intraday-Intervall', ['5m','15m','30m','60m'], index=1)
    intraday_period = st.selectbox('Intraday-Periode', ['7d','30d','60d'], index=2)

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


def load_intraday(symbols: list[str], interval: str, period: str, progress_cb=None) -> dict[str, pd.DataFrame]:
    out = {}
    kind = f'{interval}_{period}'
    need = []
    for s in symbols:
        p = cache_path(s, kind)
        if p.exists():
            try:
                d = pd.read_parquet(p)
                col = 'Datetime' if 'Datetime' in d.columns else ('Date' if 'Date' in d.columns else None)
                if col:
                    d[col] = pd.to_datetime(d[col])
                    if d[col].max().date() >= (pd.Timestamp.today().date() - pd.Timedelta(days=1)):
                        out[s] = d.rename(columns={col:'Datetime'})
                        continue
            except Exception:
                pass
        need.append(s)

    if need:
        chunk = 60
        total = int(np.ceil(len(need) / chunk))
        for k, i in enumerate(range(0, len(need), chunk)):
            batch = need[i:i+chunk]
            if progress_cb is not None:
                progress_cb(k, total, f'Intraday: {i}/{len(need)}')
            got = unpack_download(fetch_yf(batch, interval=interval, period=period), batch)
            for sym, sub in got.items():
                if 'Datetime' not in sub.columns:
                    if 'Date' in sub.columns:
                        sub = sub.rename(columns={'Date':'Datetime'})
                    else:
                        sub = sub.rename(columns={'index':'Datetime'})
                out[sym] = sub
                sub.to_parquet(cache_path(sym, kind), index=False)
        if progress_cb is not None:
            progress_cb(total, total, 'Intraday: done')

    return out


def intraday_scan(daily_data: dict[str, pd.DataFrame], intraday_data: dict[str, pd.DataFrame], cfg: dict):
    lookback = cfg['breakout_lookback']
    rows = []

    reg = daily_data[cfg['regime_symbol']].copy()
    reg['Date'] = pd.to_datetime(reg['Date'])
    reg = reg.sort_values('Date').set_index('Date')
    risk_on = bool(reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])

    for sym, ddf in daily_data.items():
        if sym == cfg['regime_symbol']:
            continue
        if sym in cfg.get('inverse_map', {}).values():
            continue
        if sym not in intraday_data:
            continue

        df = ddf.copy(); df['Date']=pd.to_datetime(df['Date']); df=df.sort_values('Date').set_index('Date')
        if len(df) < lookback + 5:
            continue
        hh = df['High'].shift(1).rolling(lookback).max().iloc[-1]
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr_v = tr.rolling(cfg['atr_period']).mean().iloc[-1]
        if pd.isna(hh) or pd.isna(atr_v) or float(atr_v) <= 0:
            continue

        idf = intraday_data[sym].copy(); idf['Datetime']=pd.to_datetime(idf['Datetime']); idf=idf.sort_values('Datetime')
        last = idf.iloc[-1]
        px = float(last['Close'])

        if risk_on and px > float(hh):
            score = (px - float(hh)) / float(atr_v)

            risk_per_share = float(cfg['atr_stop_mult'] * float(atr_v))
            stop_price = float(px - risk_per_share)
            tp_price = float(px + float(cfg.get('take_profit_R', 2.0)) * risk_per_share)
            shares_for_1000eur = int(max(0, (1000.0 * float(cfg['risk_per_trade'])) // max(1e-9, risk_per_share)))

            rows.append({
                'symbol': sym,
                'side': 'LONG',
                'price': px,
                'breakout_level': float(hh),
                'score': float(score),
                'asof': str(last['Datetime']),
                'atr': float(atr_v),
                'risk_per_share': float(risk_per_share),
                'stop_price': float(stop_price),
                'tp_price': float(tp_price),
                'shares_for_1000eur': shares_for_1000eur,
            })

    return pd.DataFrame(rows).sort_values('score', ascending=False), risk_on


if run_btn:
    cfg = json.loads(cfg_text)

    with st.spinner('Universe laden & ggf. auflösen...'):
        if universe_mode.startswith('S&P 500'):
            symbols = load_sp500_symbols()
            resolve_table = pd.DataFrame({'input': symbols, 'resolved_symbol': symbols})
        else:
            lines = [x for x in custom_list.splitlines()]
            symbols, resolve_table = resolve_inputs(lines, prefer_regions)

    st.subheader('Symbol‑Auflösung')
    st.dataframe(resolve_table, use_container_width=True)

    cfg = dict(cfg)
    cfg['symbols'] = [s for s in symbols if s not in [cfg['regime_symbol']] and s not in cfg.get('inverse_map', {}).values()]
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
        m1.metric('Final Equity', f"{summary['final_equity']:.2f}")
        m2.metric('CAGR', f"{summary['CAGR']*100:.2f}%")
        m3.metric('Max Drawdown', f"{summary['MaxDrawdown']*100:.2f}%")
        m4.metric('Volatility', f"{summary['Volatility']*100:.2f}%")
        m5.metric('Trades', str(summary['Trades']))

        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric('Profit Factor', '-' if pd.isna(summary.get('ProfitFactor')) else f"{summary.get('ProfitFactor'):.2f}")
        m7.metric('Win Rate', '-' if pd.isna(summary.get('WinRate')) else f"{summary.get('WinRate')*100:.1f}%")
        m8.metric('Avg Win', '-' if pd.isna(summary.get('AvgWin')) else f"{summary.get('AvgWin'):.2f}")
        m9.metric('Avg Loss', '-' if pd.isna(summary.get('AvgLoss')) else f"{summary.get('AvgLoss'):.2f}")
        m10.metric('Expectancy (R)', '-' if pd.isna(summary.get('Expectancy_R')) else f"{summary.get('Expectancy_R'):.2f}")

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
        with b1:
            st.caption('By setup')
            st.dataframe(breakdown.get('by_setup', pd.DataFrame()), use_container_width=True)
        with b2:
            st.caption('By reason')
            st.dataframe(breakdown.get('by_reason', pd.DataFrame()), use_container_width=True)

        st.subheader('Trades')
        st.dataframe(trades_df, use_container_width=True)

    elif action == 'Daily Signalscan':
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'], progress_cb=lambda d,t,m: prog_step(d, t, m))

        reg = daily[cfg['regime_symbol']].copy(); reg['Date']=pd.to_datetime(reg['Date']); reg=reg.sort_values('Date').set_index('Date')
        risk_on = bool(reg['Close'].iloc[-1] > reg['Close'].rolling(cfg['sma_regime']).mean().iloc[-1])

        rows = []
        for sym in cfg['symbols']:
            if sym not in daily:
                continue
            df = daily[sym].copy(); df['Date']=pd.to_datetime(df['Date']); df=df.sort_values('Date').set_index('Date')
            hh = df['High'].shift(1).rolling(cfg['breakout_lookback']).max().iloc[-1]
            high, low, close = df['High'], df['Low'], df['Close']
            tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
            atr_v = tr.rolling(cfg['atr_period']).mean().iloc[-1]
            if pd.isna(hh) or pd.isna(atr_v) or float(atr_v) <= 0:
                continue
            px = float(df['Close'].iloc[-1])
            if risk_on and px > float(hh):
                risk_per_share = float(cfg['atr_stop_mult'] * float(atr_v))
                stop_price = float(px - risk_per_share)
                tp_price = float(px + float(cfg.get('take_profit_R', 2.0)) * risk_per_share)
                shares_for_1000eur = int(max(0, (1000.0 * float(cfg['risk_per_trade'])) // max(1e-9, risk_per_share)))
                rows.append({'symbol': sym, 'side':'LONG', 'price':px, 'breakout_level':float(hh), 'asof': str(df.index[-1].date()), 'atr': float(atr_v), 'risk_per_share': risk_per_share, 'stop_price': stop_price, 'tp_price': tp_price, 'shares_for_1000eur': shares_for_1000eur})

        ui_status.caption('Daily scan: done')
        st.subheader(f'Daily Signale (RiskOn={risk_on})')
        st.dataframe(pd.DataFrame(rows).sort_values('atr', ascending=False), use_container_width=True)

    else:
        ui_status.caption('Daily Daten laden...')
        daily = load_daily(needed_daily, cfg['start'], cfg['end'], progress_cb=lambda d,t,m: prog_step(d, t, m))
        ui_status.caption('Intraday Daten laden...')
        intra = load_intraday([s for s in needed_daily if s in daily], interval=intraday_interval, period=intraday_period, progress_cb=lambda d,t,m: prog_step(d, t, m))

        ui_status.caption('Intraday scan...')
        sig, risk_on = intraday_scan(daily, intra, cfg)
        ui_status.caption('Intraday scan: done')
        st.subheader(f'Intraday Breakout‑Signale (RiskOn={risk_on})')
        st.dataframe(sig.head(50), use_container_width=True)
