import json
import time
from pathlib import Path

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
    preset = st.selectbox('Preset', ['Swing (Top-5, Risk-On only)'], index=0)
    cfg_default_text = json.dumps(DEFAULT_CFG, indent=2)

    cfg_text = st.text_area('config.json (ohne symbols)', value=cfg_default_text, height=420)
    run_btn = st.button('Start', type='primary')


@st.cache_data(ttl=6*60*60, show_spinner=False)
def load_sp500_symbols() -> list[str]:
    """Load S&P 500 tickers.

    Streamlit Cloud sometimes blocks/ratelimits Wikipedia HTML parsing via pandas.read_html.
    We therefore try:
      1) Wikipedia via requests + pandas.read_html on the HTML text
      2) GitHub raw CSV fallback (datasets repo)
    """

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


def load_daily(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
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
        for i in range(0, len(need), chunk):
            batch = need[i:i+chunk]
            got = unpack_download(fetch_yf(batch, start=start, end=end, interval='1d'), batch)
            for sym, sub in got.items():
                out[sym] = sub
                sub.to_parquet(cache_path(sym, '1d'), index=False)
    return out


def load_intraday(symbols: list[str], interval: str, period: str) -> dict[str, pd.DataFrame]:
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
        for i in range(0, len(need), chunk):
            batch = need[i:i+chunk]
            got = unpack_download(fetch_yf(batch, interval=interval, period=period), batch)
            for sym, sub in got.items():
                if 'Datetime' not in sub.columns:
                    if 'Date' in sub.columns:
                        sub = sub.rename(columns={'Date':'Datetime'})
                    else:
                        sub = sub.rename(columns={'index':'Datetime'})
                out[sym] = sub
                sub.to_parquet(cache_path(sym, kind), index=False)
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

    if action == 'Backtest (Daily, 5y)':
        with st.spinner(f'Daily Daten laden ({len(needed_daily)})...'):
            daily = load_daily(needed_daily, cfg['start'], cfg['end'])
        cfg['symbols'] = [s for s in cfg['symbols'] if s in daily]

        with st.spinner('Backtest läuft...'):
            equity_df, trades_df, summary = run_backtest(daily, cfg)

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
        m10.metric('Expectancy (R approx)', '-' if pd.isna(summary.get('Expectancy_R')) else f"{summary.get('Expectancy_R'):.2f}")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.plotly_chart(px.line(equity_df.reset_index(), x='Date', y='Equity', title='Equity Curve'), use_container_width=True)
        with c2:
            eq = equity_df['Equity']
            dd = (eq/eq.cummax() - 1).rename('Drawdown')
            st.plotly_chart(px.area(dd.reset_index(), x='Date', y='Drawdown', title='Drawdown'), use_container_width=True)

        st.subheader('Trades')
        st.dataframe(trades_df, use_container_width=True)

    elif action == 'Daily Signalscan':
        with st.spinner(f'Daily Daten laden ({len(needed_daily)})...'):
            daily = load_daily(needed_daily, cfg['start'], cfg['end'])

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

        st.subheader(f'Daily Signale (RiskOn={risk_on})')
        st.dataframe(pd.DataFrame(rows).sort_values('atr', ascending=False), use_container_width=True)

    else:
        with st.spinner(f'Daily Daten laden ({len(needed_daily)})...'):
            daily = load_daily(needed_daily, cfg['start'], cfg['end'])
        with st.spinner(f'Intraday Daten laden ({len(needed_daily)})...'):
            intra = load_intraday([s for s in needed_daily if s in daily], interval=intraday_interval, period=intraday_period)

        sig, risk_on = intraday_scan(daily, intra, cfg)
        st.subheader(f'Intraday Breakout‑Signale (RiskOn={risk_on})')
        st.dataframe(sig.head(50), use_container_width=True)
