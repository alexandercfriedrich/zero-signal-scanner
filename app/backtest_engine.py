import pandas as pd
import numpy as np


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def dollar_volume(df: pd.DataFrame) -> pd.Series:
    return df['Close'] * df['Volume']


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=cols)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('Missing Date column and index is not DatetimeIndex')
        df = df.sort_index()

    for c in ['Open', 'High', 'Low', 'Close']:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    return df


def run_backtest(data: dict, cfg: dict):
    symbols = cfg['symbols']
    regime_symbol = cfg['regime_symbol']

    data = {s: normalize_ohlcv(df) for s, df in data.items()}

    needed = set(symbols + [regime_symbol] + list(cfg.get('inverse_map', {}).values()))
    missing = sorted([s for s in needed if s not in data])
    if missing:
        raise ValueError(f"Missing symbols in loaded data: {missing}")

    # align dates
    idx = None
    for s in needed:
        idx = data[s].index if idx is None else idx.intersection(data[s].index)

    start = pd.Timestamp(cfg['start'])
    end = pd.Timestamp(cfg['end'])
    idx = idx[(idx >= start) & (idx <= end)]

    for s in needed:
        df = data[s].reindex(idx)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        data[s] = df

    # recompute idx after dropna
    idx = None
    for s in needed:
        idx = data[s].index if idx is None else idx.intersection(data[s].index)
    for s in needed:
        data[s] = data[s].reindex(idx).dropna(subset=['Open', 'High', 'Low', 'Close'])

    # indicators
    for s in needed:
        df = data[s]
        df['ATR'] = atr(df, cfg['atr_period'])
        df['SMA_regime'] = sma(df['Close'], cfg['sma_regime'])
        df['HH'] = df['High'].shift(1).rolling(cfg['breakout_lookback']).max()
        df['LL'] = df['Low'].shift(1).rolling(cfg['breakout_lookback']).min()
        df['DollarVol'] = dollar_volume(df)

    reg = data[regime_symbol]
    reg['RiskOn'] = reg['Close'] > reg['SMA_regime']

    spread = cfg['spread_bps_per_side'] / 10_000.0

    cash = float(cfg['initial_cash'])
    equity_rows = []
    open_positions = []
    trades = []

    def mark_to_market(date):
        eq = cash
        for p in open_positions:
            px = float(data[p['symbol']].loc[date, 'Close'])
            eq += p['shares'] * px
        return float(eq)

    for i, date in enumerate(idx):
        # exits first
        new_open = []
        for p in open_positions:
            df = data[p['symbol']]
            h = float(df.loc[date, 'High'])
            l = float(df.loc[date, 'Low'])
            c = float(df.loc[date, 'Close'])

            exit_px = None
            reason = None
            # conservative intraday path
            if l <= p['stop']:
                exit_px = p['stop']
                reason = 'STOP'
            elif h >= p['tp']:
                exit_px = p['tp']
                reason = 'TP'

            if exit_px is None and (date - p['entry_date']).days >= cfg['max_holding_days']:
                exit_px = c
                reason = 'TIME'

            if exit_px is None:
                new_open.append(p)
                continue

            exit_px_eff = exit_px * (1 - spread)
            cash += p['shares'] * exit_px_eff
            trades.append(
                {
                    'symbol': p['symbol'],
                    'side': p['side'],
                    'entry_date': p['entry_date'].date().isoformat(),
                    'entry_px': p['entry_px'],
                    'exit_date': date.date().isoformat(),
                    'exit_px': exit_px_eff,
                    'shares': p['shares'],
                    'pnl': (exit_px_eff - p['entry_px']) * p['shares'],
                    'reason': reason,
                }
            )

        open_positions = new_open

        eq_today = mark_to_market(date)
        equity_rows.append({'Date': date, 'Equity': eq_today, 'Cash': cash, 'OpenPositions': len(open_positions)})

        # entries require next open
        if i >= len(idx) - 1:
            continue
        next_date = idx[i + 1]

        risk_on = bool(reg.loc[date, 'RiskOn']) if not pd.isna(reg.loc[date, 'RiskOn']) else False
        if len(open_positions) >= cfg['max_positions']:
            continue

        candidates = []
        if risk_on:
            for sym in symbols:
                row = data[sym].loc[date]
                if pd.isna(row['HH']) or pd.isna(row['ATR']) or float(row['Close']) < cfg['min_price']:
                    continue
                dv = row['DollarVol']
                if not pd.isna(dv) and float(dv) < cfg['min_dollar_volume']:
                    continue
                if float(row['Close']) > float(row['HH']):
                    score = (float(row['Close']) - float(row['HH'])) / float(row['ATR'])
                    candidates.append((sym, score))
        else:
            inv = cfg.get('inverse_map', {}).get(regime_symbol)
            if inv and inv in data:
                row = reg.loc[date]
                if (not pd.isna(row['LL'])) and (not pd.isna(row['ATR'])) and (float(row['Close']) < float(row['LL'])):
                    score = (float(row['LL']) - float(row['Close'])) / float(row['ATR'])
                    candidates.append((inv, score))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        n_new = min(cfg['max_new_trades_per_day'], cfg['max_positions'] - len(open_positions))
        picks = candidates[:n_new]

        for sym, _ in picks:
            entry_px = float(data[sym].loc[next_date, 'Open'])
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue

            entry_px_eff = entry_px * (1 + spread)
            atr_v = float(data[sym].loc[date, 'ATR'])
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue

            stop_dist = cfg['atr_stop_mult'] * atr_v
            stop = entry_px_eff - stop_dist
            tp = entry_px_eff + cfg['take_profit_R'] * stop_dist

            risk_eur = cfg['risk_per_trade'] * eq_today
            denom = max(1e-9, entry_px_eff - stop)
            shares = int(max(0, np.floor(risk_eur / denom)))
            if shares <= 0:
                continue

            cost = shares * entry_px_eff
            if cost > cash:
                shares = int(np.floor(cash / entry_px_eff))
                if shares <= 0:
                    continue
                cost = shares * entry_px_eff

            cash -= cost
            open_positions.append(
                {
                    'symbol': sym,
                    'side': 'LONG',
                    'entry_date': next_date,
                    'entry_px': entry_px_eff,
                    'shares': shares,
                    'stop': stop,
                    'tp': tp,
                }
            )

    equity_df = pd.DataFrame(equity_rows).set_index('Date')
    trades_df = pd.DataFrame(trades)

    eq = equity_df['Equity']
    ret = eq.pct_change().fillna(0)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(1, len(eq) - 1)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / (ret.std() * np.sqrt(252) + 1e-12)
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min()

    summary = {
        'start': str(eq.index.min().date()),
        'end': str(eq.index.max().date()),
        'initial_cash': float(cfg['initial_cash']),
        'final_equity': float(eq.iloc[-1]),
        'CAGR': float(cagr),
        'Volatility': float(vol),
        'Sharpe_approx': float(sharpe),
        'MaxDrawdown': float(max_dd),
        'Trades': int(len(trades_df)),
    }

    return equity_df, trades_df, summary
