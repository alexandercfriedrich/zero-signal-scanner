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
    """Daily backtest.

    IMPORTANT: For variant-1 backtests ("today's constituents"), we must not
    force an intersection of dates across all symbols. That can collapse the
    timeline to near-zero because some tickers have missing history.

    We therefore:
      - use the regime symbol calendar as the master index
      - for each symbol on each day, skip if data is missing

    Swing preset support:
      - hard risk-on gate (if configured): no new entries when market is risk-off
    """

    symbols = cfg['symbols']
    regime_symbol = cfg['regime_symbol']

    data = {s: normalize_ohlcv(df) for s, df in data.items()}

    needed = set(symbols + [regime_symbol] + list(cfg.get('inverse_map', {}).values()))
    missing = sorted([s for s in needed if s not in data])
    if missing:
        raise ValueError(f"Missing symbols in loaded data: {missing}")

    # master calendar from regime symbol
    start = pd.Timestamp(cfg['start'])
    end = pd.Timestamp(cfg['end'])
    idx = data[regime_symbol].index
    idx = idx[(idx >= start) & (idx <= end)]

    # reindex regime and inverse symbols to master calendar (keep NaN; we will guard)
    for s in [regime_symbol] + list(cfg.get('inverse_map', {}).values()):
        if s in data:
            data[s] = data[s].reindex(idx)

    # indicators (compute per symbol on its own index; then align to master calendar)
    for s in needed:
        df = data[s]
        # If the symbol isn't on the master calendar yet (universe symbols), align it.
        if not df.index.equals(idx):
            df = df.reindex(idx)
            data[s] = df

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

    # swing preset: hard gate
    hard_risk_on = bool(cfg.get('hard_risk_on', False))

    def mark_to_market(date):
        eq = cash
        for p in open_positions:
            px = data[p['symbol']].loc[date, 'Close']
            if pd.isna(px):
                continue
            eq += p['shares'] * float(px)
        return float(eq)

    for i, date in enumerate(idx):
        # exits first
        new_open = []
        for p in open_positions:
            df = data[p['symbol']]
            h = df.loc[date, 'High']
            l = df.loc[date, 'Low']
            c = df.loc[date, 'Close']
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                new_open.append(p)
                continue

            h = float(h)
            l = float(l)
            c = float(c)

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

        # hard gate: no entries when risk off
        if hard_risk_on and (not risk_on):
            continue

        if len(open_positions) >= cfg['max_positions']:
            continue

        candidates = []
        if risk_on:
            for sym in symbols:
                row = data[sym].loc[date]
                # if any key field missing, skip that symbol on that date
                if pd.isna(row.get('HH')) or pd.isna(row.get('ATR')) or pd.isna(row.get('Close')):
                    continue
                if float(row['Close']) < cfg['min_price']:
                    continue
                dv = row.get('DollarVol')
                if (dv is not None) and (not pd.isna(dv)) and float(dv) < cfg['min_dollar_volume']:
                    continue
                if float(row['Close']) > float(row['HH']):
                    score = (float(row['Close']) - float(row['HH'])) / float(row['ATR'])
                    candidates.append((sym, score))
        else:
            # legacy risk-off hedge behavior (disabled when hard_risk_on is True)
            inv = cfg.get('inverse_map', {}).get(regime_symbol)
            if inv and inv in data:
                row = reg.loc[date]
                if (not pd.isna(row.get('LL'))) and (not pd.isna(row.get('ATR'))) and (not pd.isna(row.get('Close'))):
                    if float(row['Close']) < float(row['LL']):
                        score = (float(row['LL']) - float(row['Close'])) / float(row['ATR'])
                        candidates.append((inv, score))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        n_new = min(cfg['max_new_trades_per_day'], cfg['max_positions'] - len(open_positions))
        picks = candidates[:n_new]

        for sym, _ in picks:
            o = data[sym].loc[next_date, 'Open']
            if pd.isna(o):
                continue
            entry_px = float(o)
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue

            entry_px_eff = entry_px * (1 + spread)
            atr_raw = data[sym].loc[date, 'ATR']
            if pd.isna(atr_raw):
                continue
            atr_v = float(atr_raw)
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
