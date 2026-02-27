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


def pct_return(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1


def detect_cup_handle(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    cup_min = int(cfg.get('cwh_cup_min_bars', 30))
    cup_max = int(cfg.get('cwh_cup_max_bars', 130))
    handle_min = int(cfg.get('cwh_handle_min_bars', 5))
    handle_max = int(cfg.get('cwh_handle_max_bars', 20))
    max_cup_depth = float(cfg.get('cwh_max_cup_depth', 0.35))
    max_handle_depth = float(cfg.get('cwh_max_handle_depth', 0.15))

    trend_sma_n = int(cfg.get('cwh_trend_sma', 50))
    df['CWH_TrendOK'] = close > sma(close, trend_sma_n)

    df['VolSMA20'] = sma(vol, 20)

    pivots = np.full(len(df), np.nan)
    handle_lows = np.full(len(df), np.nan)
    vol_ok = np.full(len(df), False)

    for t in range(len(df)):
        if t < cup_min + handle_min + 5:
            continue

        if not bool(df['CWH_TrendOK'].iloc[t]):
            continue

        best = None
        h_end = t - 1
        for h_len in range(handle_min, handle_max + 1):
            h_start = h_end - h_len + 1
            if h_start <= 0:
                continue

            handle_high = float(high.iloc[h_start:h_end + 1].max())
            handle_low = float(low.iloc[h_start:h_end + 1].min())
            if not np.isfinite(handle_high) or not np.isfinite(handle_low) or handle_high <= 0:
                continue

            handle_depth = (handle_high - handle_low) / handle_high
            if handle_depth > max_handle_depth:
                continue

            cup_end = h_start - 1
            for cup_len in range(cup_min, cup_max + 1, 5):
                cup_start = cup_end - cup_len + 1
                if cup_start <= 0:
                    continue

                cup_high = float(high.iloc[cup_start:cup_end + 1].max())
                cup_low = float(low.iloc[cup_start:cup_end + 1].min())
                if not np.isfinite(cup_high) or not np.isfinite(cup_low) or cup_high <= 0:
                    continue

                cup_depth = (cup_high - cup_low) / cup_high
                if cup_depth <= 0 or cup_depth > max_cup_depth:
                    continue

                if handle_high < (0.8 * cup_high):
                    continue

                cup_vol = df['VolSMA20'].iloc[cup_start:cup_end + 1].median() if 'VolSMA20' in df.columns else np.nan
                handle_vol = df['VolSMA20'].iloc[h_start:h_end + 1].median() if 'VolSMA20' in df.columns else np.nan
                v_ok = False
                if np.isfinite(cup_vol) and np.isfinite(handle_vol) and cup_vol > 0:
                    v_ok = bool(handle_vol < cup_vol)

                quality = 0.0
                quality += (1.0 - handle_depth) * 2.0
                quality += (1.0 - cup_depth)
                quality += 0.5 if v_ok else 0.0

                if (best is None) or (quality > best[0]):
                    best = (quality, handle_high, handle_low, v_ok)

        if best is None:
            continue

        _, p, hl, v_ok = best
        pivots[t] = p
        handle_lows[t] = hl
        vol_ok[t] = v_ok

    df['CWH_Pivot'] = pivots
    df['CWH_HandleLow'] = handle_lows
    df['CWH_VolOK'] = vol_ok
    df['CWH_Signal'] = (df['Close'] > df['CWH_Pivot']) & np.isfinite(df['CWH_Pivot'])

    return df


def compute_trade_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {
            'ProfitFactor': np.nan,
            'WinRate': np.nan,
            'AvgWin': np.nan,
            'AvgLoss': np.nan,
            'Expectancy_R': np.nan,
        }

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] < 0]

    gross_win = float(wins['pnl'].sum()) if not wins.empty else 0.0
    gross_loss = float((-losses['pnl']).sum()) if not losses.empty else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf

    win_rate = float(len(wins) / len(trades_df))
    avg_win = float(wins['pnl'].mean()) if not wins.empty else 0.0
    avg_loss = float(losses['pnl'].mean()) if not losses.empty else 0.0

    # True R-multiples when available
    exp_r = np.nan
    if 'R_multiple' in trades_df.columns:
        exp_r = float(trades_df['R_multiple'].mean())

    return {
        'ProfitFactor': float(pf),
        'WinRate': float(win_rate),
        'AvgWin': float(avg_win),
        'AvgLoss': float(avg_loss),
        'Expectancy_R': float(exp_r) if np.isfinite(exp_r) else np.nan,
    }


def breakdown_tables(trades_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    if trades_df is None or trades_df.empty:
        out['by_setup'] = pd.DataFrame()
        out['by_reason'] = pd.DataFrame()
        return out

    def agg(g: pd.DataFrame) -> pd.Series:
        wins = g[g['pnl'] > 0]
        losses = g[g['pnl'] < 0]
        gross_win = float(wins['pnl'].sum()) if not wins.empty else 0.0
        gross_loss = float((-losses['pnl']).sum()) if not losses.empty else 0.0
        pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf
        win_rate = float(len(wins) / len(g)) if len(g) else np.nan
        avg_r = float(g['R_multiple'].mean()) if 'R_multiple' in g.columns and not g['R_multiple'].isna().all() else np.nan
        return pd.Series({
            'Trades': int(len(g)),
            'WinRate': win_rate,
            'ProfitFactor': pf,
            'AvgPnL': float(g['pnl'].mean()) if len(g) else np.nan,
            'AvgR': avg_r,
        })

    out['by_setup'] = trades_df.groupby('setup', dropna=False).apply(agg).reset_index()
    out['by_reason'] = trades_df.groupby('reason', dropna=False).apply(agg).reset_index()
    return out


def run_backtest(data: dict, cfg: dict):
    symbols = cfg['symbols']
    regime_symbol = cfg['regime_symbol']

    data = {s: normalize_ohlcv(df) for s, df in data.items()}

    needed = set(symbols + [regime_symbol] + list(cfg.get('inverse_map', {}).values()))
    missing = sorted([s for s in needed if s not in data])
    if missing:
        raise ValueError(f"Missing symbols in loaded data: {missing}")

    start = pd.Timestamp(cfg['start'])
    end = pd.Timestamp(cfg['end'])
    idx = data[regime_symbol].index
    idx = idx[(idx >= start) & (idx <= end)]

    for s in [regime_symbol] + list(cfg.get('inverse_map', {}).values()):
        if s in data:
            data[s] = data[s].reindex(idx)

    for s in needed:
        df = data[s]
        if not df.index.equals(idx):
            df = df.reindex(idx)
            data[s] = df

        df['ATR'] = atr(df, cfg['atr_period'])
        df['SMA_regime'] = sma(df['Close'], cfg['sma_regime'])
        df['PivotClose'] = df['Close'].shift(1).rolling(cfg['breakout_lookback']).max()
        df['HH'] = df['High'].shift(1).rolling(cfg['breakout_lookback']).max()
        df['LL'] = df['Low'].shift(1).rolling(cfg['breakout_lookback']).min()
        df['DollarVol'] = dollar_volume(df)

        mom_n = int(cfg.get('mom_lookback', 126))
        df['Mom'] = pct_return(df['Close'], mom_n)

        df['VolSMA50'] = sma(df['Volume'], 50)
        df['VolSMA20'] = sma(df['Volume'], 20)

        if bool(cfg.get('enable_cwh', True)):
            data[s] = detect_cup_handle(df, cfg)

    reg = data[regime_symbol]
    reg['RiskOn'] = reg['Close'] > reg['SMA_regime']

    spread = cfg['spread_bps_per_side'] / 10_000.0

    cash = float(cfg['initial_cash'])
    equity_rows = []
    open_positions = []
    trades = []

    hard_risk_on = bool(cfg.get('hard_risk_on', False))
    weekly_rerank = bool(cfg.get('weekly_rerank', True))

    use_trailing = bool(cfg.get('use_trailing_stop', True))
    trail_mult = float(cfg.get('atr_trail_mult', cfg.get('atr_stop_mult', 2.0)))

    def mark_to_market(date):
        eq = cash
        for p in open_positions:
            px = data[p['symbol']].loc[date, 'Close']
            if pd.isna(px):
                continue
            eq += p['shares'] * float(px)
        return float(eq)

    def is_weekly_rebalance_day(date: pd.Timestamp) -> bool:
        wd = int(date.weekday())
        return wd == int(cfg.get('weekly_rebalance_weekday', 0))

    def score_candidate(sym: str, date: pd.Timestamp) -> float:
        row = data[sym].loc[date]
        if pd.isna(row.get('ATR')) or float(row['ATR']) <= 0:
            return -np.inf

        mom = row.get('Mom')
        mom_v = float(mom) if (mom is not None and not pd.isna(mom)) else 0.0

        vol_pen = 0.0
        if not pd.isna(row.get('Close')) and float(row['Close']) > 0:
            vol_pen = float(row['ATR']) / float(row['Close'])

        vol_ok = 0.0
        if (not pd.isna(row.get('VolSMA20'))) and (not pd.isna(row.get('VolSMA50'))):
            if float(row['VolSMA20']) > float(row['VolSMA50']):
                vol_ok = 1.0

        pivot = row.get('PivotClose')
        brk = 0.0
        if pivot is not None and not pd.isna(pivot):
            brk = (float(row['Close']) - float(pivot)) / float(row['ATR'])

        return (2.0 * mom_v) + (1.0 * brk) + (0.3 * vol_ok) - (2.0 * vol_pen)

    for i, date in enumerate(idx):
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

            if use_trailing:
                atr_raw = df.loc[date, 'ATR']
                if not pd.isna(atr_raw) and float(atr_raw) > 0:
                    new_stop = c - trail_mult * float(atr_raw)
                    p['stop'] = float(max(p['stop'], new_stop))

            exit_px = None
            reason = None
            if l <= p['stop']:
                exit_px = p['stop']
                reason = 'STOP'
            elif (p.get('tp') is not None) and (not pd.isna(p.get('tp'))) and (h >= p['tp']):
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

            initial_risk_per_share = float(p.get('initial_risk_per_share', np.nan))
            r_mult = np.nan
            if np.isfinite(initial_risk_per_share) and initial_risk_per_share > 0:
                r_mult = float((exit_px_eff - p['entry_px']) / initial_risk_per_share)

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
                    'setup': p.get('setup', 'BREAKOUT'),
                    'initial_risk_per_share': initial_risk_per_share,
                    'R_multiple': r_mult,
                }
            )

        open_positions = new_open

        eq_today = mark_to_market(date)
        equity_rows.append({'Date': date, 'Equity': eq_today, 'Cash': cash, 'OpenPositions': len(open_positions)})

        risk_on = bool(reg.loc[date, 'RiskOn']) if not pd.isna(reg.loc[date, 'RiskOn']) else False

        if weekly_rerank and risk_on and is_weekly_rebalance_day(date) and open_positions:
            scored = []
            for p in open_positions:
                sc = score_candidate(p['symbol'], date)
                scored.append((p, sc))
            scored.sort(key=lambda x: x[1], reverse=True)
            keep = scored[: int(cfg['max_positions'])]
            keep_syms = set([p['symbol'] for p, _ in keep])

            new_open = []
            for p in open_positions:
                if p['symbol'] in keep_syms:
                    new_open.append(p)
                    continue
                c = data[p['symbol']].loc[date, 'Close']
                if pd.isna(c):
                    new_open.append(p)
                    continue
                exit_px_eff = float(c) * (1 - spread)
                cash += p['shares'] * exit_px_eff

                initial_risk_per_share = float(p.get('initial_risk_per_share', np.nan))
                r_mult = np.nan
                if np.isfinite(initial_risk_per_share) and initial_risk_per_share > 0:
                    r_mult = float((exit_px_eff - p['entry_px']) / initial_risk_per_share)

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
                        'reason': 'RERANK',
                        'setup': p.get('setup', 'BREAKOUT'),
                        'initial_risk_per_share': initial_risk_per_share,
                        'R_multiple': r_mult,
                    }
                )
            open_positions = new_open

        if i >= len(idx) - 1:
            continue
        next_date = idx[i + 1]

        if hard_risk_on and (not risk_on):
            continue

        if len(open_positions) >= cfg['max_positions']:
            continue

        candidates = []
        if risk_on:
            for sym in symbols:
                row = data[sym].loc[date]
                if pd.isna(row.get('ATR')) or pd.isna(row.get('Close')):
                    continue
                if float(row['Close']) < cfg['min_price']:
                    continue
                dv = row.get('DollarVol')
                if (dv is not None) and (not pd.isna(dv)) and float(dv) < cfg['min_dollar_volume']:
                    continue

                setup = None
                piv = row.get('PivotClose')
                if piv is not None and (not pd.isna(piv)) and (float(row['Close']) > float(piv)):
                    setup = 'BREAKOUT'

                if bool(cfg.get('enable_cwh', True)) and bool(row.get('CWH_Signal', False)):
                    setup = 'CUP_HANDLE'

                if setup is None:
                    continue

                sc = score_candidate(sym, date)
                if setup == 'CUP_HANDLE' and bool(row.get('CWH_VolOK', False)):
                    sc += float(cfg.get('cwh_vol_bonus', 0.3))

                candidates.append((sym, sc, setup))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        n_new = min(cfg['max_new_trades_per_day'], cfg['max_positions'] - len(open_positions))
        picks = candidates[:n_new]

        for sym, _, setup in picks:
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
            if setup == 'CUP_HANDLE':
                hl = data[sym].loc[date].get('CWH_HandleLow')
                if hl is not None and not pd.isna(hl) and float(hl) > 0:
                    stop = min(stop, float(hl))

            tp = None
            if not use_trailing:
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
                    'stop': float(stop),
                    'tp': tp,
                    'setup': setup,
                    'initial_risk_per_share': float(entry_px_eff - float(stop)),
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

    stats = compute_trade_stats(trades_df)

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
        **stats,
    }

    breakdown = breakdown_tables(trades_df)

    return equity_df, trades_df, summary, breakdown
