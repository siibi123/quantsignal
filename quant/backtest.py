"""Backtester — event-driven long-only simulation of the composite signal.

Rules (deliberately simple and honest):
- Signals computed on bar t are executed at the NEXT bar's open (no look-ahead).
- Long when signal == BUY, flat when signal == SELL. HOLD keeps current state.
- Commission + slippage charged per side.
- ATR trailing stop exits a position independently of the signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import atr, composite


@dataclass
class BTConfig:
    starting_cash: float = 5000.0
    commission_pct: float = 0.001    # 0.1% per side (Blink-style broker + slippage)
    atr_stop_mult: float = 2.5       # trailing stop distance in ATRs
    risk_per_trade: float = 0.01     # 1% of equity risked per trade


@dataclass
class BTResult:
    equity: pd.Series
    bh_equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def _metrics(equity: pd.Series, trades: pd.DataFrame, bh: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    n_years = max(len(equity) / 252, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / n_years) - 1
    return {
        "CAGR %": round(float(cagr) * 100, 1),
        "Buy&Hold CAGR %": round(float(bh_cagr) * 100, 1),
        "Sharpe": round(float(sharpe), 2),
        "Sortino": round(float(sortino), 2),
        "Max Drawdown %": round(float(dd) * 100, 1),
        "Trades": int(len(trades)),
        "Win Rate %": round(float(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
        "Final Equity $": round(float(equity.iloc[-1]), 0),
        "Buy&Hold Final $": round(float(bh.iloc[-1]), 0),
    }


def run_backtest(df: pd.DataFrame, cfg: BTConfig = BTConfig()) -> BTResult:
    comp = composite(df)
    a = atr(df)

    cash = cfg.starting_cash
    shares = 0.0
    entry_price = 0.0
    trail_stop = 0.0
    equity_rows, trade_rows = [], []

    opens = df["Open"].values
    highs = df["High"].values
    closes = df["Close"].values
    signals = comp["signal"].values
    atrs = a.values
    idx = df.index

    for i in range(1, len(df)):
        o, h, c = opens[i], highs[i], closes[i]
        prev_sig = signals[i - 1]          # decide on yesterday's info
        prev_atr = atrs[i - 1]

        # --- exits first -------------------------------------------------
        if shares > 0:
            trail_stop = max(trail_stop, h - cfg.atr_stop_mult * prev_atr)
            stop_hit = df["Low"].values[i] <= trail_stop
            if prev_sig == "SELL" or stop_hit:
                exit_px = min(o, trail_stop) if stop_hit and o > trail_stop else o
                exit_px = max(exit_px, 0.01)
                proceeds = shares * exit_px * (1 - cfg.commission_pct)
                pnl = proceeds - shares * entry_price * (1 + cfg.commission_pct)
                trade_rows.append({
                    "entry_date": entry_date, "exit_date": idx[i],
                    "entry": round(entry_price, 2), "exit": round(exit_px, 2),
                    "pnl": round(pnl, 2),
                    "reason": "stop" if stop_hit else "signal",
                })
                cash += proceeds
                shares = 0.0

        # --- entries ------------------------------------------------------
        if shares == 0 and prev_sig == "BUY" and prev_atr > 0:
            stop_dist = cfg.atr_stop_mult * prev_atr
            # Risk-based size: lose at most (risk_per_trade × equity) if stopped out.
            size_by_risk = (cash * cfg.risk_per_trade) / stop_dist
            max_by_cash = cash / (o * (1 + cfg.commission_pct))
            shares = float(min(size_by_risk, max_by_cash))
            if shares * o < 100:            # ignore dust trades
                shares = 0.0
            else:
                cost = shares * o * (1 + cfg.commission_pct)
                cash -= cost
                entry_price = o
                entry_date = idx[i]
                trail_stop = o - stop_dist

        equity_rows.append(cash + shares * c)

    equity = pd.Series(equity_rows, index=idx[1:], name="strategy")
    bh_shares = cfg.starting_cash / closes[0]
    bh = pd.Series(bh_shares * closes[1:], index=idx[1:], name="buy_hold")
    trades = pd.DataFrame(trade_rows)
    return BTResult(equity, bh, trades, _metrics(equity, trades, bh))


def walk_forward(df: pd.DataFrame, cfg: BTConfig = BTConfig(),
                 n_folds: int = 4) -> pd.DataFrame:
    """Split history into sequential folds and report per-fold metrics.
    If the strategy only works in one period, that's a red flag."""
    fold_len = len(df) // n_folds
    rows = []
    for k in range(n_folds):
        chunk = df.iloc[k * fold_len:(k + 1) * fold_len + 1]
        if len(chunk) < 120:
            continue
        res = run_backtest(chunk, cfg)
        row = {"fold": k + 1,
               "start": chunk.index[0].date(),
               "end": chunk.index[-1].date()}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
