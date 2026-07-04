"""Backtester v2 — dual-strategy engine with institutional risk mechanics.

Strategies:
  TREND — composite-signal following. Entries gated by the 200-day SMA
          (Faber 2007). Chandelier 2.5×ATR trail, breakeven after +1R,
          time-stop on stalled trades.
  DIP   — Connors-style RSI(2) pullback buyer: short-term panic INSIDE an
          uptrend. High win rate, small wins, strict time exit.
  AUTO  — picks per ticker by Hurst exponent (trending vs mean-reverting).

Risk mechanics (applied to both):
  * next-bar-open execution (no look-ahead), commission per side
  * volatility-targeted sizing (Moreira & Muir 2017): risk scales down
    when ATR% is elevated vs its own history
  * breakeven stop once the trade is +1R
  * time stop: unprofitable after N bars -> out
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import atr, composite, rsi, sma


@dataclass
class BTConfig:
    starting_cash: float = 5000.0
    commission_pct: float = 0.001
    atr_stop_mult: float = 2.5
    risk_per_trade: float = 0.01
    mode: str = "auto"              # "auto" | "trend" | "dip"
    breakeven_r: float = 1.0        # move stop to entry after +1R
    time_stop_trend: int = 20       # bars; exit if unprofitable by then
    time_stop_dip: int = 10
    regime_filter: bool = True      # longs only above SMA200
    vol_target: bool = True         # inverse-vol position scaling


@dataclass
class BTResult:
    equity: pd.Series
    bh_equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    mode_used: str = "trend"


def _hurst_quick(close: pd.Series, max_lag: int = 80) -> float:
    p = np.log(close.dropna().values)
    if len(p) < max_lag * 2:
        max_lag = max(20, len(p) // 4)
    lags = range(2, max_lag)
    tau = np.maximum([np.std(p[l:] - p[:-l]) for l in lags], 1e-12)
    return float(np.clip(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0],
                         0.0, 1.0))


def _metrics(equity: pd.Series, trades: pd.DataFrame, bh: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    n_years = max(len(equity) / 252, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    scr = (trades["pnl"].abs() < trades["pnl"].abs().mean() * 0.1).sum() if len(trades) else 0
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / n_years) - 1
    pf = None
    if len(trades):
        gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_l = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        pf = round(float(gross_w / gross_l), 2) if gross_l > 0 else None
    return {
        "CAGR %": round(float(cagr) * 100, 1),
        "Buy&Hold CAGR %": round(float(bh_cagr) * 100, 1),
        "Sharpe": round(float(sharpe), 2),
        "Sortino": round(float(sortino), 2),
        "Max Drawdown %": round(float(dd) * 100, 1),
        "Trades": int(len(trades)),
        "Win Rate %": round(float(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
        "Profit Factor": pf,
        "Final Equity $": round(float(equity.iloc[-1]), 0),
        "Buy&Hold Final $": round(float(bh.iloc[-1]), 0),
    }


def run_backtest(df: pd.DataFrame, cfg: BTConfig = BTConfig()) -> BTResult:
    mode = cfg.mode
    if mode == "auto":
        h = _hurst_quick(df["Close"])
        mode = "trend" if h >= 0.5 else "dip"

    comp = composite(df) if mode == "trend" else None
    a = atr(df)
    s200 = sma(df["Close"], 200)
    r2 = rsi(df["Close"], 2)

    # volatility-target scaler: current ATR% vs its 1y median
    atr_pct = (a / df["Close"])
    med = atr_pct.rolling(252, min_periods=60).median()
    vt = (med / atr_pct).clip(0.5, 1.5).fillna(1.0) if cfg.vol_target else \
        pd.Series(1.0, index=df.index)

    cash = cfg.starting_cash
    shares = 0.0
    entry_price = stop = 0.0
    entry_i = 0
    be_armed = False
    equity_rows, trade_rows = [], []

    o_, h_, l_, c_ = (df[k].values for k in ("Open", "High", "Low", "Close"))
    sig = comp["signal"].values if comp is not None else None
    idx = df.index
    time_stop = cfg.time_stop_trend if mode == "trend" else cfg.time_stop_dip

    for i in range(1, len(df)):
        o, hi, lo, c = o_[i], h_[i], l_[i], c_[i]
        prev_atr = a.values[i - 1]
        above200 = c_[i - 1] > s200.values[i - 1] if not np.isnan(s200.values[i - 1]) else False

        # ---------------- exits ----------------
        if shares > 0:
            bars_in = i - entry_i
            r_dist = cfg.atr_stop_mult * a.values[entry_i - 1]

            # breakeven arming
            if not be_armed and hi >= entry_price + cfg.breakeven_r * r_dist:
                stop = max(stop, entry_price)
                be_armed = True
            # chandelier trail (trend only)
            if mode == "trend":
                stop = max(stop, hi - cfg.atr_stop_mult * prev_atr)

            exit_now, reason, exit_px = False, "", o
            if lo <= stop:
                exit_now, reason = True, "breakeven" if be_armed and stop <= entry_price * 1.001 else "stop"
                exit_px = min(o, stop) if o > stop else o
            elif mode == "trend" and sig[i - 1] == "SELL":
                exit_now, reason = True, "signal"
            elif mode == "dip" and r2.values[i - 1] > 65:
                exit_now, reason = True, "target(rsi)"
            elif bars_in >= time_stop and c_[i - 1] < entry_price:
                exit_now, reason = True, "time"

            if exit_now:
                exit_px = max(exit_px, 0.01)
                proceeds = shares * exit_px * (1 - cfg.commission_pct)
                pnl = proceeds - shares * entry_price * (1 + cfg.commission_pct)
                trade_rows.append({"entry_date": idx[entry_i], "exit_date": idx[i],
                                   "entry": round(entry_price, 2),
                                   "exit": round(exit_px, 2),
                                   "pnl": round(pnl, 2), "reason": reason})
                cash += proceeds
                shares = 0.0
                be_armed = False

        # ---------------- entries ----------------
        if shares == 0 and prev_atr > 0:
            gate = above200 if cfg.regime_filter else True
            enter = False
            if mode == "trend":
                enter = gate and sig[i - 1] == "BUY"
            else:  # dip
                enter = gate and r2.values[i - 1] < 10
            if enter:
                stop_dist = cfg.atr_stop_mult * prev_atr
                risk_dollars = cash * cfg.risk_per_trade * vt.values[i - 1]
                size_by_risk = risk_dollars / stop_dist
                max_by_cash = cash / (o * (1 + cfg.commission_pct))
                shares = float(min(size_by_risk, max_by_cash))
                if shares * o < 100:
                    shares = 0.0
                else:
                    cash -= shares * o * (1 + cfg.commission_pct)
                    entry_price = o
                    entry_i = i
                    stop = o - stop_dist
                    be_armed = False

        equity_rows.append(cash + shares * c)

    equity = pd.Series(equity_rows, index=idx[1:], name="strategy")
    bh = pd.Series(cfg.starting_cash / c_[0] * c_[1:], index=idx[1:],
                   name="buy_hold")
    trades = pd.DataFrame(trade_rows)
    return BTResult(equity, bh, trades, _metrics(equity, trades, bh), mode)


def walk_forward(df: pd.DataFrame, cfg: BTConfig = BTConfig(),
                 n_folds: int = 4) -> pd.DataFrame:
    fold_len = len(df) // n_folds
    rows = []
    for k in range(n_folds):
        chunk = df.iloc[k * fold_len:(k + 1) * fold_len + 1]
        if len(chunk) < 120:
            continue
        res = run_backtest(chunk, cfg)
        row = {"fold": k + 1, "start": chunk.index[0].date(),
               "end": chunk.index[-1].date(), "mode": res.mode_used}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
