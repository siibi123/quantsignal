"""BX Lab — B-Xtrender, upgraded from paint to probabilities.

Two testing modules:
  1. PARAMETER SWEEP — tests B-X parameter presets on the FIRST 70% of
     history, ranks by out-of-sample Sharpe on the LAST 30%. Finds the
     tuning that actually fits the ticker, validated honestly.
  2. STATE PROBABILITY TABLE — every bar classified into one of 8 B-X
     states; for each state: the historical probability the next N bars
     close higher, average forward return, t-stat and sample size.
     This turns "green means go" into calibrated statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bxtrender import ema, rsi_wilder, t3

PRESETS = [
    (5, 20, 15),   # original Puppytherapy
    (3, 15, 10),   # fast
    (5, 15, 10),
    (8, 25, 15),   # slow
    (5, 30, 20),
    (4, 12, 8),    # scalper
    (10, 30, 15),
    (6, 18, 12),
]


def _bx_core(close: pd.Series, s1: int, s2: int, s3: int):
    osc = rsi_wilder(ema(close, s1) - ema(close, s2), s3) - 50
    sig = t3(osc, 5)
    return osc, sig


def parameter_sweep(df: pd.DataFrame, train_frac: float = 0.7,
                    bars_per_year: int = 252) -> pd.DataFrame:
    """Simple BX strategy (long when osc>0 & T3 rising) per preset.

    Trained implicitly (no fitting beyond the preset), reported ONLY on the
    out-of-sample segment — presets that only worked in-sample get exposed.
    """
    close = df["Close"]
    rets = close.pct_change().shift(-1).fillna(0)
    split = int(len(df) * train_frac)
    rows = []
    for s1, s2, s3 in PRESETS:
        osc, sig = _bx_core(close, s1, s2, s3)
        pos = ((osc > 0) & (sig > sig.shift(1))).astype(int)
        strat = pos * rets
        for label, seg in (("IS", slice(220, split)),
                           ("OOS", slice(split, len(df) - 1))):
            r = strat.iloc[seg]
            sh = float(r.mean() / r.std() * np.sqrt(bars_per_year)) \
                if r.std() > 0 else 0.0
            if label == "IS":
                is_sh = sh
            else:
                rows.append({
                    "preset": f"{s1}/{s2}/{s3}",
                    "IS Sharpe": round(is_sh, 2),
                    "OOS Sharpe": round(sh, 2),
                    "OOS ret %": round(float((1 + r).prod() - 1) * 100, 1),
                    "exposure %": round(float(pos.iloc[seg].mean()) * 100),
                    "overfit gap": round(is_sh - sh, 2),
                })
    out = pd.DataFrame(rows).sort_values("OOS Sharpe", ascending=False)
    return out.reset_index(drop=True)


BX_STATES = [
    "Long+ · T3↑ · Short+", "Long+ · T3↑ · Short−",
    "Long+ · T3↓ · Short+", "Long+ · T3↓ · Short−",
    "Long− · T3↑ · Short+", "Long− · T3↑ · Short−",
    "Long− · T3↓ · Short+", "Long− · T3↓ · Short−",
]


def state_probabilities(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """P(price higher in `horizon` bars | current B-X state), with t-stats."""
    from .bxtrender import bxtrender
    bx = bxtrender(df)
    close = df["Close"]
    fwd = close.shift(-horizon) / close - 1

    state = ((bx["long_osc"] <= 0).astype(int) * 4 +
             (~bx["t3_rising"]).astype(int) * 2 +
             (bx["short_osc"] <= 0).astype(int))
    valid = fwd.notna() & state.notna()
    rows = []
    for s in range(8):
        mask = valid & (state == s)
        n = int(mask.sum())
        if n < 10:
            rows.append({"state": BX_STATES[s], "n": n, "P(up) %": None,
                         "avg fwd %": None, "t-stat": None, "signal": "—"})
            continue
        f = fwd[mask]
        p_up = float((f > 0).mean() * 100)
        avg = float(f.mean() * 100)
        t = float(f.mean() / (f.std() / np.sqrt(n))) if f.std() > 0 else 0
        sig_lbl = ("🟢 edge long" if t > 1.5 else
                   "🔴 edge short/avoid" if t < -1.5 else "⚪ noise")
        rows.append({"state": BX_STATES[s], "n": n,
                     "P(up) %": round(p_up, 1),
                     "avg fwd %": round(avg, 2),
                     "t-stat": round(t, 2), "signal": sig_lbl})
    tab = pd.DataFrame(rows)
    cur = int(state.iloc[-1]) if not np.isnan(state.iloc[-1]) else None
    tab.attrs["current_state"] = BX_STATES[cur] if cur is not None else "—"
    return tab
