"""RL Lab — TradeMaster-inspired reinforcement learning, sized for the web.

Three ideas adapted from TradeMaster (NTU, NeurIPS 2023):
  1. RL trading agent  — tabular Q-learning over a discretized market state
                         (trend x B-Xtrender x RSI x volatility). Trained on
                         the FIRST 70% of history, evaluated ONLY on the
                         unseen last 30% — the anti-"untrustworthy FinRL
                         results" rule TradeMaster was built around.
  2. Market Dynamics Modeling — label every period into one of five market
                         styles, like TradeMaster's MDM module.
  3. PRUDEX-lite       — multi-axis evaluation (profitability, risk control,
                         consistency, efficiency, exposure) as a radar,
                         inspired by PRUDEX-Compass (TMLR 2023).

Pure numpy — trains in under a second, honest by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bxtrender import bxtrender
from .signals import atr, rsi, sma

N_STATES = 12            # 2 trend x 2 bx x 3 rsi
ACTIONS = (0, 1)         # 0 = flat, 1 = long


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _states(df: pd.DataFrame) -> np.ndarray:
    c = df["Close"]
    trend = (c > sma(c, 200)).astype(int).values
    bx = (bxtrender(df)["long_osc"] > 0).astype(int).values
    r = rsi(c, 14).values
    rsi_b = np.digitize(r, [30, 70])                     # 0,1,2
    return trend * 6 + bx * 3 + rsi_b


STATE_LABELS = [
    f"{t} | BX{b} | RSI {r}"
    for t in ("Downtrend", "Uptrend") for b in ("−", "+")
    for r in ("<30", "30-70", ">70")
]


# ---------------------------------------------------------------------------
# Contextual-bandit learner (the honest form of RL for this problem)
# ---------------------------------------------------------------------------

def train_agent(df: pd.DataFrame, train_frac: float = 0.7,
                shrink_k: float = 40.0, hurdle: float = 0.00015,
                switch_cost: float = 0.0002, **_) -> dict:
    """Estimate E[next-day return | state] on the first 70% of history with
    shrinkage toward the global mean; act only where the evidence clears a
    hurdle. Evaluated ONLY on the untouched last 30% (TradeMaster's
    anti-untrustworthy-results rule).

    Why a bandit and not deep Q-learning: our position does not move the
    market, so there is no state transition to control — estimating the
    conditional mean IS the optimal policy, and it doesn't hallucinate
    structure the data can't support.
    """
    if len(df) < 400:
        return {"error": "Need at least ~400 bars of history."}

    states = _states(df)
    rets = df["Close"].pct_change().shift(-1).fillna(0).values
    split = int(len(df) * train_frac)
    warm = 220
    tr = np.arange(warm, split)
    te = np.arange(split, len(df) - 1)

    g_mean = float(np.mean(rets[tr]))
    mu = np.zeros(N_STATES)
    n = np.zeros(N_STATES)
    sd = np.zeros(N_STATES)
    for s in range(N_STATES):
        mask = states[tr] == s
        n[s] = mask.sum()
        if n[s] > 2:
            mu[s] = float(np.mean(rets[tr][mask]))
            sd[s] = float(np.std(rets[tr][mask]))
    mu_shrunk = (n * mu + shrink_k * g_mean) / (n + shrink_k)
    policy_long = mu_shrunk > hurdle
    tstat = np.where(n > 2, (mu - 0.0) / (sd / np.sqrt(np.maximum(n, 1)) + 1e-12), 0.0)

    Q = np.column_stack([np.zeros(N_STATES), mu_shrunk])   # for display

    def walk(idx_range):
        eq, bh = [1.0], [1.0]
        pos = 0; switches = 0
        for i in idx_range:
            a = int(policy_long[states[i]])
            if a != pos:
                switches += 1
            r_ = a * rets[i] - switch_cost * abs(a - pos)
            eq.append(eq[-1] * (1 + r_))
            bh.append(bh[-1] * (1 + rets[i]))
            pos = a
        dates = df.index[list(idx_range)[0]:list(idx_range)[-1] + 2]
        return (pd.Series(eq, index=dates[:len(eq)]),
                pd.Series(bh, index=dates[:len(bh)]), switches)

    eq_te, bh_te, switches = walk(te)

    def _stats(eq: pd.Series) -> dict:
        r_ = eq.pct_change().dropna()
        n_years = max(len(eq) / 252, 1e-9)
        sharpe = float(r_.mean() / r_.std() * np.sqrt(252)) if r_.std() > 0 else 0.0
        return {"CAGR %": round(float((eq.iloc[-1]) ** (1 / n_years) - 1) * 100, 1),
                "Sharpe": round(sharpe, 2),
                "Max DD %": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
                "Final multiple": round(float(eq.iloc[-1]), 3)}

    exposure = float(np.mean([int(policy_long[states[i]]) for i in te]))
    cur_s = int(states[-1])

    pol = pd.DataFrame({
        "state": STATE_LABELS,
        "action": ["LONG" if policy_long[s] else "FLAT" for s in range(N_STATES)],
        "E[next-day ret] bps": np.round(mu_shrunk * 1e4, 1),
        "t-stat": np.round(tstat, 2),
        "train samples": n.astype(int),
    })

    return {
        "Q": Q,
        "policy": pol,
        "oos_equity": eq_te,
        "oos_bh": bh_te,
        "oos_stats": _stats(eq_te),
        "bh_stats": _stats(bh_te),
        "oos_exposure_pct": round(exposure * 100),
        "oos_switches": switches,
        "split_date": df.index[split].date(),
        "current_state": STATE_LABELS[cur_s],
        "current_action": "LONG" if policy_long[cur_s] else "FLAT",
        "current_confidence": round(abs(float(mu_shrunk[cur_s])) * 1e4, 1),
    }


# ---------------------------------------------------------------------------
# Market Dynamics Modeling (5 styles)
# ---------------------------------------------------------------------------

MDM_STYLES = ["🚀 Strong bull", "📈 Bull", "😴 Sideways", "📉 Bear", "🌪️ Crash/volatile"]
MDM_COLORS = ["#10b981", "#6ee7b7", "#8b98a5", "#f59e0b", "#ef4444"]


def market_dynamics(df: pd.DataFrame, win: int = 21) -> pd.DataFrame:
    """Label each bar with a market style from rolling return & volatility."""
    c = df["Close"]
    ret = c.pct_change(win)
    vol = c.pct_change().rolling(win).std() * np.sqrt(252)
    vol_hi = vol.rolling(252, min_periods=60).quantile(0.8)

    style = np.select(
        [ (vol > vol_hi) & (ret < 0),
          ret > 0.08,
          ret > 0.02,
          ret < -0.04 ],
        [4, 0, 1, 3], default=2)
    return pd.DataFrame({"style": style, "label": [MDM_STYLES[s] for s in style]},
                        index=df.index)


# ---------------------------------------------------------------------------
# PRUDEX-lite scoring (0-100 per axis)
# ---------------------------------------------------------------------------

def prudex_scores(eq: pd.Series, trades_per_year: float | None = None,
                  exposure_pct: float | None = None) -> dict:
    r = eq.pct_change().dropna()
    n_years = max(len(eq) / 252, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    roll = r.rolling(63)
    rs = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
    consistency = float((rs > 0).mean()) if len(rs) else 0.5

    return {
        "Profitability": float(np.clip(cagr / 0.30, 0, 1) * 100),
        "Risk control": float(np.clip(1 + dd / 0.40, 0, 1) * 100),
        "Sharpe quality": float(np.clip(sharpe / 2.0, 0, 1) * 100),
        "Consistency": round(consistency * 100, 0),
        "Capital efficiency": float(np.clip((exposure_pct or 100) / 100, 0, 1)
                                    * np.clip(sharpe / 1.5, 0, 1) * 100),
    }
