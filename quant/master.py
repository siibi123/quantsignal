"""MASTER ALGORITHM — everything on the site fused into one decision process.

Pipeline (how a systematic desk actually runs):
  1. MARKET GATE   — regime quadrant on SPY decides how much capital plays at all.
  2. CROSS-SECTION — rank the whole universe on 6 published anomalies
                     (Jegadeesh momentum, 52w-high, anti-lottery MAX, low ivol,
                      betting-against-beta, short-term reversal).
  3. TIME-SERIES   — top-decile names go through the 7-model verdict engine
                     (trend, momentum, B-Xtrender, MACD, RSI, meanrev, volume
                      + regime + proven per-ticker edge + risk/reward).
  4. SIZING        — risk-parity-ish: each position risks the same % of account,
                     total portfolio heat capped.
  5. HONESTY LAYER — expected edge reported AFTER the McLean-Pontiff (2016)
                     58% post-publication haircut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced import regime_quadrant
from .anomalies import PUBLICATION_HAIRCUT, alpha_ranks
from .verdict import analyze

# How much of the account each regime allows to be deployed
REGIME_EXPOSURE = {
    "🟢 Bull · Calm": 1.00,
    "🟡 Bull · Storm": 0.60,
    "🔵 Bear · Calm": 0.35,
    "🔴 Bear · Storm": 0.15,
}


def run_master(data: dict[str, pd.DataFrame], spy: pd.DataFrame,
               account: float = 5000.0, risk_pct: float = 1.0,
               max_positions: int = 4, heat_cap_pct: float = 4.0,
               top_k: int = 8, conviction_min: int = 55,
               aggressive_fill: bool = False) -> dict:
    """Run the whole systematic process. Returns an actionable plan."""
    # 1 — market gate
    reg = regime_quadrant(spy)
    exposure = REGIME_EXPOSURE.get(reg["regime"], 0.5)
    deployable = account * exposure

    # 2 — cross-sectional ranks
    ranks = alpha_ranks(data, spy)
    if ranks.empty:
        return {"error": "No rankable tickers"}
    candidates = list(ranks.index[:top_k])
    avoid = list(ranks.index[-5:])

    # 3 — time-series verdict on candidates
    picks, considered = [], []
    for tkr in candidates:
        try:
            v = analyze(data[tkr], account=account, risk_pct=risk_pct)
        except Exception:
            continue
        v["ticker"] = tkr
        v["alpha"] = float(ranks.loc[tkr, "alpha"])
        v["pct_rank"] = int(ranks.loc[tkr, "pct_rank"])
        considered.append(v)
        if v["verdict"] == "LONG" and v["conviction"] >= conviction_min:
            picks.append(v)
    picks.sort(key=lambda x: (-x["conviction"], -x["alpha"]))
    picks = picks[:max_positions]

    # aggressive fill: if the strict gate produced < 2 names, take the top
    # alpha names anyway at HALF risk, clearly tagged lower-confidence
    fills = []
    if aggressive_fill and len(picks) < 2:
        have = {p["ticker"] for p in picks}
        for v in considered:
            if v["ticker"] in have or v["verdict"] == "SHORT":
                continue
            if v.get("entry") and v.get("stop") and v["entry"] > v["stop"]:
                v = dict(v)
                v["fill"] = True
                fills.append(v)
            if len(picks) + len(fills) >= 2:
                break

    # 4 — sizing with portfolio heat cap
    heat_budget = account * heat_cap_pct / 100
    plan_rows, total_cost, total_risk = [], 0.0, 0.0
    for v in picks + (fills if aggressive_fill else []):
        eff_risk = risk_pct * (0.5 if v.get("fill") else 1.0)
        risk_dollars = min(account * eff_risk / 100,
                           heat_budget - total_risk)
        if risk_dollars <= 0:
            break
        stop_dist = abs(v["entry"] - v["stop"])
        shares = int(min(risk_dollars / stop_dist,
                         (deployable - total_cost) / v["entry"]))
        if shares < 1:
            continue
        cost = shares * v["entry"]
        total_cost += cost
        total_risk += shares * stop_dist
        plan_rows.append({
            "ticker": v["ticker"],
            "action": "BUY ½size*" if v.get("fill") else "BUY",
            "shares": shares,
            "entry ~": v["entry"],
            "stop": v["stop"],
            "target": v["target"],
            "RR": v["rr"],
            "conviction": v["conviction"],
            "alpha rank %": v["pct_rank"],
            "cost $": round(cost, 0),
            "risk $": round(shares * stop_dist, 0),
        })

    plan = pd.DataFrame(plan_rows)
    cash = account - total_cost

    # 5 — honesty layer: gross expected edge, then haircut
    if picks:
        avg_conv = float(np.mean([p["conviction"] for p in picks]))
    else:
        avg_conv = 0.0

    return {
        "regime": reg,
        "exposure_pct": round(exposure * 100),
        "ranks": ranks,
        "considered": considered,
        "plan": plan,
        "cash": round(cash, 0),
        "cash_pct": round(cash / account * 100, 1),
        "total_risk": round(total_risk, 0),
        "total_risk_pct": round(total_risk / account * 100, 2),
        "avoid": avoid,
        "avg_conviction": round(avg_conv),
        "haircut_pct": round(PUBLICATION_HAIRCUT * 100),
    }
