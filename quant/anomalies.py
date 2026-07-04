"""Cross-sectional anomalies — published, replicated factors from the literature.

Implemented signals (paper, journal, finding):
  MOM   Jegadeesh & Titman (1993, JoF)   12-1 momentum: winners keep winning
  STREV Jegadeesh (1990, JoF)            1-month short-term reversal
  H52   George & Hwang (2004, JoF)       proximity to 52-week high (anchoring)
  MAX   Bali, Cakici & Whitelaw (2011, JFE) lottery stocks underperform
  IVOL  Ang, Hodrick, Xing & Zhang (2006, JoF) low idiosyncratic vol outperforms
  BAB   Frazzini & Pedersen (2014, JFE)  low beta outperforms per unit of risk

Context (why we trust these): Jensen, Kelly & Pedersen (2023, JoF) show most
published factors replicate and work out-of-sample globally. McLean & Pontiff
(2016, JoF) show returns decay ~26% out-of-sample and ~58% post-publication —
so we report expectations WITH that haircut applied.

Cross-sectional = each stock is scored RELATIVE to the rest of the universe
(z-scored ranks), which is how these anomalies are defined in the literature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PUBLICATION_HAIRCUT = 0.58   # McLean & Pontiff (2016) post-publication decay

ANOMALY_INFO = {
    "mom_12_1": ("Momentum 12-1", "Jegadeesh & Titman 1993",
                 "Return months t-12..t-1 excluding the last month. Winners keep winning."),
    "strev_1m": ("Short-term reversal", "Jegadeesh 1990",
                 "Last month's sharp losers bounce, sharp winners cool off."),
    "high_52w": ("52-week high", "George & Hwang 2004",
                 "Price near its 52w high keeps working (anchoring bias)."),
    "max_lottery": ("Anti-lottery (MAX)", "Bali, Cakici & Whitelaw 2011",
                    "Stocks with recent huge single-day pops (lottery tickets) underperform."),
    "low_ivol": ("Low idiosyncratic vol", "Ang, Hodrick, Xing & Zhang 2006",
                 "Boring stocks beat exciting ones after adjusting for beta."),
    "low_beta": ("Betting against beta", "Frazzini & Pedersen 2014",
                 "Low-beta stocks outperform per unit of risk."),
}

WEIGHTS = {
    "mom_12_1": 0.25,
    "high_52w": 0.20,
    "max_lottery": 0.20,   # Bali et al. 2017: MAX subsumes much of BAB
    "low_ivol": 0.15,
    "low_beta": 0.10,
    "strev_1m": 0.10,
}


def _beta_ivol(df: pd.DataFrame, spy: pd.DataFrame,
               window: int = 252) -> tuple[float, float]:
    """OLS beta vs SPY and idiosyncratic (residual) daily vol."""
    r = df["Close"].pct_change().dropna()
    m = spy["Close"].pct_change().dropna()
    joined = pd.concat([r, m], axis=1, join="inner").dropna().iloc[-window:]
    if len(joined) < 60:
        return np.nan, np.nan
    y, x = joined.iloc[:, 0].values, joined.iloc[:, 1].values
    beta = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 0 else np.nan
    resid = y - beta * x
    return beta, float(np.std(resid))


def raw_signals(data: dict[str, pd.DataFrame],
                spy: pd.DataFrame) -> pd.DataFrame:
    """Compute raw anomaly characteristics for every ticker."""
    rows = {}
    for tkr, df in data.items():
        c = df["Close"]
        if len(c) < 260:
            continue
        mom = float(c.iloc[-21] / c.iloc[-252] - 1)          # 12-1
        strev = -float(c.iloc[-1] / c.iloc[-21] - 1)         # reversal: minus 1m ret
        h52 = float(c.iloc[-1] / df["High"].iloc[-252:].max())
        daily = c.pct_change().iloc[-21:]
        mx = -float(daily.nlargest(5).mean())                # anti-lottery
        beta, ivol = _beta_ivol(df, spy)
        rows[tkr] = {
            "mom_12_1": mom,
            "strev_1m": strev,
            "high_52w": h52,
            "max_lottery": mx,
            "low_ivol": -ivol if not np.isnan(ivol) else np.nan,
            "low_beta": -beta if not np.isnan(beta) else np.nan,
            "beta": round(beta, 2) if not np.isnan(beta) else None,
        }
    return pd.DataFrame(rows).T


def alpha_ranks(data: dict[str, pd.DataFrame],
                spy: pd.DataFrame) -> pd.DataFrame:
    """Z-score each signal ACROSS the universe and combine into alpha score.

    Returns table sorted by alpha (best first) with percentile rank.
    """
    raw = raw_signals(data, spy)
    if raw.empty:
        return raw
    z = pd.DataFrame(index=raw.index)
    for col in WEIGHTS:
        s = raw[col].astype(float)
        z[col] = ((s - s.mean()) / s.std(ddof=0)).clip(-3, 3)
    z = z.fillna(0)
    alpha = sum(z[c] * w for c, w in WEIGHTS.items())
    out = z.round(2)
    out["alpha"] = alpha.round(3)
    out["pct_rank"] = (alpha.rank(pct=True) * 100).round(0).astype(int)
    out["beta"] = raw["beta"]
    return out.sort_values("alpha", ascending=False)
