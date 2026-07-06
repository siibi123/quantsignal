"""Portfolio lab — optimization via PyPortfolioOpt (from awesome-quant).

Three allocators, three philosophies:
  MAX SHARPE — the classic Markowitz tangency portfolio (needs return
               estimates, which are noisy — handle with care).
  MIN VOL    — ignores returns entirely; just the quietest mix.
  HRP        — Hierarchical Risk Parity (Lopez de Prado 2016): clusters
               assets by correlation and splits risk down the tree. No
               matrix inversion, no return estimates — the robust choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_prices(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {t: df["Close"] for t, df in data.items() if len(df) > 200}
    px = pd.DataFrame(cols).dropna()
    return px


def optimize(px: pd.DataFrame, account: float = 5000.0) -> dict:
    """Run all three optimizers + efficient frontier + $ allocation."""
    from pypfopt import (DiscreteAllocation, EfficientFrontier, HRPOpt,
                         expected_returns, risk_models)

    if px.shape[1] < 3:
        return {"error": "Need at least 3 tickers with shared history."}

    mu = expected_returns.mean_historical_return(px)
    S = risk_models.CovarianceShrinkage(px).ledoit_wolf()

    out: dict = {"tickers": list(px.columns)}

    # --- Max Sharpe -----------------------------------------------------
    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.35))
        ef.max_sharpe(risk_free_rate=0.045)
        w_ms = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=0.045)
        out["max_sharpe"] = {"weights": w_ms,
                             "ret": round(perf[0] * 100, 1),
                             "vol": round(perf[1] * 100, 1),
                             "sharpe": round(perf[2], 2)}
    except Exception as exc:
        out["max_sharpe"] = {"error": str(exc)}

    # --- Min Vol ----------------------------------------------------------
    try:
        ef2 = EfficientFrontier(mu, S, weight_bounds=(0, 0.35))
        ef2.min_volatility()
        w_mv = ef2.clean_weights()
        perf2 = ef2.portfolio_performance(risk_free_rate=0.045)
        out["min_vol"] = {"weights": w_mv,
                          "ret": round(perf2[0] * 100, 1),
                          "vol": round(perf2[1] * 100, 1),
                          "sharpe": round(perf2[2], 2)}
    except Exception as exc:
        out["min_vol"] = {"error": str(exc)}

    # --- HRP ----------------------------------------------------------------
    try:
        rets = px.pct_change().dropna()
        hrp = HRPOpt(rets)
        w_h = hrp.optimize()
        perf3 = hrp.portfolio_performance(risk_free_rate=0.045)
        out["hrp"] = {"weights": {k: round(v, 4) for k, v in w_h.items()},
                      "ret": round(perf3[0] * 100, 1),
                      "vol": round(perf3[1] * 100, 1),
                      "sharpe": round(perf3[2], 2)}
    except Exception as exc:
        out["hrp"] = {"error": str(exc)}

    # --- Efficient frontier points -------------------------------------------
    try:
        pts = []
        for tv in np.linspace(float(np.sqrt(np.diag(S)).min()) * 1.01,
                              float(np.sqrt(np.diag(S)).max()) * 0.99, 18):
            try:
                efp = EfficientFrontier(mu, S, weight_bounds=(0, 0.35))
                efp.efficient_risk(tv)
                r_, v_, _ = efp.portfolio_performance()
                pts.append((v_ * 100, r_ * 100))
            except Exception:
                continue
        out["frontier"] = pts
        out["assets"] = [(float(np.sqrt(S.loc[t, t])) * 100,
                          float(mu[t]) * 100, t) for t in px.columns]
    except Exception:
        out["frontier"] = []

    # --- Discrete allocation for the account (HRP weights, robust default) ---
    try:
        w_use = out.get("hrp", {}).get("weights") or out.get(
            "max_sharpe", {}).get("weights")
        if w_use:
            latest = px.iloc[-1]
            da = DiscreteAllocation(w_use, latest, total_portfolio_value=account)
            alloc, leftover = da.greedy_portfolio()
            out["allocation"] = {"shares": alloc,
                                 "leftover": round(float(leftover), 0)}
    except Exception as exc:
        out["allocation"] = {"error": str(exc)}

    return out
