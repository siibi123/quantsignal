"""GARCH(1,1) volatility forecast (arch) + pairs trading (statsmodels).

Both straight from the awesome-quant toolbox.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def garch_forecast(df: pd.DataFrame, horizon: int = 5) -> dict:
    """Fit GARCH(1,1) on daily returns; forecast next-day & 5-day vol."""
    from arch import arch_model

    rets = 100 * df["Close"].pct_change().dropna()
    if len(rets) < 250:
        return {}
    try:
        am = arch_model(rets.iloc[-750:], vol="GARCH", p=1, q=1,
                        mean="Constant", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=horizon, reindex=False)
        var_path = fc.variance.values[0]
        sig1 = float(np.sqrt(var_path[0]))                 # % daily
        sig5 = float(np.sqrt(var_path.mean()))
        price = float(df["Close"].iloc[-1])
        persistence = float(res.params.get("alpha[1]", 0) +
                            res.params.get("beta[1]", 0))
        return {
            "sigma1d_pct": round(sig1, 2),
            "sigma_annual_pct": round(sig1 * np.sqrt(252), 1),
            "move_1d": round(price * sig1 / 100, 2),
            "sigma5d_avg_pct": round(sig5, 2),
            "persistence": round(persistence, 3),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pairs trading
# ---------------------------------------------------------------------------

def pairs_analysis(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   z_window: int = 60) -> dict:
    """Engle-Granger cointegration + hedge ratio + spread z-score."""
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint

    a = df_a["Close"]
    b = df_b["Close"]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j.columns = ["A", "B"]
    if len(j) < 250:
        return {"error": "Need at least ~1y of overlapping history."}

    la, lb = np.log(j["A"]), np.log(j["B"])
    _, pvalue, _ = coint(la, lb)

    X = sm.add_constant(lb)
    ols = sm.OLS(la, X).fit()
    hedge = float(ols.params.iloc[1])
    spread = la - hedge * lb
    z = (spread - spread.rolling(z_window).mean()) / \
        spread.rolling(z_window).std()
    z = z.dropna()
    cur_z = float(z.iloc[-1])

    if pvalue > 0.10:
        signal = "❌ Not cointegrated — this is not a tradeable pair"
    elif cur_z > 2:
        signal = "🔻 Spread rich: SHORT A / LONG B (bet on convergence)"
    elif cur_z < -2:
        signal = "🔺 Spread cheap: LONG A / SHORT B (bet on convergence)"
    elif abs(cur_z) < 0.5:
        signal = "🎯 Spread at fair value — exit zone / no entry"
    else:
        signal = "⏳ Inside the bands — wait for |z| ≥ 2"

    half_life = None
    try:
        ds = spread.diff().dropna()
        lag = spread.shift(1).dropna().loc[ds.index]
        beta = float(sm.OLS(ds, sm.add_constant(lag)).fit().params.iloc[1])
        if beta < 0:
            half_life = round(float(-np.log(2) / beta), 1)
    except Exception:
        pass

    return {
        "pvalue": round(float(pvalue), 4),
        "cointegrated": pvalue <= 0.05,
        "borderline": 0.05 < pvalue <= 0.10,
        "hedge_ratio": round(hedge, 3),
        "z": round(cur_z, 2),
        "z_series": z,
        "spread": spread,
        "half_life_days": half_life,
        "signal": signal,
    }
