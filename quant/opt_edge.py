"""Options edge engine — every model on the site, pointed at the option market.

The only durable retail edges in options come from ONE comparison:
    what the OPTIONS MARKET prices  vs  what OUR MODELS forecast.

  * Variance Risk Premium (VRP) — ATM IV minus forecast realized vol
    (GARCH + EWMA blend). IV persistently overprices RV (Carr & Wu 2009);
    when the gap is unusually wide, selling premium has tailwind; when IV
    is BELOW forecast, owning options is statistically cheap.
  * IV richness percentile — today's IV vs the ticker's own 1y realized-vol
    distribution. Rank matters more than level.
  * Model-vs-market expected move — our Monte Carlo cone vs the straddle.
  * Structure suggester — fuses the desk verdict (direction) with the VRP
    (rich/cheap vol) into a concrete structure, with strikes picked by delta
    from the live chain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(df: pd.DataFrame, window: int = 21) -> float:
    """Annualized realized vol (%) over the last `window` days."""
    r = df["Close"].pct_change().dropna().iloc[-window:]
    return float(r.std() * np.sqrt(252) * 100)


def iv_richness(df: pd.DataFrame, atm_iv: float) -> dict:
    """Percentile of ATM IV vs the ticker's own rolling 21d RV over 1y."""
    r = df["Close"].pct_change().dropna()
    rv = (r.rolling(21).std() * np.sqrt(252) * 100).dropna().iloc[-252:]
    if len(rv) < 60 or not atm_iv:
        return {}
    pct = float((rv < atm_iv).mean() * 100)
    return {"iv_pctile": round(pct),
            "rv_median": round(float(rv.median()), 1),
            "rv_now": round(float(rv.iloc[-1]), 1)}


def vrp(atm_iv: float, garch_annual: float | None,
        ewma_annual: float | None) -> dict:
    """Variance risk premium: IV minus model-forecast vol (GARCH/EWMA blend)."""
    fcs = [x for x in (garch_annual, ewma_annual) if x]
    if not fcs or not atm_iv:
        return {}
    forecast = float(np.mean(fcs))
    premium = atm_iv - forecast
    return {
        "iv": round(atm_iv, 1),
        "forecast_vol": round(forecast, 1),
        "vrp_pts": round(premium, 1),
        "state": ("💰 IV RICH — premium selling favored" if premium > 4 else
                  "🔥 IV CHEAP — owning options favored" if premium < -2 else
                  "⚖️ Fairly priced — no vol edge"),
    }


def move_vs_model(exp_move: float | None, mc_paths: np.ndarray | None,
                  spot: float, dte: int) -> dict:
    """Straddle expected move vs our Monte Carlo cone at the same horizon."""
    if not exp_move or mc_paths is None or mc_paths.size == 0:
        return {}
    h = min(max(dte, 1), mc_paths.shape[1] - 1)
    terminal = mc_paths[:, h]
    mc_move = float(np.percentile(np.abs(terminal - spot), 68))  # ~1σ
    ratio = exp_move / mc_move if mc_move > 0 else np.nan
    return {
        "market_move": round(exp_move, 2),
        "model_move": round(mc_move, 2),
        "ratio": round(float(ratio), 2),
        "read": ("options overprice the move" if ratio > 1.15 else
                 "options underprice the move" if ratio < 0.85 else
                 "market and model agree"),
    }


def _strike_by_delta(chain: pd.DataFrame, expiry: str, side: str,
                     target: float, spot: float, greeks_fn) -> float | None:
    sub = chain[(chain["expiry"] == expiry) & (chain["type"] == side)]
    if sub.empty:
        return None
    g = greeks_fn(spot, sub["strike"].values, sub["dte"].values / 365.0,
                  sub["iv"].values / 100.0,
                  np.full(len(sub), side == "C"))
    d = np.abs(np.abs(g["delta"].values) - target)
    return float(sub["strike"].values[int(np.argmin(d))])


def suggest_structure(direction: str, vol_state: str, chain: pd.DataFrame,
                      expiry: str, spot: float, greeks_fn) -> dict:
    """Fuse desk direction + vol edge into ONE concrete structure."""
    rich = "RICH" in vol_state
    cheap = "CHEAP" in vol_state

    def K(side, tgt):
        return _strike_by_delta(chain, expiry, side, tgt, spot, greeks_fn)

    if direction == "LONG" and rich:
        s, l = K("P", 0.30), K("P", 0.15)
        return {"name": "Bull put credit spread",
                "legs": f"SELL {expiry} {s}P / BUY {l}P",
                "logic": "Bullish view + rich IV → get PAID to be long. "
                         "Profits if price rises, chops, or falls slightly; "
                         "defined risk = strike width − credit."}
    if direction == "LONG" and cheap:
        b, s = K("C", 0.50), K("C", 0.25)
        return {"name": "Bull call debit spread",
                "legs": f"BUY {expiry} {b}C / SELL {s}C",
                "logic": "Bullish view + cheap IV → own the move at a "
                         "discount; the short call cuts theta bleed."}
    if direction == "SHORT" and rich:
        s, l = K("C", 0.30), K("C", 0.15)
        return {"name": "Bear call credit spread",
                "legs": f"SELL {expiry} {s}C / BUY {l}C",
                "logic": "Bearish view + rich IV → sell the overpriced "
                         "upside. Defined risk."}
    if direction == "SHORT" and cheap:
        b, s = K("P", 0.50), K("P", 0.25)
        return {"name": "Bear put debit spread",
                "legs": f"BUY {expiry} {b}P / SELL {s}P",
                "logic": "Bearish view + cheap IV → own downside "
                         "convexity cheaply."}
    if direction == "NO TRADE" and rich:
        cs, cl = K("C", 0.16), K("C", 0.08)
        ps, pl = K("P", 0.16), K("P", 0.08)
        return {"name": "Iron condor",
                "legs": f"SELL {ps}P/{cs}C · BUY {pl}P/{cl}C ({expiry})",
                "logic": "No directional edge + rich IV → harvest the "
                         "variance premium inside the expected range. "
                         "The classic 'market is overpaying for insurance' "
                         "trade."}
    return {"name": "Stand aside",
            "legs": "—",
            "logic": "No directional edge and no vol edge — an option "
                     "trade here is paying the market maker for "
                     "entertainment."}
