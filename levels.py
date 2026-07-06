"""Whale flow — dealer gamma exposure (GEX) & unusual options activity.

The in-house version of what flow services (unusual whales / Cheddar Flow /
Quant Data style) sell, computed from the same public option-chain data.

Conventions (standard retail GEX model):
  * Dealers are assumed long calls sold to them? No — the common convention:
    dealers are SHORT puts and LONG calls hedges aside, the practical retail
    model treats call gamma as positive GEX and put gamma as negative GEX.
  * Dollar GEX per strike = gamma * OI * 100 (contract size) * S^2 * 0.01
    (i.e., dollars of delta-hedging per 1% move in the underlying).
  * Positive net GEX  -> dealers dampen moves (buy dips, sell rips) = pinning.
  * Negative net GEX  -> dealers amplify moves = volatility fuel.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RISK_FREE = 0.045


def _gamma(S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray) -> np.ndarray:
    K = np.asarray(K, float)
    T = np.clip(np.asarray(T, float), 1e-6, None)
    iv = np.clip(np.asarray(iv, float), 1e-4, None)
    d1 = (np.log(S / K) + (RISK_FREE + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    pdf = np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (S * iv * np.sqrt(T))


def gex_profile(chain: pd.DataFrame, spot: float,
                max_expiries: int = 4) -> pd.DataFrame:
    """Net dollar GEX per strike (calls +, puts −), nearest expiries."""
    if chain.empty:
        return pd.DataFrame()
    exps = sorted(chain["expiry"].unique())[:max_expiries]
    sub = chain[chain["expiry"].isin(exps)].copy()
    sub = sub[sub["oi"].fillna(0) > 0]
    if sub.empty:
        return pd.DataFrame()

    g = _gamma(spot, sub["strike"].values, sub["dte"].values / 365.0,
               sub["iv"].values / 100.0)
    sign = np.where(sub["type"].values == "C", 1.0, -1.0)
    sub["gex"] = g * sub["oi"].fillna(0).values * 100 * spot ** 2 * 0.01 * sign

    prof = (sub.groupby("strike")["gex"].sum().reset_index()
            .sort_values("strike"))
    prof["gex_m"] = prof["gex"] / 1e6           # in $ millions
    return prof


def gex_summary(prof: pd.DataFrame, spot: float) -> dict:
    """Net GEX, call wall, put wall, and the zero-gamma flip level."""
    if prof.empty:
        return {}
    net = float(prof["gex"].sum())
    call_wall = float(prof.loc[prof["gex"].idxmax(), "strike"])
    put_wall = float(prof.loc[prof["gex"].idxmin(), "strike"])

    # Flip point: where cumulative GEX (from low strikes up) crosses zero
    cum = prof["gex"].cumsum().values
    strikes = prof["strike"].values
    flip = None
    sgn = np.sign(cum)
    for i in range(1, len(cum)):
        if sgn[i] != sgn[i - 1] and sgn[i] != 0:
            flip = float(strikes[i])
            break

    return {
        "net_gex_m": round(net / 1e6, 1),
        "regime": "🧲 Pinning (dealers dampen moves)" if net > 0
                  else "⛽ Vol fuel (dealers amplify moves)",
        "call_wall": call_wall,
        "put_wall": put_wall,
        "flip": flip,
        "spot_vs_flip": (None if flip is None else
                         "above (stable zone)" if spot > flip
                         else "below (unstable zone)"),
    }


def unusual_flow(chain: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Rank contracts by opening-activity signature: volume >> open interest.

    vol/OI > 1 means more contracts traded TODAY than existed before —
    someone is opening fresh positions. Premium estimates the money behind it.
    """
    if chain.empty:
        return pd.DataFrame()
    f = chain.copy()
    f["volume"] = f["volume"].fillna(0)
    f["oi"] = f["oi"].fillna(0)
    f = f[f["volume"] >= 100]
    if f.empty:
        return pd.DataFrame()

    mid = np.where((f["bid"] > 0) & (f["ask"] > 0),
                   (f["bid"] + f["ask"]) / 2, f["last"].fillna(0))
    f["premium_$"] = (f["volume"] * mid * 100).round(0)
    f["vol/oi"] = (f["volume"] / f["oi"].replace(0, np.nan)).round(2)
    f["vol/oi"] = f["vol/oi"].fillna(np.inf)
    f["signature"] = np.where(f["vol/oi"] >= 1.0, "🔥 opening",
                              np.where(f["vol/oi"] >= 0.5, "warm", ""))
    f = f.sort_values(["premium_$"], ascending=False).head(top_n)
    cols = ["type", "strike", "expiry", "volume", "oi", "vol/oi",
            "premium_$", "iv", "signature"]
    return f[cols].reset_index(drop=True)
