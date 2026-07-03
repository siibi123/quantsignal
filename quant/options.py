"""Options analytics — IV surface, smile, term structure, Greeks.

Data: yfinance option chains (free, delayed). Greeks: Black-Scholes closed
form in pure numpy — no extra dependencies.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

RISK_FREE = 0.045  # annualised; fine for surface visualisation purposes


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (vectorised, pure numpy)
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def bs_greeks(S: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray,
              is_call: np.ndarray, r: float = RISK_FREE) -> pd.DataFrame:
    """Vectorised BS Greeks. T in years, iv as decimal (0.30 = 30%)."""
    K = np.asarray(K, float)
    T = np.clip(np.asarray(T, float), 1e-6, None)
    iv = np.clip(np.asarray(iv, float), 1e-4, None)
    is_call = np.asarray(is_call, bool)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT

    delta = np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    gamma = _norm_pdf(d1) / (S * iv * sqrtT)
    vega = S * _norm_pdf(d1) * sqrtT / 100.0                # per 1 vol point
    theta_call = (-S * _norm_pdf(d1) * iv / (2 * sqrtT)
                  - r * K * np.exp(-r * T) * _norm_cdf(d2)) / 365.0
    theta_put = (-S * _norm_pdf(d1) * iv / (2 * sqrtT)
                 + r * K * np.exp(-r * T) * _norm_cdf(-d2)) / 365.0
    theta = np.where(is_call, theta_call, theta_put)

    return pd.DataFrame({
        "delta": np.round(delta, 4),
        "gamma": np.round(gamma, 6),
        "vega": np.round(vega, 4),
        "theta": np.round(theta, 4),
    })


# ---------------------------------------------------------------------------
# Chain fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_chains(ticker: str, max_expiries: int = 8) -> tuple[float, pd.DataFrame]:
    """Return (spot, tidy chain df) for the nearest `max_expiries` expiries.

    Columns: expiry (str), dte (int), strike, iv (%), type (C/P),
             volume, oi, bid, ask, last, moneyness.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        return 0.0, pd.DataFrame()
    spot = float(hist["Close"].iloc[-1])

    expiries = list(t.options)[:max_expiries]
    today = dt.date.today()
    rows = []
    for exp in expiries:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        dte = max((dt.date.fromisoformat(exp) - today).days, 1)
        for df_side, side in ((chain.calls, "C"), (chain.puts, "P")):
            if df_side is None or df_side.empty:
                continue
            sub = df_side[["strike", "impliedVolatility", "volume",
                           "openInterest", "bid", "ask", "lastPrice"]].copy()
            sub.columns = ["strike", "iv", "volume", "oi", "bid", "ask", "last"]
            sub["expiry"], sub["dte"], sub["type"] = exp, dte, side
            rows.append(sub)

    if not rows:
        return spot, pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce") * 100.0   # -> %
    df["moneyness"] = df["strike"] / spot
    # Clean junk quotes: dead IVs, deep wings, no market
    df = df[(df["iv"].between(1.0, 300.0))
            & (df["moneyness"].between(0.5, 1.6))
            & ((df["oi"].fillna(0) > 0) | (df["volume"].fillna(0) > 0))]
    return spot, df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Surface construction
# ---------------------------------------------------------------------------

def build_surface(df: pd.DataFrame, n_strikes: int = 40
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid (strikes, dtes, iv_matrix) for a 3D plot.

    For each expiry we take the OTM side (puts below spot, calls above —
    the liquid side market-makers actually quote), median-aggregate per
    strike, then interpolate onto a common strike grid.
    """
    if df.empty:
        return np.array([]), np.array([]), np.array([])

    # OTM filter: puts with moneyness<=1, calls with moneyness>=1
    otm = df[((df["type"] == "P") & (df["moneyness"] <= 1.0)) |
             ((df["type"] == "C") & (df["moneyness"] >= 1.0))]
    if otm.empty:
        otm = df

    k_lo, k_hi = otm["strike"].quantile(0.02), otm["strike"].quantile(0.98)
    strikes = np.linspace(k_lo, k_hi, n_strikes)
    dtes = np.array(sorted(otm["dte"].unique()))

    grid = np.full((len(dtes), n_strikes), np.nan)
    for i, d in enumerate(dtes):
        sl = (otm[otm["dte"] == d].groupby("strike")["iv"].median()
              .sort_index())
        if len(sl) < 4:
            continue
        grid[i] = np.interp(strikes, sl.index.values, sl.values,
                            left=np.nan, right=np.nan)

    # Drop empty expiry rows, forward/back-fill small gaps along strikes
    keep = ~np.isnan(grid).all(axis=1)
    grid, dtes = grid[keep], dtes[keep]
    grid = pd.DataFrame(grid).interpolate(axis=1, limit=5,
                                          limit_direction="both").values
    return strikes, dtes, grid


def atm_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Median IV of near-the-money quotes (0.97–1.03 moneyness) per expiry."""
    atm = df[df["moneyness"].between(0.97, 1.03)]
    ts = (atm.groupby(["expiry", "dte"])["iv"].median()
          .reset_index().sort_values("dte"))
    return ts


def skew_25(df: pd.DataFrame) -> float | None:
    """Simple skew proxy: IV(95% put) − IV(105% call) on the nearest expiry."""
    if df.empty:
        return None
    d0 = df["dte"].min()
    near = df[df["dte"] == d0]
    put_wing = near[(near["type"] == "P")
                    & near["moneyness"].between(0.93, 0.97)]["iv"].median()
    call_wing = near[(near["type"] == "C")
                     & near["moneyness"].between(1.03, 1.07)]["iv"].median()
    if pd.isna(put_wing) or pd.isna(call_wing):
        return None
    return round(float(put_wing - call_wing), 2)
