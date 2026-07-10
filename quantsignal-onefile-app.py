"""QuantSignal v26 — SINGLE-FILE build (29 modules incl. AutoTrader).
Upload this ONE file as app.py + requirements.txt. No folders needed."""
from __future__ import annotations

import datetime
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime as _datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy import stats

# ===== quant/data.py =====
"""Data layer — OHLCV from Yahoo Finance with caching.

Price convention (important for correctness):
  auto_adjust=False  -> Close is the REAL traded price, matching your broker
                        and TradingView. This is what we chart, set stops/
                        targets on, and quote. 'Adj Close' is kept separately
                        for return calculations that need dividend adjustment.
"""


import time

import pandas as pd
import streamlit as st
import yfinance as yf


def _with_retry(fn, tries: int = 3, base_sleep: float = 2.0):
    """Run fn(); on rate-limit/network errors back off and retry."""
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(base_sleep * (k + 1))
    return None

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD", "CRM",
    "JPM", "BAC", "GS", "V", "MA", "CAT", "GE", "XOM", "CVX", "COP",
    "UNH", "LLY", "JNJ", "PG", "KO", "COST", "WMT", "MCD", "NKE", "DIS",
    "TSM", "INTC", "MU", "QCOM", "ORCL", "ADBE", "NOW", "PLTR", "SMCI", "PANW",
    "SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "SMH", "GLD", "TLT",
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV. Close = real traded price (matches broker/TradingView).

    'AdjClose' kept as an extra column for dividend-adjusted return math.
    """
    df = _with_retry(lambda: yf.Ticker(ticker).history(
        period=period, interval=interval, auto_adjust=False))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)          # Open/High/Low/Close/Adj Close/Volume
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        df["AdjClose"] = df["Adj Close"]
        cols = cols + ["AdjClose"]
    return df[cols].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_many(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Batched download: the whole universe in ONE Yahoo request instead of
    50 — the single biggest rate-limit saver on the site."""
    out: dict[str, pd.DataFrame] = {}
    raw = _with_retry(lambda: yf.download(
        list(tickers), period=period, interval="1d", auto_adjust=False,
        group_by="ticker", threads=False, progress=False))
    if raw is not None and not raw.empty:
        for t in tickers:
            try:
                df = raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) \
                    else raw.dropna()
                df = df.rename(columns=str.title)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) >= 60:
                    out[t] = df
            except Exception:
                continue
        if out:
            return out
    # fallback: polite sequential
    for t in tickers:
        try:
            df = fetch_history(t, period=period)
            if len(df) >= 60:
                out[t] = df
            time.sleep(0.15)
        except Exception:
            continue
    return out


# ===== quant/signals.py =====
"""Signal engine — technical indicators combined into a composite quant score.

Every indicator produces a sub-score in [-1, +1].
The composite is a weighted average, mapped to BUY / HOLD / SELL.
All indicators use only past data (no look-ahead).
"""


import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger_z(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - m) / sd.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Sub-scores (each in [-1, +1])
# ---------------------------------------------------------------------------

def score_trend(df: pd.DataFrame) -> pd.Series:
    """Trend: price vs SMA50/SMA200 + golden/death cross direction."""
    c = df["Close"]
    s50, s200 = sma(c, 50), sma(c, 200)
    above50 = np.sign(c - s50)
    above200 = np.sign(c - s200)
    cross = np.sign(s50 - s200)
    return pd.Series((above50 + above200 + cross) / 3.0, index=df.index)


def score_momentum(df: pd.DataFrame) -> pd.Series:
    """12-1 momentum (classic academic factor), squashed with tanh."""
    c = df["Close"]
    mom = c.shift(21) / c.shift(252) - 1.0  # skip last month, look at prior 11
    return np.tanh(mom * 3.0).fillna(0)


def score_rsi(df: pd.DataFrame) -> pd.Series:
    """RSI regime: >50 bullish, <50 bearish; extremes fade slightly."""
    r = rsi(df["Close"])
    base = (r - 50) / 50.0                      # -1..+1
    fade = np.where(r > 75, -0.3, np.where(r < 25, 0.3, 0.0))
    return (base + fade).clip(-1, 1)


def score_macd(df: pd.DataFrame) -> pd.Series:
    """MACD histogram sign, normalised by price."""
    _, _, hist = macd(df["Close"])
    norm = hist / df["Close"] * 100
    return np.tanh(norm * 2.0).fillna(0)


def score_meanrev(df: pd.DataFrame) -> pd.Series:
    """Mean reversion: fade extreme Bollinger z-scores."""
    z = bollinger_z(df["Close"])
    return (-z / 2.5).clip(-1, 1).fillna(0)


def score_volume(df: pd.DataFrame) -> pd.Series:
    """Volume confirmation: surge in direction of the daily move."""
    v_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    direction = np.sign(df["Close"].pct_change())
    conf = np.where(v_ratio > 1.5, direction * 0.8,
                    np.where(v_ratio > 1.0, direction * 0.3, 0.0))
    return pd.Series(conf, index=df.index).fillna(0)


def vol_regime(df: pd.DataFrame) -> pd.Series:
    """Volatility regime filter: 1 = calm (trade), 0.5 = elevated, 0.25 = storm."""
    a = atr(df) / df["Close"]
    pct = a.rolling(252, min_periods=60).rank(pct=True)
    return pd.Series(np.where(pct > 0.9, 0.25, np.where(pct > 0.7, 0.5, 1.0)),
                     index=df.index)


WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "bxtrender": 0.15,
    "macd": 0.125,
    "rsi": 0.10,
    "meanrev": 0.10,
    "volume": 0.075,
}

BUY_TH, SELL_TH = 0.25, -0.25


def composite(df: pd.DataFrame) -> pd.DataFrame:
    """Return df of sub-scores + composite score + signal label per bar."""

    parts = {
        "trend": score_trend(df),
        "momentum": score_momentum(df),
        "bxtrender": score_bx(df),
        "macd": score_macd(df),
        "rsi": score_rsi(df),
        "meanrev": score_meanrev(df),
        "volume": score_volume(df),
    }
    out = pd.DataFrame(parts, index=df.index)
    raw = sum(out[k] * w for k, w in WEIGHTS.items())
    out["score"] = raw * vol_regime(df)          # dampen in vol storms
    out["signal"] = np.where(out["score"] >= BUY_TH, "BUY",
                     np.where(out["score"] <= SELL_TH, "SELL", "HOLD"))
    return out


def latest_snapshot(df: pd.DataFrame) -> dict:
    """Latest composite reading for the screener table."""
    comp = composite(df)
    last = comp.iloc[-1]
    a = atr(df).iloc[-1]
    price = df["Close"].iloc[-1]
    return {
        "price": round(float(price), 2),
        "score": round(float(last["score"]), 3),
        "signal": str(last["signal"]),
        "trend": round(float(last["trend"]), 2),
        "momentum": round(float(last["momentum"]), 2),
        "bxtrender": round(float(last["bxtrender"]), 2),
        "macd": round(float(last["macd"]), 2),
        "rsi_score": round(float(last["rsi"]), 2),
        "meanrev": round(float(last["meanrev"]), 2),
        "atr": round(float(a), 2),
        "ret_1m": round(float(df["Close"].pct_change(21).iloc[-1] * 100), 1),
        "ret_3m": round(float(df["Close"].pct_change(63).iloc[-1] * 100), 1)
        if len(df) > 63 else None,
        "off_52w_high": round(float(price / df["High"].rolling(min(252, len(df))).max().iloc[-1] - 1) * 100, 1),
    }


# ===== quant/bxtrender.py =====
"""B-Xtrender — institutional-grade port & upgrade.

Original: Bharat Jhunjhunwala (IFTA Journal), Pine port by @Puppytherapy.
  short osc = RSI( EMA(close,5) - EMA(close,20), 15 ) - 50
  long  osc = RSI( EMA(close,20), 15 ) - 50
  signal    = Tillson T3(short osc, 5, b=0.7)

Upgrades here:
  * exact Wilder RSI (TradingView-faithful)
  * turn signals on the T3 line (local trough/peak flips)
  * regular divergence detection (price vs oscillator pivots)
  * multi-timeframe confluence (weekly long-term oscillator)
  * event study — real forward returns after each historical signal
"""


import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Building blocks (TradingView-faithful)
# ---------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(s: pd.Series, n: int) -> pd.Series:
    """TradingView rsi(): Wilder RMA smoothing of gains/losses."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def t3(s: pd.Series, n: int, b: float = 0.7) -> pd.Series:
    """Tillson T3 — six cascaded EMAs with volume-factor coefficients."""
    e1 = ema(s, n); e2 = ema(e1, n); e3 = ema(e2, n)
    e4 = ema(e3, n); e5 = ema(e4, n); e6 = ema(e5, n)
    c1 = -b ** 3
    c2 = 3 * b ** 2 + 3 * b ** 3
    c3 = -6 * b ** 2 - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ---------------------------------------------------------------------------
# Core oscillators
# ---------------------------------------------------------------------------

def bxtrender(df: pd.DataFrame, s1: int = 5, s2: int = 20, s3: int = 15,
              l1: int = 20, l2: int = 15) -> pd.DataFrame:
    """Return short/long oscillators, T3 signal line and turn markers."""
    close = df["Close"]
    short_osc = rsi_wilder(ema(close, s1) - ema(close, s2), s3) - 50
    long_osc = rsi_wilder(ema(close, l1), l2) - 50
    sig = t3(short_osc, 5)

    up_turn = (sig > sig.shift(1)) & (sig.shift(1) < sig.shift(2))
    dn_turn = (sig < sig.shift(1)) & (sig.shift(1) > sig.shift(2))

    return pd.DataFrame({
        "short_osc": short_osc,
        "long_osc": long_osc,
        "t3": sig,
        "t3_rising": sig > sig.shift(1),
        "buy_turn": up_turn,
        "sell_turn": dn_turn,
    }, index=df.index)


def score_bx(df: pd.DataFrame) -> pd.Series:
    """Sub-score in [-1, +1] for the composite model.

    Direction from the long oscillator, timing from the short osc + T3 slope.
    """
    bx = bxtrender(df)
    long_dir = np.tanh(bx["long_osc"] / 25.0)
    short_dir = np.tanh(bx["short_osc"] / 25.0)
    slope = np.where(bx["t3_rising"], 0.3, -0.3)
    raw = 0.45 * long_dir + 0.35 * short_dir + slope
    return pd.Series(raw, index=df.index).clip(-1, 1).fillna(0)


# ---------------------------------------------------------------------------
# Institutional upgrades
# ---------------------------------------------------------------------------

def _pivots(s: pd.Series, k: int = 3) -> tuple[list[int], list[int]]:
    """Indices of confirmed local highs and lows (k bars each side)."""
    v = s.values
    highs, lows = [], []
    for i in range(k, len(v) - k):
        win = v[i - k:i + k + 1]
        if v[i] == win.max() and (win.argmax() == k):
            highs.append(i)
        if v[i] == win.min() and (win.argmin() == k):
            lows.append(i)
    return highs, lows


def detect_divergence(df: pd.DataFrame, lookback: int = 120) -> dict:
    """Regular divergences between price and the short oscillator.

    Bearish: price higher high, oscillator lower high.
    Bullish: price lower low, oscillator higher low.
    """
    bx = bxtrender(df)
    price = df["Close"].iloc[-lookback:]
    osc = bx["short_osc"].iloc[-lookback:]

    p_hi, p_lo = _pivots(price, k=3)
    out = {"bearish": False, "bullish": False, "detail": ""}

    if len(p_hi) >= 2:
        i1, i2 = p_hi[-2], p_hi[-1]
        if (price.iloc[i2] > price.iloc[i1]
                and osc.iloc[i2] < osc.iloc[i1] and osc.iloc[i1] > 0):
            out["bearish"] = True
            out["detail"] = (f"Price HH {price.iloc[i1]:.2f}→{price.iloc[i2]:.2f} "
                             f"but oscillator LH {osc.iloc[i1]:.1f}→{osc.iloc[i2]:.1f}")
    if len(p_lo) >= 2:
        i1, i2 = p_lo[-2], p_lo[-1]
        if (price.iloc[i2] < price.iloc[i1]
                and osc.iloc[i2] > osc.iloc[i1] and osc.iloc[i1] < 0):
            out["bullish"] = True
            out["detail"] = (f"Price LL {price.iloc[i1]:.2f}→{price.iloc[i2]:.2f} "
                             f"but oscillator HL {osc.iloc[i1]:.1f}→{osc.iloc[i2]:.1f}")
    return out


def weekly_alignment(df: pd.DataFrame) -> dict:
    """Compute the long-term oscillator on WEEKLY bars — the boss timeframe."""
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna()
    if len(w) < 40:
        return {"weekly_osc": None, "weekly_rising": None}
    bx_w = bxtrender(w)
    osc = float(bx_w["long_osc"].iloc[-1])
    rising = bool(bx_w["long_osc"].iloc[-1] > bx_w["long_osc"].iloc[-2])
    return {"weekly_osc": round(osc, 1), "weekly_rising": rising}


def event_study(df: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """What ACTUALLY happened after every historical turn signal on this ticker."""
    bx = bxtrender(df)
    close = df["Close"]
    rows = []
    for name, mask in (("Buy turns", bx["buy_turn"]),
                       ("Sell turns", bx["sell_turn"])):
        idx = np.where(mask.values)[0]
        idx = idx[idx > 50]                       # skip warm-up
        row = {"signal": name, "count": len(idx)}
        for hzn in horizons:
            valid = idx[idx + hzn < len(close)]
            if len(valid) == 0:
                row[f"avg {hzn}d %"] = None
                row[f"win {hzn}d %"] = None
                continue
            fwd = close.values[valid + hzn] / close.values[valid] - 1.0
            row[f"avg {hzn}d %"] = round(float(fwd.mean()) * 100, 2)
            row[f"win {hzn}d %"] = round(float((fwd > 0).mean()) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ===== quant/levels.py =====
"""Price-structure analytics — Fibonacci retracements & Hurst exponent."""


import numpy as np
import pandas as pd

FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fib_levels(df: pd.DataFrame, lookback: int = 126) -> dict:
    """Auto-detect the dominant swing in `lookback` bars and return fib levels.

    If the swing low came before the swing high -> uptrend swing, retracements
    measured down from the high. Otherwise downtrend swing, measured up.
    """
    win = df.iloc[-lookback:]
    hi_pos = int(np.argmax(win["High"].values))
    lo_pos = int(np.argmin(win["Low"].values))
    hi = float(win["High"].iloc[hi_pos])
    lo = float(win["Low"].iloc[lo_pos])
    up_swing = lo_pos < hi_pos                      # low first, then high

    levels = {}
    rng = hi - lo
    for r in FIB_RATIOS:
        price = hi - rng * r if up_swing else lo + rng * r
        levels[f"{r:.3f}".rstrip("0").rstrip(".") or "0"] = round(price, 2)

    return {
        "up_swing": up_swing,
        "swing_high": round(hi, 2),
        "swing_low": round(lo, 2),
        "high_date": win.index[hi_pos],
        "low_date": win.index[lo_pos],
        "levels": levels,
    }


def hurst(df: pd.DataFrame, max_lag: int = 100) -> float:
    """Hurst exponent via rescaled variance of lagged differences.

    H > 0.5 -> trending (momentum works), H < 0.5 -> mean-reverting
    (fade extremes), H ~ 0.5 -> random walk (no memory).
    """
    prices = np.log(df["Close"].dropna().values)
    if len(prices) < max_lag * 2:
        max_lag = max(20, len(prices) // 4)
    lags = range(2, max_lag)
    tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]
    tau = np.maximum(tau, 1e-12)
    h = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return round(float(np.clip(h, 0.0, 1.0)), 3)


# ===== quant/montecarlo.py =====
"""Monte Carlo engine — simulate thousands of possible futures.

Geometric Brownian Motion calibrated to the ticker's own history
(EWMA-weighted drift & volatility so recent behaviour matters more).
"""


import numpy as np
import pandas as pd


def calibrate(df: pd.DataFrame, lookback: int = 252) -> tuple[float, float]:
    """Annualised (mu, sigma) from log returns, recent-weighted."""
    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().iloc[-lookback:]
    w = np.exp(np.linspace(-1.0, 0.0, len(rets)))          # recent = heavier
    w /= w.sum()
    mu_d = float((rets * w).sum())
    var_d = float((w * (rets - mu_d) ** 2).sum())
    return mu_d * 252, np.sqrt(var_d * 252)


def simulate(df: pd.DataFrame, days: int = 30, n_paths: int = 2000,
             seed: int = 42) -> np.ndarray:
    """Return (n_paths, days+1) matrix of simulated prices, col 0 = spot."""
    mu, sigma = calibrate(df)
    s0 = float(df["Close"].iloc[-1])
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, days))
    steps = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = s0 * np.exp(np.cumsum(steps, axis=1))
    return np.hstack([np.full((n_paths, 1), s0), paths])


def cone(paths: np.ndarray,
         pcts=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
    """Percentile bands across time for the probability cone chart."""
    return {p: np.percentile(paths, p, axis=0) for p in pcts}


def trade_odds(paths: np.ndarray, entry: float, stop: float, target: float,
               direction: int) -> dict:
    """First-touch simulation: does each path hit target or stop first?

    direction: +1 long, -1 short.
    """
    if direction >= 0:
        hit_t = paths >= target
        hit_s = paths <= stop
    else:
        hit_t = paths <= target
        hit_s = paths >= stop

    def first_true(m):
        idx = np.argmax(m, axis=1)
        idx[~m.any(axis=1)] = m.shape[1] + 1     # never touched
        return idx

    t_idx, s_idx = first_true(hit_t), first_true(hit_s)
    p_target = float((t_idx < s_idx).mean())
    p_stop = float((s_idx < t_idx).mean())
    p_neither = max(0.0, 1.0 - p_target - p_stop)

    # Terminal P&L distribution (per share, direction-adjusted)
    pnl = (paths[:, -1] - entry) * direction
    var95 = float(np.percentile(pnl, 5))
    cvar95 = float(pnl[pnl <= var95].mean()) if (pnl <= var95).any() else var95

    return {
        "p_target_first": round(p_target * 100, 1),
        "p_stop_first": round(p_stop * 100, 1),
        "p_neither": round(p_neither * 100, 1),
        "p_profit_end": round(float((pnl > 0).mean()) * 100, 1),
        "exp_pnl_share": round(float(pnl.mean()), 2),
        "var95_share": round(var95, 2),
        "cvar95_share": round(cvar95, 2),
    }


# ===== quant/options.py =====
"""Options analytics — IV surface, smile, term structure, Greeks.

Data: yfinance option chains (free, delayed). Greeks: Black-Scholes closed
form in pure numpy — no extra dependencies.
"""


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
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return 0.0, pd.DataFrame()
        spot = float(hist["Close"].iloc[-1])
        expiries = list(t.options)[:max_expiries]
    except Exception:
        return 0.0, pd.DataFrame()
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


def put_call_ratio(df: pd.DataFrame) -> float | None:
    """Total put OI / call OI across loaded expiries."""
    if df.empty:
        return None
    p = df[df["type"] == "P"]["oi"].fillna(0).sum()
    c = df[df["type"] == "C"]["oi"].fillna(0).sum()
    return round(float(p / c), 2) if c > 0 else None


def max_pain(df: pd.DataFrame, expiry: str) -> float | None:
    """Strike where total option-holder payout is minimised at expiry."""
    sub = df[df["expiry"] == expiry]
    if sub.empty:
        return None
    strikes = np.sort(sub["strike"].unique())
    calls = sub[sub["type"] == "C"].groupby("strike")["oi"].sum()
    puts = sub[sub["type"] == "P"].groupby("strike")["oi"].sum()
    pain = []
    for s in strikes:
        call_pay = sum(max(s - k, 0) * calls.get(k, 0) for k in calls.index)
        put_pay = sum(max(k - s, 0) * puts.get(k, 0) for k in puts.index)
        pain.append(call_pay + put_pay)
    return round(float(strikes[int(np.argmin(pain))]), 2)


# ===== quant/advanced.py =====
"""Advanced desk analytics — EWMA vol forecast, Kelly sizing, S/R, regimes."""


import numpy as np
import pandas as pd


def ewma_vol(df: pd.DataFrame, lam: float = 0.94) -> dict:
    """RiskMetrics EWMA volatility forecast (lambda = 0.94, daily).

    Returns tomorrow's expected daily move and annualised vol.
    """
    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    var = rets[0] ** 2
    for r in rets[1:]:
        var = lam * var + (1 - lam) * r ** 2
    sigma_d = float(np.sqrt(var))
    price = float(df["Close"].iloc[-1])
    return {
        "sigma_daily_pct": round(sigma_d * 100, 2),
        "sigma_annual_pct": round(sigma_d * np.sqrt(252) * 100, 1),
        "expected_move_1d": round(price * sigma_d, 2),
    }


def kelly(p_win: float, rr: float) -> dict:
    """Kelly fraction for a binary bet: f* = p − (1−p)/b."""
    p = min(max(p_win, 0.0), 1.0)
    b = max(rr, 1e-9)
    f = p - (1 - p) / b
    return {
        "kelly_pct": round(f * 100, 1),
        "half_kelly_pct": round(f * 50, 1),
        "edge_positive": f > 0,
    }


def support_resistance(df: pd.DataFrame, lookback: int = 252,
                       k: int = 5, n_levels: int = 4,
                       tol: float = 0.015) -> list[dict]:
    """Cluster swing pivots into the most-touched support/resistance zones."""
    win = df.iloc[-lookback:]
    highs, lows = [], []
    hv, lv = win["High"].values, win["Low"].values
    for i in range(k, len(win) - k):
        if hv[i] == hv[i - k:i + k + 1].max():
            highs.append(hv[i])
        if lv[i] == lv[i - k:i + k + 1].min():
            lows.append(lv[i])
    pivots = sorted(highs + lows)
    if not pivots:
        return []

    clusters: list[list[float]] = []
    for p in pivots:
        if clusters and p <= clusters[-1][-1] * (1 + tol):
            clusters[-1].append(p)
        else:
            clusters.append([p])

    price = float(df["Close"].iloc[-1])
    levels = [{"price": round(float(np.mean(c)), 2), "touches": len(c),
               "kind": "support" if np.mean(c) < price else "resistance"}
              for c in clusters if len(c) >= 2]
    levels.sort(key=lambda x: -x["touches"])
    return levels[:n_levels]


def regime_quadrant(df: pd.DataFrame) -> dict:
    """Classify the current market regime: trend x volatility quadrant."""
    c = df["Close"]
    sma200 = c.rolling(200).mean()
    bull = bool(c.iloc[-1] > sma200.iloc[-1])
    rets = c.pct_change().dropna()
    vol_now = float(rets.iloc[-21:].std() * np.sqrt(252))
    vol_hist = float(rets.rolling(21).std().dropna().quantile(0.7) * np.sqrt(252))
    calm = vol_now < vol_hist
    name = ("🟢 Bull · Calm" if bull and calm else
            "🟡 Bull · Storm" if bull else
            "🔵 Bear · Calm" if calm else
            "🔴 Bear · Storm")
    playbook = {
        "🟢 Bull · Calm": "Best regime for longs — trend signals shine, full size allowed.",
        "🟡 Bull · Storm": "Uptrend but violent — halve size, widen stops, expect shakeouts.",
        "🔵 Bear · Calm": "Quiet downtrend — shorts/cash; long signals need extra proof.",
        "🔴 Bear · Storm": "Crash conditions — capital preservation mode. Most edges die here.",
    }[name]
    return {"regime": name, "playbook": playbook,
            "vol_now_pct": round(vol_now * 100, 1)}


# ===== quant/anomalies.py =====
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

ANOM_WEIGHTS = {
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
    for col in ANOM_WEIGHTS:
        s = raw[col].astype(float)
        z[col] = ((s - s.mean()) / s.std(ddof=0)).clip(-3, 3)
    z = z.fillna(0)
    alpha = sum(z[c] * w for c, w in ANOM_WEIGHTS.items())
    out = z.round(2)
    out["alpha"] = alpha.round(3)
    out["pct_rank"] = (alpha.rank(pct=True) * 100).round(0).astype(int)
    out["beta"] = raw["beta"]
    return out.sort_values("alpha", ascending=False)


# ===== quant/verdict.py =====
"""Verdict engine — turns all signals into ONE trading decision.

Philosophy (how an actual quant desk thinks):
1. Signal strength alone is not enough — models must AGREE.
2. A signal that never worked on this ticker historically deserves no trust.
3. High-volatility regimes kill edges — stand aside.
4. No trade without risk/reward: defined stop, defined target, RR >= 1.3.
5. "NO TRADE" is the default. A trade must EARN its conviction.
"""


import numpy as np
import pandas as pd




MODELS = ["trend", "momentum", "bxtrender", "macd", "rsi", "meanrev", "volume"]


def analyze(df: pd.DataFrame, account: float = 5000.0,
            risk_pct: float = 1.0, skew: float | None = None,
            flow_call_share: float | None = None) -> dict:
    """Full desk-style analysis of one ticker. Returns a verdict dict."""
    comp = composite(df)
    last = comp.iloc[-1]
    score = float(last["score"])
    price = float(df["Close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    regime = float(vol_regime(df).iloc[-1])

    direction = 1 if score >= BUY_TH else (-1 if score <= SELL_TH else 0)

    # --- 1. Model agreement ------------------------------------------------
    signs = np.sign([float(last[m]) for m in MODELS])
    agree = int((signs == direction).sum()) if direction != 0 else 0
    agree_frac = agree / len(MODELS)

    # --- 2. Historical edge on THIS ticker ---------------------------------
    sharpe = 0.0
    n_trades = 0
    try:
        bt = run_backtest(df, BTConfig(starting_cash=account,
                                       risk_per_trade=risk_pct / 100))
        sharpe = float(bt.metrics["Sharpe"])
        n_trades = int(bt.metrics["Trades"])
    except Exception:
        pass
    edge_ok = sharpe > 0.3 and n_trades >= 5

    # --- 3. Levels & risk/reward -------------------------------------------
    look = min(63, len(df) - 1)
    if direction >= 0:
        stop = price - 2.5 * a
        risk_dist = price - stop
        swing = float(df["High"].rolling(look).max().iloc[-1])
        # At/near new highs there is no overhead resistance — use a
        # measured-move target (2R) instead of punishing the breakout.
        target = max(swing, price + 2.0 * risk_dist)
    else:
        stop = price + 2.5 * a
        risk_dist = stop - price
        swing = float(df["Low"].rolling(look).min().iloc[-1])
        target = min(swing, price - 2.0 * risk_dist)
    risk = abs(price - stop)
    reward = abs(target - price)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    # --- 4. Options sentiment (optional) ------------------------------------
    skew_adj = 0.0
    if skew is not None:
        # heavy put skew argues against fresh longs / supports shorts
        if direction == 1 and skew > 8:
            skew_adj = -8.0
        elif direction == -1 and skew > 8:
            skew_adj = +5.0
        elif direction == 1 and skew < 2:
            skew_adj = +3.0

    # --- 4b. Unusual-flow tilt (live options positioning) ---------------------
    flow_adj = 0.0
    if flow_call_share is not None:
        if direction == 1 and flow_call_share >= 0.65:
            flow_adj = +5.0
        elif direction == 1 and flow_call_share <= 0.35:
            flow_adj = -5.0
        elif direction == -1 and flow_call_share <= 0.35:
            flow_adj = +5.0

    # --- 5. Conviction (0-100) ----------------------------------------------
    conviction = 100 * (
        0.40 * min(abs(score) / 0.50, 1.0)      # signal strength
        + 0.25 * agree_frac                     # model agreement
        + 0.15 * regime                         # calm regime
        + 0.20 * min(max(sharpe, 0) / 1.2, 1.0) # proven edge here
    ) + skew_adj + flow_adj
    conviction = float(np.clip(conviction, 0, 100))

    # --- 6. Verdict ----------------------------------------------------------
    reasons_pro, reasons_con = [], []

    if direction == 1:
        reasons_pro.append(f"Composite score {score:+.2f} above BUY threshold")
    elif direction == -1:
        reasons_pro.append(f"Composite score {score:+.2f} below SELL threshold")
    else:
        reasons_con.append(f"Composite score {score:+.2f} is in the dead zone "
                           f"({SELL_TH} to {BUY_TH}) — no directional edge")

    if direction != 0:
        if agree >= 5:
            reasons_pro.append(f"{agree}/{len(MODELS)} models agree on direction")
        else:
            reasons_con.append(f"Only {agree}/{len(MODELS)} models agree — mixed signals")

    if regime >= 1.0:
        reasons_pro.append("Calm volatility regime — edges work best here")
    elif regime >= 0.5:
        reasons_con.append("Elevated volatility — position sizes should shrink")
    else:
        reasons_con.append("Volatility storm — historically the worst time to trade signals")

    if edge_ok:
        reasons_pro.append(f"Signal has real history on this ticker "
                           f"(Sharpe {sharpe:.2f}, {n_trades} trades)")
    else:
        reasons_con.append(f"Weak historical edge on this ticker "
                           f"(Sharpe {sharpe:.2f}, {n_trades} trades)")

    if rr >= 1.8:
        reasons_pro.append(f"Attractive risk/reward {rr}:1 to the {look}-day level")
    elif rr >= 1.3:
        reasons_pro.append(f"Acceptable risk/reward {rr}:1")
    else:
        reasons_con.append(f"Poor risk/reward {rr}:1 — target too close to stop")

    if flow_call_share is not None and abs(flow_adj) > 0:
        (reasons_pro if flow_adj > 0 else reasons_con).append(
            f"Unusual options flow: {flow_call_share*100:.0f}% of fresh premium "
            f"in calls ({flow_adj:+.0f} conviction)")
    if skew is not None and abs(skew_adj) > 0:
        (reasons_pro if skew_adj > 0 else reasons_con).append(
            f"Options skew {skew:+.1f} pts adjusts conviction {skew_adj:+.0f}")

    tradeable = (direction != 0 and conviction >= 55 and rr >= 1.3
                 and regime > 0.25)
    if tradeable:
        verdict = "LONG" if direction == 1 else "SHORT"
    else:
        verdict = "NO TRADE"

    # --- 7. Sizing ------------------------------------------------------------
    risk_dollars = account * risk_pct / 100
    shares = int(min(risk_dollars / risk, account / price)) if risk > 0 else 0

    return {
        "verdict": verdict,
        "conviction": round(conviction),
        "score": round(score, 3),
        "price": round(price, 2),
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": rr,
        "atr": round(a, 2),
        "regime": regime,
        "agree": agree,
        "sharpe": round(sharpe, 2),
        "n_trades": n_trades,
        "shares": shares,
        "risk_dollars": round(shares * risk, 0),
        "reasons_pro": reasons_pro,
        "reasons_con": reasons_con,
    }


# ===== quant/backtest.py =====
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


from dataclasses import dataclass

import numpy as np
import pandas as pd





@dataclass
class BTConfig:
    starting_cash: float = 5000.0
    commission_pct: float = 0.001
    atr_stop_mult: float = 2.5
    risk_per_trade: float = 0.01
    mode: str = "auto"              # "auto" | "trend" | "dip" | "core" | "blend"
    allow_short: bool = False       # trend mode: short below SMA200
    fib_filter: bool = True         # dip mode: only buy inside 0.382-0.786 zone
    breakeven_r: float = 1.0        # move stop to entry after +1R
    time_stop_trend: int = 20       # bars; exit if unprofitable by then
    time_stop_dip: int = 10
    regime_filter: bool = True      # longs only above SMA200
    vol_target: bool = True         # inverse-vol position scaling
    bars_per_year: int = 252        # 1638 hourly, 52 weekly, 12 monthly


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


def _metrics(equity: pd.Series, trades: pd.DataFrame, bh: pd.Series,
             bpy: int = 252) -> dict:
    rets = equity.pct_change().dropna()
    n_years = max(len(equity) / bpy, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(bpy)) if downside and downside > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    scr = (trades["pnl"].abs() < trades["pnl"].abs().mean() * 0.1).sum() if len(trades) else 0
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / n_years) - 1
    pf = None
    if len(trades):
        gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_l = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        pf = round(float(gross_w / gross_l), 2) if gross_l > 0 else None
    extra = {}
    if len(trades):
        w = trades[trades["pnl"] > 0]["pnl"]
        l = trades[trades["pnl"] < 0]["pnl"]
        extra["Avg win $"] = round(float(w.mean()), 0) if len(w) else 0
        extra["Avg loss $"] = round(float(l.mean()), 0) if len(l) else 0
        extra["Expectancy R"] = round(float(trades["R"].mean()), 2) \
            if "R" in trades else None
        extra["Avg hold (bars)"] = round(float(trades["bars"].mean()), 1) \
            if "bars" in trades else None
        signs = (trades["pnl"] > 0).astype(int).values
        mx_w = mx_l = cw = cl = 0
        for s_ in signs:
            cw = cw + 1 if s_ else 0
            cl = cl + 1 if not s_ else 0
            mx_w, mx_l = max(mx_w, cw), max(mx_l, cl)
        extra["Max consec W/L"] = f"{mx_w} / {mx_l}"
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
        **extra,
    }


def run_backtest(df: pd.DataFrame, cfg: BTConfig = BTConfig()) -> BTResult:
    mode = cfg.mode
    if mode == "auto":
        h = _hurst_quick(df["Close"])
        mode = "trend" if h >= 0.5 else "dip"

    comp = composite(df) if mode in ("trend", "blend") else None
    bx = bxtrender(df)
    a = atr(df)
    s200 = sma(df["Close"], 200)
    r2 = rsi(df["Close"], 2)

    # Fibonacci retracement zone of the rolling 126-bar swing (for dip mode):
    roll_hi = df["High"].rolling(126, min_periods=40).max()
    roll_lo = df["Low"].rolling(126, min_periods=40).min()
    rng_ = roll_hi - roll_lo
    fib_lo = roll_hi - 0.786 * rng_          # deep edge of the pocket
    fib_hi = roll_hi - 0.382 * rng_          # shallow edge

    atr_pct = (a / df["Close"])
    med = atr_pct.rolling(252, min_periods=60).median()
    vt = (med / atr_pct).clip(0.5, 1.5).fillna(1.0) if cfg.vol_target else \
        pd.Series(1.0, index=df.index)

    cash = cfg.starting_cash
    shares = 0.0                      # negative = short
    entry_price = stop = 0.0
    entry_i = 0
    be_armed = False
    entry_kind = ""                   # "trend" | "dip" (matters in blend)
    trade_lo = trade_hi = 0.0         # MAE/MFE tracking
    equity_rows, trade_rows = [], []

    o_, h_, l_, c_ = (df[k].values for k in ("Open", "High", "Low", "Close"))
    sig = comp["signal"].values if comp is not None else None
    bx_long = bx["long_osc"].values
    bx_rising = bx["t3_rising"].values
    bx_buyturn = bx["buy_turn"].values
    idx = df.index
    time_stop = cfg.time_stop_trend if mode == "trend" else cfg.time_stop_dip

    def close_pos(i, exit_px, reason):
        nonlocal cash, shares, be_armed
        exit_px = max(exit_px, 0.01)
        if shares > 0:
            proceeds = shares * exit_px * (1 - cfg.commission_pct)
            pnl = proceeds - shares * entry_price * (1 + cfg.commission_pct)
            cash += proceeds
        else:  # short cover
            qty = -shares
            cost = qty * exit_px * (1 + cfg.commission_pct)
            pnl = qty * entry_price * (1 - cfg.commission_pct) - cost
            cash += qty * entry_price * (1 - cfg.commission_pct) - cost + qty * entry_price * 0  # margin release handled via cash below
            cash = cash  # cash already holds short proceeds at entry
        r_unit_ = cfg.atr_stop_mult * a.values[max(entry_i - 1, 0)]
        mae_r = (entry_price - trade_lo) / r_unit_ if r_unit_ > 0 else 0
        mfe_r = (trade_hi - entry_price) / r_unit_ if r_unit_ > 0 else 0
        trade_rows.append({"entry_date": idx[entry_i], "exit_date": idx[i],
                           "side": "LONG" if shares > 0 else "SHORT",
                           "entry": round(entry_price, 2),
                           "exit": round(exit_px, 2),
                           "pnl": round(pnl, 2), "reason": reason,
                           "bars": int(i - entry_i),
                           "R": round(pnl / (abs(shares) * r_unit_), 2)
                           if r_unit_ > 0 and shares != 0 else 0.0,
                           "MAE_R": round(float(mae_r), 2),
                           "MFE_R": round(float(mfe_r), 2)})
        shares = 0.0
        be_armed = False

    for i in range(1, len(df)):
        o, hi, lo, c = o_[i], h_[i], l_[i], c_[i]
        prev_atr = a.values[i - 1]
        s200_ok = not np.isnan(s200.values[i - 1])
        above200 = s200_ok and c_[i - 1] > s200.values[i - 1]
        below200 = s200_ok and c_[i - 1] < s200.values[i - 1]

        # ================= CORE mode: improved buy & hold =================
        if mode == "core":
            in_market = shares > 0
            healthy = above200 and bx_long[i - 1] > 0
            if in_market and not healthy:
                close_pos(i, o, "regime exit")
            elif (not in_market) and healthy:
                shares = float((cash * 0.98) / (o * (1 + cfg.commission_pct)))
                if shares * o >= 100:
                    cash -= shares * o * (1 + cfg.commission_pct)
                    entry_price, entry_i = o, i
                else:
                    shares = 0.0
            equity_rows.append(cash + shares * c)
            continue

        # ================= exits (trend / dip) =================
        if shares != 0:
            trade_lo = min(trade_lo, lo)
            trade_hi = max(trade_hi, hi)
            bars_in = i - entry_i
            r_dist = cfg.atr_stop_mult * a.values[entry_i - 1]
            long_pos = shares > 0

            if long_pos:
                if not be_armed and hi >= entry_price + cfg.breakeven_r * r_dist:
                    stop = max(stop, entry_price); be_armed = True
                if mode == "trend" or (mode == "blend"
                                       and entry_kind == "trend"):
                    stop = max(stop, hi - cfg.atr_stop_mult * prev_atr)
                hit = lo <= stop
            else:
                if not be_armed and lo <= entry_price - cfg.breakeven_r * r_dist:
                    stop = min(stop, entry_price); be_armed = True
                if mode == "trend":
                    stop = min(stop, lo + cfg.atr_stop_mult * prev_atr)
                hit = hi >= stop

            exit_now, reason = False, ""
            if hit:
                exit_now, reason = True, "breakeven" if be_armed else "stop"
            elif mode in ("trend", "blend") and long_pos and sig[i - 1] == "SELL":
                exit_now, reason = True, "signal"
            elif mode == "trend" and not long_pos and sig[i - 1] == "BUY":
                exit_now, reason = True, "signal"
            elif long_pos and r2.values[i - 1] > 65 and (
                    mode == "dip" or
                    (mode == "blend" and entry_kind == "dip")):
                exit_now, reason = True, "target(rsi)"
            elif bars_in >= time_stop and (
                    (long_pos and c_[i - 1] < entry_price) or
                    (not long_pos and c_[i - 1] > entry_price)):
                exit_now, reason = True, "time"

            if exit_now:
                px = o
                if hit:
                    px = min(o, stop) if long_pos and o > stop else \
                         max(o, stop) if (not long_pos) and o < stop else o
                close_pos(i, px, reason)

        # ================= entries =================
        if shares == 0 and prev_atr > 0:
            stop_dist = cfg.atr_stop_mult * prev_atr
            risk_dollars = cash * cfg.risk_per_trade * vt.values[i - 1]
            size = min(risk_dollars / stop_dist,
                       cash / (o * (1 + cfg.commission_pct)))

            go_long = go_short = False
            if mode == "blend":
                trend_go = (above200 or not cfg.regime_filter) and \
                           sig[i - 1] == "BUY" and bx_long[i - 1] > 0 and \
                           bx_rising[i - 1]
                pocket_b = (not np.isnan(fib_lo.values[i - 1]) and
                            fib_lo.values[i - 1] <= c_[i - 1]
                            <= fib_hi.values[i - 1])
                dip_go = (above200 or not cfg.regime_filter) and \
                         r2.values[i - 1] < 10 and (pocket_b or bx_buyturn[i - 1])
                go_long = trend_go or dip_go
                entry_kind = "dip" if (dip_go and not trend_go) else "trend"
            elif mode == "trend":
                # B-Xtrender confirmation: long osc positive & T3 rising
                go_long = (above200 or not cfg.regime_filter) and \
                          sig[i - 1] == "BUY" and \
                          bx_long[i - 1] > 0 and bx_rising[i - 1]
                if cfg.allow_short:
                    go_short = below200 and sig[i - 1] == "SELL" and \
                               bx_long[i - 1] < 0 and not bx_rising[i - 1]
            else:  # dip
                in_pocket = True
                if cfg.fib_filter and not np.isnan(fib_lo.values[i - 1]):
                    in_pocket = fib_lo.values[i - 1] <= c_[i - 1] <= fib_hi.values[i - 1]
                go_long = (above200 or not cfg.regime_filter) and \
                          r2.values[i - 1] < 10 and \
                          (in_pocket or bx_buyturn[i - 1])

            if go_long and size * o >= 100:
                shares = float(size)
                cash -= shares * o * (1 + cfg.commission_pct)
                entry_price, entry_i = o, i
                stop = o - stop_dist
                be_armed = False
                trade_lo = trade_hi = o
            elif go_short and size * o >= 100:
                shares = -float(size)
                cash += size * o * (1 - cfg.commission_pct)   # short proceeds
                entry_price, entry_i = o, i
                stop = o + stop_dist
                be_armed = False
                trade_lo = trade_hi = o

        equity_rows.append(cash + shares * c)

    equity = pd.Series(equity_rows, index=idx[1:], name="strategy")
    bh = pd.Series(cfg.starting_cash / c_[0] * c_[1:], index=idx[1:],
                   name="buy_hold")
    trades = pd.DataFrame(trade_rows)
    return BTResult(equity, bh, trades, _metrics(equity, trades, bh, cfg.bars_per_year), mode)


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


# ===== quant/master.py =====
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


import numpy as np
import pandas as pd





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


# ===== quant/flow.py =====
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


# ===== quant/seasonality.py =====
"""Seasonality stats (Detrick-style) + fundamental quality snapshot."""


import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar month: historical win rate & average return."""
    m = df["Close"].resample("ME").last().pct_change().dropna()
    if len(m) < 12:
        return pd.DataFrame()
    tab = pd.DataFrame({"month_n": m.index.month, "ret": m.values})
    g = tab.groupby("month_n")["ret"]
    out = pd.DataFrame({
        "win rate %": (g.apply(lambda s: (s > 0).mean()) * 100).round(0),
        "avg return %": (g.mean() * 100).round(2),
        "years": g.count(),
    })
    out.index = [MONTHS[i - 1] for i in out.index]
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def fundamental_snapshot(ticker: str) -> dict:
    """DebttoValue-style card: 'price follows growth & margin & free cash flow'."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    out = {
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margin": info.get("grossMargins"),
        "op_margin": info.get("operatingMargins"),
        "fcf_yield": (fcf / mcap) if (fcf and mcap) else None,
        "fwd_pe": info.get("forwardPE"),
        "name": info.get("shortName") or ticker,
    }

    checks = 0
    total = 0
    for key, thresh in (("revenue_growth", 0.10), ("gross_margin", 0.40),
                        ("op_margin", 0.15), ("fcf_yield", 0.03)):
        v = out.get(key)
        if v is not None:
            total += 1
            if v >= thresh:
                checks += 1
    out["quality_score"] = f"{checks}/{total}" if total else None
    return out


# ===== quant/rl_lab.py =====
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


import numpy as np
import pandas as pd




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


# ===== quant/journal.py =====
"""Track record — institutional-style paper-trading journal.

Design principles (what an allocator actually checks):
  * AUDIT TRAIL — every recorded plan is stamped: timestamp (UTC), model
    version, market regime at entry. Entries are append-only.
  * MARK-TO-MARKET — open positions are revalued on real daily bars; stops
    and targets are enforced mechanically (first touch, stop wins ties —
    the conservative convention).
  * VERIFIABLE — the whole journal exports to CSV/JSON so the record can be
    inspected, backed up, and re-imported. Free-tier hosting wipes local
    files on redeploy; the export IS the custody solution.
"""


import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

JOURNAL_PATH = "data/journal.json"
MODEL_VERSION = "QuantSignal v12"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load_journal(path: str = JOURNAL_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"meta": {"inception": None, "account": 5000.0,
                     "version": MODEL_VERSION},
            "positions": []}


def save_journal(j: dict, path: str = JOURNAL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(j, f, indent=1, default=str)


def journal_to_csv(j: dict) -> str:
    return pd.DataFrame(j["positions"]).to_csv(index=False)


def journal_from_csv(csv_text: str, account: float = 5000.0) -> dict:
    df = pd.read_csv(pd.io.common.StringIO(csv_text))
    j = {"meta": {"inception": df["recorded_utc"].min() if len(df) else None,
                  "account": account, "version": MODEL_VERSION},
         "positions": df.to_dict(orient="records")}
    return j


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_plan(j: dict, plan: pd.DataFrame, regime: str,
                account: float) -> tuple[dict, int]:
    """Append an Alpha-Engine plan to the journal with full stamps."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    added = 0
    open_tkrs = {p["ticker"] for p in j["positions"]
                 if p.get("status") == "OPEN"}
    for _, r in plan.iterrows():
        if r["ticker"] in open_tkrs:
            continue                      # no doubling into an open name
        j["positions"].append({
            "id": len(j["positions"]) + 1,
            "recorded_utc": now,
            "model_version": MODEL_VERSION,
            "regime_at_entry": regime,
            "ticker": r["ticker"],
            "side": "LONG",
            "shares": int(r["shares"]),
            "entry": float(r["entry ~"]),
            "stop": float(r["stop"]),
            "target": float(r["target"]),
            "conviction": int(r["conviction"]),
            "status": "OPEN",
            "exit": None, "exit_date": None, "exit_reason": None,
        })
        added += 1
    if j["meta"]["inception"] is None and added:
        j["meta"]["inception"] = now
    j["meta"]["account"] = float(account)
    return j, added


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------

def mark_to_market(j: dict, fetcher) -> dict:
    """Revalue every position on real bars. `fetcher(ticker)` -> OHLCV df.

    Returns dict with blotter df, equity curve, benchmark curve, stats.
    Mutates position statuses when stops/targets were touched.
    """
    if not j["positions"]:
        return {"empty": True}

    account = float(j["meta"].get("account", 5000.0))
    histories: dict[str, pd.DataFrame] = {}
    data_issues: list[str] = []
    rows = []

    for p in j["positions"]:
        t = p["ticker"]
        if t not in histories:
            try:
                histories[t] = fetcher(t)
                if histories[t].empty:
                    data_issues.append(f"{t}: empty history")
            except Exception as exc:
                histories[t] = pd.DataFrame()
                data_issues.append(f"{t}: {type(exc).__name__}")
        df = histories[t]
        entry_date = pd.to_datetime(str(p["recorded_utc"])[:10])
        cur_px, pnl = p["entry"], 0.0

        if not df.empty:
            bars = df[df.index > entry_date]
            if p["status"] == "OPEN":
                for dt_, b in bars.iterrows():
                    if b["Low"] <= p["stop"]:
                        p.update(status="CLOSED", exit=float(p["stop"]),
                                 exit_date=str(dt_.date()),
                                 exit_reason="stop")
                        break
                    if b["High"] >= p["target"]:
                        p.update(status="CLOSED", exit=float(p["target"]),
                                 exit_date=str(dt_.date()),
                                 exit_reason="target")
                        break
            cur_px = float(p["exit"]) if p["status"] == "CLOSED" else \
                float(df["Close"].iloc[-1])
        pnl = (cur_px - p["entry"]) * p["shares"]
        rows.append({**{k: p[k] for k in ("id", "recorded_utc", "ticker",
                                          "shares", "entry", "stop", "target",
                                          "conviction", "status",
                                          "exit_reason")},
                     "mark": round(cur_px, 2),
                     "P&L $": round(pnl, 0),
                     "P&L %": round(pnl / (p["entry"] * p["shares"]) * 100, 1)
                     if p["shares"] else 0,
                     "regime_at_entry": p.get("regime_at_entry", "")})

    blotter = pd.DataFrame(rows)

    # ---- daily portfolio equity curve ------------------------------------
    start = pd.to_datetime(min(str(p["recorded_utc"])[:10]
                               for p in j["positions"]))
    all_days = pd.bdate_range(start, pd.Timestamp.today())
    invested_cost = sum(p["entry"] * p["shares"] for p in j["positions"])
    cash = account - invested_cost

    eq = pd.Series(0.0, index=all_days)
    for p in j["positions"]:
        t = p["ticker"]
        df = histories.get(t, pd.DataFrame())
        if df.empty:
            continue
        e_date = pd.to_datetime(str(p["recorded_utc"])[:10])
        px = df["Close"].reindex(all_days).ffill()
        val = px * p["shares"]
        val[all_days < e_date] = 0.0
        # freeze value after exit
        if p["status"] == "CLOSED" and p["exit_date"]:
            x_date = pd.to_datetime(p["exit_date"])
            val[all_days >= x_date] = p["exit"] * p["shares"]
        # before entry, that cash was uninvested -> add cost back
        val[all_days < e_date] = p["entry"] * p["shares"]
        eq += val
    equity = (eq + cash).dropna()
    equity = equity[equity > 0]

    # ---- benchmark: SPY scaled to same start ------------------------------
    try:
        spy = fetcher("SPY")["Close"].reindex(all_days).ffill().dropna()
        bench = spy / spy.iloc[0] * account
    except Exception:
        bench = pd.Series(dtype=float)

    stats = _stats(equity, bench, blotter, account)
    monthly = _monthly(equity)
    return {"empty": False, "blotter": blotter, "equity": equity,
            "bench": bench, "stats": stats, "monthly": monthly,
            "data_issues": data_issues}


def _stats(eq: pd.Series, bench: pd.Series, blotter: pd.DataFrame,
           account: float) -> dict:
    out = {"Account $": account}
    if len(eq) < 2:
        out["Note"] = "Need a few days of marks for statistics"
        return out
    r = eq.pct_change().dropna()
    days = len(eq)
    tot = eq.iloc[-1] / account - 1
    out["Equity $"] = round(float(eq.iloc[-1]), 0)
    out["Total return %"] = round(tot * 100, 2)
    if len(bench) >= 2:
        bt = bench.iloc[-1] / bench.iloc[0] - 1
        out["SPY same period %"] = round(float(bt) * 100, 2)
        out["Alpha vs SPY %"] = round((tot - float(bt)) * 100, 2)
    if r.std() > 0 and days > 10:
        out["Sharpe (live)"] = round(float(r.mean() / r.std() * np.sqrt(252)), 2)
    out["Max DD %"] = round(float((eq / eq.cummax() - 1).min()) * 100, 2)

    closed = blotter[blotter["status"] == "CLOSED"]
    out["Open / Closed"] = f"{int((blotter['status'] == 'OPEN').sum())} / {len(closed)}"
    if len(closed):
        wins = closed[closed["P&L $"] > 0]
        out["Hit rate %"] = round(len(wins) / len(closed) * 100, 1)
        gw = wins["P&L $"].sum()
        gl = -closed.loc[closed["P&L $"] < 0, "P&L $"].sum()
        out["Profit factor"] = round(float(gw / gl), 2) if gl > 0 else "∞"
    out["Heat (risk if all stops hit) $"] = round(float(
        ((blotter.loc[blotter["status"] == "OPEN", "entry"] -
          blotter.loc[blotter["status"] == "OPEN", "stop"]) *
         blotter.loc[blotter["status"] == "OPEN", "shares"]).sum()), 0)
    return out


def _monthly(eq: pd.Series) -> pd.DataFrame:
    if len(eq) < 22:
        return pd.DataFrame()
    m = eq.resample("ME").last().pct_change().dropna() * 100
    if not len(m):
        return pd.DataFrame()
    return pd.DataFrame({"month": m.index.strftime("%Y-%m"),
                         "return %": m.round(2).values})


# ===== quant/garch_pairs.py =====
"""GARCH(1,1) volatility forecast (arch) + pairs trading (statsmodels).

Both straight from the awesome-quant toolbox.
"""


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


# ===== quant/portfolio.py =====
"""Portfolio lab — optimization via PyPortfolioOpt (from awesome-quant).

Three allocators, three philosophies:
  MAX SHARPE — the classic Markowitz tangency portfolio (needs return
               estimates, which are noisy — handle with care).
  MIN VOL    — ignores returns entirely; just the quietest mix.
  HRP        — Hierarchical Risk Parity (Lopez de Prado 2016): clusters
               assets by correlation and splits risk down the tree. No
               matrix inversion, no return estimates — the robust choice.
"""


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


# ===== quant/opt_edge.py =====
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


# ===== quant/timeframes.py =====
"""Timeframes — one selector, correct math on every horizon.

Indicator windows are in BARS (standard practice): a 50-bar average on
weekly bars is a ~1-year trend measure, on hourly bars a ~2-week one.
Annualization factors keep Sharpe/CAGR honest per timeframe.
"""


import pandas as pd
import streamlit as st
import yfinance as yf

# label -> (yf interval, default period, bars per year, min bars needed)
TIMEFRAMES = {
    "Hours (1h)": ("1h", "720d", 1638, 300),
    "Daily": ("1d", "2y", 252, 260),
    "Weekly": ("1wk", "10y", 52, 150),
    "Monthly": ("1mo", "max", 12, 60),
}
TF_LABELS = list(TIMEFRAMES.keys())


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_tf(ticker: str, tf_label: str, period: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV at the chosen timeframe. Real (unadjusted) prices."""
    interval, default_period, _, _ = TIMEFRAMES[tf_label]
    try:
        df = yf.Ticker(ticker).history(period=period or default_period,
                                       interval=interval, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols].dropna()


def tf_meta(tf_label: str) -> dict:
    interval, period, bpy, min_bars = TIMEFRAMES[tf_label]
    return {"interval": interval, "period": period,
            "bars_per_year": bpy, "min_bars": min_bars}


# ===== quant/live.py =====
"""Live engine — market clock, near-real-time quotes, live-bar patching.

The trick that makes the whole site "live": every model runs on daily bars,
so we fetch the current quote (cached ~20s) and PATCH it into today's bar.
Every downstream number — composite score, verdict, stops, GEX distance,
track-record marks — then moves with the market automatically.
"""


from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

ET = ZoneInfo("America/New_York")


def market_status() -> dict:
    """NYSE regular-session clock (holidays not modeled — weekends are)."""
    now = datetime.now(ET)
    is_weekday = now.weekday() < 5
    open_t, close_t = dtime(9, 30), dtime(16, 0)
    is_open = is_weekday and open_t <= now.time() <= close_t
    if is_open:
        label, emoji = "MARKET OPEN", "🟢"
        mins = (close_t.hour * 60 + close_t.minute) - \
               (now.hour * 60 + now.minute)
        detail = f"closes in {mins // 60}h {mins % 60}m"
    else:
        label, emoji = "MARKET CLOSED", "⚫"
        pre = is_weekday and dtime(4, 0) <= now.time() < open_t
        post = is_weekday and close_t < now.time() <= dtime(20, 0)
        detail = ("pre-market" if pre else
                  "after-hours" if post else "next session Mon–Fri 9:30 ET")
    return {"open": is_open, "label": label, "emoji": emoji,
            "detail": detail,
            "et_time": now.strftime("%H:%M:%S ET")}


@st.cache_data(ttl=45, show_spinner=False)
def live_quote(ticker: str) -> dict:
    """Latest price + day change. Cached 20s so refresh loops stay polite."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        px = float(fi["last_price"])
        prev = float(fi.get("previous_close") or px)
        day_hi = fi.get("day_high")
        day_lo = fi.get("day_low")
        return {"price": px,
                "prev": prev,
                "chg_pct": round((px / prev - 1) * 100, 2) if prev else 0.0,
                "day_high": float(day_hi) if day_hi else px,
                "day_low": float(day_lo) if day_lo else px}
    except Exception:
        try:
            h = yf.Ticker(ticker).history(period="1d", interval="1m")
            px = float(h["Close"].iloc[-1])
            return {"price": px, "prev": px, "chg_pct": 0.0,
                    "day_high": float(h["High"].max()),
                    "day_low": float(h["Low"].min())}
        except Exception:
            return {}


def patch_live_bar(df: pd.DataFrame, ticker: str,
                   quote: dict | None = None) -> pd.DataFrame:
    """Overwrite/append today's bar with the live quote.

    Every indicator computed on the result reflects the price RIGHT NOW.
    `quote` injectable for testing.
    """
    if df is None or df.empty:
        return df
    q = quote if quote is not None else live_quote(ticker)
    if not q or not q.get("price"):
        return df

    out = df.copy()
    today = pd.Timestamp(datetime.now(ET).date())
    last = out.index[-1].normalize()
    px = q["price"]

    if last == today:                        # session bar exists -> update it
        out.iloc[-1, out.columns.get_loc("Close")] = px
        out.iloc[-1, out.columns.get_loc("High")] = max(
            float(out["High"].iloc[-1]), q.get("day_high", px))
        out.iloc[-1, out.columns.get_loc("Low")] = min(
            float(out["Low"].iloc[-1]), q.get("day_low", px))
    else:                                    # append a synthetic live bar
        new = pd.DataFrame({
            "Open": [q.get("prev", px)],
            "High": [q.get("day_high", px)],
            "Low": [q.get("day_low", px)],
            "Close": [px],
            "Volume": [float(out["Volume"].iloc[-1])],
        }, index=[today])
        out = pd.concat([out, new])
    return out


# ===== quant/risk.py =====
"""Risk management — the quant models that keep a $5K account alive.

Everything here answers ONE question a desk asks before every trade:
"how much can this realistically cost me, and am I over the line?"

  * Position risk      — $ and % at risk to the stop (the only sizing that matters)
  * Portfolio heat     — total simultaneous risk if every stop hits at once
  * Parametric VaR/CVaR — 1-day 95/99% loss estimate on the whole book
  * Correlation-adjusted heat — naive heat lies when positions move together
  * Risk of ruin       — probability of losing X% given your edge & bet size
  * Kelly ladder       — full / half / quarter Kelly with the growth-vs-pain tradeoff
"""


import numpy as np
import pandas as pd
from scipy import stats


def position_risk(entry: float, stop: float, shares: int) -> dict:
    per_share = abs(entry - stop)
    return {
        "risk_$": round(per_share * shares, 2),
        "risk_per_share": round(per_share, 2),
        "notional_$": round(entry * shares, 2),
    }


def portfolio_var(positions: list[dict], returns: dict[str, pd.Series],
                  account: float, horizon_days: int = 1,
                  conf: float = 0.95) -> dict:
    """Parametric (variance-covariance) VaR/CVaR on the open book.

    positions: [{ticker, shares, entry}], returns: {ticker: daily ret series}.
    """
    tks = [p["ticker"] for p in positions if p["ticker"] in returns]
    if not tks:
        return {}
    w_dollar = np.array([next(p["shares"] * p["entry"] for p in positions
                              if p["ticker"] == t) for t in tks])
    R = pd.DataFrame({t: returns[t] for t in tks}).dropna()
    if len(R) < 30:
        return {}
    cov = R.cov().values * horizon_days
    port_var_dollar = float(np.sqrt(w_dollar @ cov @ w_dollar))
    z = stats.norm.ppf(conf)
    var = z * port_var_dollar
    # CVaR (expected shortfall) for a normal dist
    cvar = port_var_dollar * stats.norm.pdf(z) / (1 - conf)
    return {
        "VaR_$": round(var, 0),
        "VaR_%": round(var / account * 100, 2),
        "CVaR_$": round(cvar, 0),
        "CVaR_%": round(cvar / account * 100, 2),
        "conf": int(conf * 100),
        "horizon": horizon_days,
        "gross_exposure_%": round(w_dollar.sum() / account * 100, 1),
    }


def correlation_heat(positions: list[dict], returns: dict[str, pd.Series],
                     account: float) -> dict:
    """Naive heat assumes independence; real heat accounts for correlation."""
    tks = [p["ticker"] for p in positions if p["ticker"] in returns]
    if len(tks) < 2:
        return {}
    R = pd.DataFrame({t: returns[t] for t in tks}).dropna()
    if len(R) < 30:
        return {}
    corr = R.corr()
    risks = np.array([next((p["entry"] - p["stop"]) * p["shares"]
                          for p in positions if p["ticker"] == t) for t in tks])
    naive = float(risks.sum())
    combined = float(np.sqrt(risks @ corr.values @ risks))
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean())
    return {
        "naive_heat_$": round(naive, 0),
        "corr_adj_heat_$": round(combined, 0),
        "avg_correlation": round(avg_corr, 2),
        "diversification_benefit_%": round((1 - combined / naive) * 100, 0)
        if naive > 0 else 0,
        "warning": avg_corr > 0.6,
    }


def risk_of_ruin(win_rate: float, avg_win: float, avg_loss: float,
                 risk_per_trade_pct: float, ruin_pct: float = 0.30,
                 n_sims: int = 5000, n_trades: int = 200,
                 seed: int = 7) -> dict:
    """Monte Carlo probability of drawing down `ruin_pct` given the edge."""
    if avg_loss <= 0:
        return {}
    rng = np.random.default_rng(seed)
    payoff = avg_win / avg_loss
    ruined = 0
    for _ in range(n_sims):
        eq = 1.0
        peak = 1.0
        for _ in range(n_trades):
            bet = risk_per_trade_pct / 100
            if rng.random() < win_rate:
                eq *= 1 + bet * payoff
            else:
                eq *= 1 - bet
            peak = max(peak, eq)
            if eq <= peak * (1 - ruin_pct):
                ruined += 1
                break
    p = ruined / n_sims
    return {
        "ruin_threshold_%": int(ruin_pct * 100),
        "prob_of_ruin_%": round(p * 100, 1),
        "payoff_ratio": round(payoff, 2),
        "expectancy_R": round(win_rate * payoff - (1 - win_rate), 3),
        "verdict": ("🟢 Robust" if p < 0.05 else "🟡 Survivable" if p < 0.20
                    else "🔴 Dangerous — cut size"),
    }


def kelly_ladder(win_rate: float, rr: float) -> dict:
    b = max(rr, 1e-9)
    f = win_rate - (1 - win_rate) / b
    return {
        "full_kelly_%": round(f * 100, 1),
        "half_kelly_%": round(f * 50, 1),
        "quarter_kelly_%": round(f * 25, 1),
        "edge": f > 0,
    }


# ===== quant/validation.py =====
"""Validation lab — the honesty engine a real desk demands.

Four tests that separate real edges from data-mined noise:

  1. DEFLATED SHARPE (Bailey & López de Prado 2014) — corrects a strategy's
     Sharpe for how many strategies were TRIED. Testing 20 models and keeping
     the best inflates Sharpe; this deflates it back to reality.

  2. MULTIPLE-TESTING p-value (Harvey, Liu & Zhu 2016, RFS) — a t-stat of 2
     is NOT significant when you mined 20 signals. Applies a Bonferroni-style
     haircut so you trust what survives.

  3. MONTE CARLO PERMUTATION — shuffle the returns 500× and ask: could this
     equity curve have happened by luck? Gives an empirical p-value on skill.

  4. BOOTSTRAP CONFIDENCE INTERVAL — resample trades 1000× for a 90% CI on
     CAGR. A wide band that straddles zero = you don't actually know if it works.
"""


import numpy as np
import pandas as pd
from scipy import stats


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & López de Prado deflated Sharpe ratio."""
    if n_obs < 20:
        return {"error": "Too few observations."}
    # Expected max Sharpe from N independent random trials (order statistic)
    emc = 0.5772156649
    e_max = (np.sqrt(2 * np.log(max(n_trials, 2)))
             - (np.log(np.log(max(n_trials, 2))) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(max(n_trials, 2)))))
    # Work in per-period Sharpe (deannualize), as the theory requires.
    sr_p = sharpe / np.sqrt(252)
    sr_std = np.sqrt((1 - skew * sr_p + (kurt - 1) / 4 * sr_p ** 2) / (n_obs - 1))
    sr0 = e_max * sr_std                              # deflated benchmark (per-period)
    dsr = stats.norm.cdf((sr_p - sr0) / sr_std) if sr_std > 0 else 0.0
    return {
        "observed_sharpe": round(float(sharpe), 2),
        "deflated_benchmark_ann": round(float(sr0 * np.sqrt(252)), 2),
        "DSR_probability": round(float(dsr), 3),
        "verdict": ("✅ Likely real edge" if dsr > 0.95 else
                    "⚠️ Borderline" if dsr > 0.75 else
                    "❌ Probably data-mined noise"),
    }


def haircut_pvalue(tstat: float, n_tests: int) -> dict:
    """Harvey-Liu-Zhu style multiple-testing haircut (Bonferroni + BY)."""
    single_p = 2 * (1 - stats.norm.cdf(abs(tstat)))
    bonferroni = min(single_p * n_tests, 1.0)
    # Benjamini-Yekutieli constant
    c = sum(1.0 / i for i in range(1, n_tests + 1))
    by = min(single_p * n_tests * c / 1.0, 1.0)
    return {
        "raw_p": round(float(single_p), 4),
        "bonferroni_p": round(float(bonferroni), 4),
        "BY_p": round(float(by), 4),
        "survives_5pct": bonferroni < 0.05,
        "verdict": ("✅ Survives multiple-testing" if bonferroni < 0.05 else
                    "⚠️ Marginal" if bonferroni < 0.20 else
                    "❌ Not significant after correction"),
    }


def permutation_test(returns: pd.Series, n_perm: int = 500,
                     seed: int = 7) -> dict:
    """Could this equity curve be luck? Shuffle returns, compare final wealth."""
    r = returns.dropna().values
    if len(r) < 30:
        return {"error": "Too few returns."}
    def _sharpe(x):
        return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0

    # Sign-flip permutation: under the null of "no edge", each return's sign
    # is equally likely +/-. This tests whether the DRIFT is real, and unlike
    # reshuffling it is not invariant to a positive mean.
    actual = float(_sharpe(r))
    rng = np.random.default_rng(seed)
    perms = np.array([_sharpe(r * rng.choice([-1, 1], size=len(r)))
                      for _ in range(n_perm)])
    pval = float((perms >= actual).mean())
    return {
        "actual_sharpe": round(actual, 2),
        "perm_sharpe_95pct": round(float(np.percentile(perms, 95)), 2),
        "perm_p_value": round(pval, 3),
        "verdict": ("✅ Beats luck (p<0.05)" if pval < 0.05 else
                    "⚠️ Weak (p<0.20)" if pval < 0.20 else
                    "❌ Indistinguishable from luck"),
    }


def bootstrap_cagr(trade_pnls: pd.Series, starting: float = 5000.0,
                   n_boot: int = 1000, seed: int = 7) -> dict:
    """90% confidence interval on total return by resampling trades."""
    p = trade_pnls.dropna().values
    if len(p) < 8:
        return {"error": "Need at least 8 closed trades for a bootstrap."}
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(n_boot):
        s = rng.choice(p, size=len(p), replace=True)
        finals.append((starting + s.sum()) / starting - 1)
    lo, med, hi = np.percentile(finals, [5, 50, 95])
    return {
        "median_return_%": round(float(med) * 100, 1),
        "CI90_low_%": round(float(lo) * 100, 1),
        "CI90_high_%": round(float(hi) * 100, 1),
        "excludes_zero": lo > 0,
        "verdict": ("✅ Profitable even at the 5th percentile" if lo > 0 else
                    "⚠️ CI straddles zero — edge unproven" if hi > 0 else
                    "❌ Likely unprofitable"),
    }


# ===== quant/events.py =====
"""Event radar — Polymarket macro odds as an INFORMATION source for stocks.

We do NOT trade prediction markets here. We read them: real-money odds on
Fed decisions, recessions, CPI, shutdowns and elections are among the best
live estimates of macro event risk — the stuff that moves US equities.

Data: Polymarket Gamma API (public, read-only, no key).
"""


import json

import pandas as pd
import requests
import streamlit as st

GAMMA = "https://gamma-api.polymarket.com/markets"

# What matters for a US-equities desk
MACRO_TERMS = ("fed", "rate cut", "rate hike", "recession", "inflation",
               "cpi", "gdp", "tariff", "shutdown", "unemployment",
               "s&p", "nasdaq", "treasury", "election", "president",
               "powell", "fomc", "debt ceiling")

RISK_MAP = {
    # keyword -> (direction for stocks if YES, weight)
    "recession": (-1, 3.0),
    "shutdown": (-1, 1.5),
    "rate hike": (-1, 2.0),
    "tariff": (-1, 1.5),
    "default": (-1, 3.0),
    "rate cut": (+1, 1.5),
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_macro_markets(limit: int = 250) -> pd.DataFrame:
    """Pull active markets, keep macro/finance ones, tidy the fields."""
    try:
        r = requests.get(GAMMA, params={
            "closed": "false", "active": "true",
            "limit": limit, "order": "volumeNum", "ascending": "false",
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for m in raw if isinstance(raw, list) else []:
        try:
            q = (m.get("question") or "").strip()
            ql = q.lower()
            if not any(t in ql for t in MACRO_TERMS):
                continue
            prices = m.get("outcomePrices")
            outcomes = m.get("outcomes")
            if isinstance(prices, str):
                prices = json.loads(prices)
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if not prices or not outcomes:
                continue
            p_yes = float(prices[0]) * 100
            rows.append({
                "question": q,
                "yes %": round(p_yes, 1),
                "outcome": str(outcomes[0]),
                "volume $": round(float(m.get("volumeNum") or 0)),
                "ends": str(m.get("endDate") or "")[:10],
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("volume $", ascending=False).reset_index(drop=True)
    return df


def equity_risk_gauge(df: pd.DataFrame) -> dict:
    """Crude but honest: weighted read of 'bad for stocks' probabilities."""
    if df.empty:
        return {}
    score = 0.0
    weight_sum = 0.0
    drivers = []
    for _, r in df.iterrows():
        ql = r["question"].lower()
        for key, (direction, w) in RISK_MAP.items():
            if key in ql:
                contrib = direction * (r["yes %"] / 100) * w
                score += contrib
                weight_sum += w
                drivers.append((r["question"], r["yes %"], direction))
                break
    if weight_sum == 0:
        return {}
    norm = score / weight_sum          # -1..+1-ish
    label = ("🟢 Tailwind" if norm > 0.15 else
             "🔴 Headwind" if norm < -0.15 else "⚪ Neutral")
    return {"score": round(norm, 2), "label": label,
            "drivers": drivers[:6]}


# ===== quant/scanner.py =====
"""Daily setup scanner — the whole universe through the Playbook gates.

Answers the desk's morning question: "what is tradeable TODAY?"
Every ticker gets the 5-gate check + dip-setup check; output is ranked by
actionability. This is the 'few trades each day, from all the data' engine.
"""


import numpy as np
import pandas as pd





RISK_PROFILES = {
    "🛡️ Conservative": {"risk_pct": 1.0, "max_pos": 3, "heat_cap": 3.0,
                        "conviction_min": 55},
    "⚖️ Balanced": {"risk_pct": 1.5, "max_pos": 4, "heat_cap": 5.0,
                    "conviction_min": 50},
    "🔥 Aggressive": {"risk_pct": 2.0, "max_pos": 6, "heat_cap": 8.0,
                      "conviction_min": 45},
}


def scan_setups(data: dict[str, pd.DataFrame], account: float = 5000.0,
                risk_pct: float = 1.0) -> pd.DataFrame:
    """Light playbook pass on every ticker. Fast: no MC, no backtests."""
    rows = []
    for tkr, df in data.items():
        try:
            if len(df) < 220:
                continue
            c = df["Close"]
            price = float(c.iloc[-1])
            a = float(atr(df).iloc[-1])
            if a <= 0:
                continue
            comp = composite(df)
            sig = str(comp["signal"].iloc[-1])
            score = float(comp["score"].iloc[-1])
            bx = bxtrender(df).iloc[-1]
            s200 = float(sma(c, 200).iloc[-1])
            r2 = float(rsi(c, 2).iloc[-1])
            reg = regime_quadrant(df)

            g = [price > s200,
                 sig == "BUY",
                 float(bx["long_osc"]) > 0,
                 bool(bx["t3_rising"]),
                 "Storm" not in reg["regime"]]
            greens = sum(g)
            dip = price > s200 and r2 < 10

            if greens == 5:
                setup, urgency, rank = "TREND ENTRY", "🟢 ENTER", 0
            elif dip:
                setup, urgency, rank = "DIP SCALP", "🟡 FAST", 1
            elif greens == 4:
                setup, urgency, rank = "1 gate away", "👀 STALK", 2
            else:
                continue                      # not actionable today

            stop = price - 2.5 * a
            shares = int((account * risk_pct / 100) / (2.5 * a))
            rows.append({
                "ticker": tkr, "setup": setup, "urgency": urgency,
                "price": round(price, 2), "stop": round(stop, 2),
                "shares": shares,
                "cost $": round(shares * price, 0),
                "risk $": round(shares * 2.5 * a, 0),
                "score": round(score, 2),
                "BX": f"{bx['long_osc']:+.0f}{'↑' if bx['t3_rising'] else '↓'}",
                "RSI2": round(r2),
                "gates": f"{greens}/5",
                "_rank": rank,
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["_rank", "score"],
                              ascending=[True, False]).drop(columns="_rank")
    return out.reset_index(drop=True)


# ===== quant/playbook.py =====
"""Playbook — the WHEN engine. One panel that answers, at all times:
enter now? wait for what? holding — do what? exit now — why?

It runs a checklist of gates (the same ones the backtest engine trades),
shows which are green and which are blocking, and produces ONE instruction
with an urgency level. This is the terminal's 'what do I do' function.
"""


import numpy as np
import pandas as pd







def build_playbook(df: pd.DataFrame, account: float = 5000.0,
                   risk_pct: float = 1.0,
                   in_position: bool = False,
                   entry: float | None = None,
                   stop: float | None = None) -> dict:
    c = df["Close"]
    price = float(c.iloc[-1])
    a = float(atr(df).iloc[-1])
    comp = composite(df)
    score = float(comp["score"].iloc[-1])
    sig = str(comp["signal"].iloc[-1])
    bx = bxtrender(df).iloc[-1]
    s200 = float(sma(c, 200).iloc[-1]) if len(c) >= 200 else np.nan
    r2 = float(rsi(c, 2).iloc[-1])
    reg = regime_quadrant(df)
    fib = fib_levels(df)
    sr = support_resistance(df)

    # ---------------- ENTRY GATES ----------------
    gates = [
        ("Price above 200-bar average (regime)",
         not np.isnan(s200) and price > s200,
         f"price ${price:,.2f} vs ${s200:,.2f}" if not np.isnan(s200) else "n/a"),
        ("Composite signal is BUY",
         sig == "BUY", f"score {score:+.2f} → {sig}"),
        ("B-Xtrender long oscillator positive",
         float(bx["long_osc"]) > 0, f"{bx['long_osc']:+.0f}"),
        ("B-Xtrender T3 rising",
         bool(bx["t3_rising"]), "rising" if bx["t3_rising"] else "falling"),
        ("Volatility regime not a storm",
         "Storm" not in reg["regime"], reg["regime"]),
    ]
    dip_setup = (not np.isnan(s200) and price > s200 and r2 < 10)
    greens = sum(1 for _, ok, _ in gates if ok)

    # levels for a fresh entry
    stop_new = price - 2.5 * a
    shares = int((account * risk_pct / 100) / (2.5 * a)) if a > 0 else 0
    scale1 = price + 2.5 * a
    scale2 = price + 5.0 * a

    # ---------------- DECISION ----------------
    if in_position and entry and stop:
        r_unit = abs(entry - stop) if abs(entry - stop) > 1e-9 else 2.5 * a
        r_now = (price - entry) / r_unit
        trail = price - 2.5 * a
        actions = []
        urgency = "🟢 CALM"
        if price <= stop:
            instruction = f"EXIT NOW — stop ${stop:,.2f} violated"
            urgency = "🔴 IMMEDIATE"
        elif sig == "SELL":
            instruction = "EXIT — composite flipped to SELL"
            urgency = "🟠 TODAY"
        elif float(bx["long_osc"]) < 0 and not bx["t3_rising"]:
            instruction = ("TIGHTEN — B-X turned fully negative; "
                           f"raise stop to ${max(stop, trail):,.2f}")
            urgency = "🟠 TODAY"
        elif r_now >= 2 :
            instruction = (f"SCALE — trade is +{r_now:.1f}R: bank ⅓, "
                           f"stop to ${entry + r_unit:,.2f} (entry+1R)")
            urgency = "🟡 SOON"
        elif r_now >= 1:
            instruction = (f"PROTECT — +{r_now:.1f}R: stop to breakeven "
                           f"${entry:,.2f}; consider first scale at "
                           f"${entry + 2 * r_unit:,.2f}")
            urgency = "🟡 SOON"
        else:
            instruction = (f"HOLD — {r_now:+.1f}R · stop ${stop:,.2f} · "
                           f"let the setup work")
        return {"mode": "MANAGE", "instruction": instruction,
                "urgency": urgency, "r_now": round(r_now, 2),
                "gates": gates, "greens": greens,
                "trail_suggestion": round(trail, 2),
                "regime": reg["regime"]}

    # flat: enter, stalk or stand down
    if greens == 5:
        instruction = (f"ENTER — all 5 gates green: buy {shares} shares "
                       f"≈ ${price:,.2f}, stop ${stop_new:,.2f}, "
                       f"scale ⅓ at ${scale1:,.2f} and ${scale2:,.2f}")
        urgency = "🟢 ACTIONABLE"
    elif dip_setup:
        pocket = (fib["levels"]["0.618"] if fib else None)
        instruction = (f"DIP SETUP — RSI2={r2:.0f} panic in an uptrend: "
                       f"scalp entry ≈ ${price:,.2f}, stop ${stop_new:,.2f},"
                       f" exit on RSI2 > 65"
                       + (f" · golden pocket ${pocket:,.2f}" if pocket else ""))
        urgency = "🟡 FAST SETUP"
    elif greens >= 3:
        missing = [name for name, ok, _ in gates if not ok]
        instruction = ("STALK — close but blocked by: " +
                       "; ".join(missing[:2]) +
                       ". Set an alert, don't force it.")
        urgency = "🟡 WATCH"
    else:
        instruction = (f"STAND DOWN — only {greens}/5 gates green. "
                       "No setup exists; capital preservation is the trade.")
        urgency = "⚪ NO TRADE"

    nearest_sup = max((lv["price"] for lv in sr if lv["price"] < price),
                      default=None)
    nearest_res = min((lv["price"] for lv in sr if lv["price"] > price),
                      default=None)
    return {"mode": "ENTRY", "instruction": instruction, "urgency": urgency,
            "gates": gates, "greens": greens,
            "plan": {"shares": shares, "entry": round(price, 2),
                     "stop": round(stop_new, 2),
                     "scale1": round(scale1, 2), "scale2": round(scale2, 2)},
            "nearest_support": nearest_sup, "nearest_resistance": nearest_res,
            "regime": reg["regime"]}


# ===== quant/bxlab.py =====
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


import numpy as np
import pandas as pd



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


# ===== quant/runner.py =====
"""RUNNER — the trade lifecycle machine.

Replays a ticker bar-by-bar and manages positions the way a desk does,
logging EVERY event with its reason and the model state at that moment:

  ENTRY       — trend (composite BUY + B-X confirm + regime gate) or
                dip (RSI2 panic inside the fib pocket of an uptrend);
                auto-picked per ticker by Hurst.
  SCALE 1/3   — at +1R: take a third off, stop jumps to breakeven.
  SCALE 2/3   — at +2R: take another third, stop jumps to entry +1R.
  TRAIL       — the runner rides a 2.5xATR chandelier under the rest.
  STOP / BE   — mechanical, first touch.
  TIME EXIT   — unprofitable and stale -> out.
  SIGNAL EXIT — the composite flips to SELL.

Ends with the CURRENT live state: in/out, remaining size, active stop,
and tomorrow's order. This is the machine actually running the models.
"""


import numpy as np
import pandas as pd






def run_machine(df: pd.DataFrame, account: float = 5000.0,
                risk_pct: float = 1.0, atr_mult: float = 2.5,
                mode: str = "auto", commission: float = 0.001) -> dict:
    if len(df) < 300:
        return {"error": "Need at least ~300 bars."}

    if mode == "auto":
        h = _hurst_quick(df["Close"])
        mode = "trend" if h >= 0.5 else "dip"

    comp = composite(df)
    bx = bxtrender(df)
    a = atr(df)
    s200 = sma(df["Close"], 200)
    r2 = rsi(df["Close"], 2)

    roll_hi = df["High"].rolling(126, min_periods=40).max()
    roll_lo = df["Low"].rolling(126, min_periods=40).min()
    rng_ = roll_hi - roll_lo
    fib_lo = (roll_hi - 0.786 * rng_).values
    fib_hi = (roll_hi - 0.382 * rng_).values

    o_, h_, l_, c_ = (df[k].values for k in ("Open", "High", "Low", "Close"))
    sig = comp["signal"].values
    score = comp["score"].values
    bxl = bx["long_osc"].values
    bxr = bx["t3_rising"].values
    bxbt = bx["buy_turn"].values
    av = a.values
    s2v = s200.values
    r2v = r2.values
    idx = df.index

    cash = account
    sh = 0.0                    # current shares
    sh0 = 0.0                   # original size
    entry_px = stop = r_unit = 0.0
    entry_i = 0
    scaled1 = scaled2 = False
    events: list[dict] = []
    equity = []

    closures: list[dict] = []
    entry_ctx: dict = {}

    def log(i, ev, px, qty, note):
        events.append({
            "date": idx[i].date(), "event": ev, "price": round(float(px), 2),
            "shares": int(qty), "position_after": int(sh),
            "equity": round(cash + sh * c_[i], 0),
            "score": round(float(score[i - 1]), 2),
            "bx": f"{bxl[i-1]:+.0f}{'↑' if bxr[i-1] else '↓'}",
            "rsi2": round(float(r2.values[i - 1])),
            "note": note,
        })

    def sell(i, qty, px, ev, note):
        nonlocal cash, sh
        qty = min(qty, sh)
        if qty <= 0:
            return
        cash += qty * px * (1 - commission)
        sh -= qty
        if entry_ctx:
            entry_ctx["realized"] += qty * (px - entry_px)
        log(i, ev, px, qty, note)
        if sh == 0 and entry_ctx:
            invested = entry_ctx["invested"]
            pnl = entry_ctx["realized"]
            closures.append({
                "opened": idx[entry_i].date(), "closed": idx[i].date(),
                "bars": int(i - entry_i),
                "entry $": round(entry_px, 2),
                "final exit $": round(px, 2),
                "price chg %": round((px / entry_px - 1) * 100, 2),
                "capital in $": round(invested, 0),
                "P&L $": round(pnl, 0),
                "return on capital %": round(pnl / invested * 100, 2)
                if invested else 0,
                "R": round(pnl / (sh0 * r_unit), 2) if r_unit > 0 else 0,
                "exit reason": ev,
                "entry score": round(entry_ctx["score"], 2),
                "entry BX": round(entry_ctx["bx_long"], 0),
                "entry RSI2": round(entry_ctx["rsi2"], 0),
            })

    for i in range(1, len(df)):
        o, hi, lo = o_[i], h_[i], l_[i]
        above200 = not np.isnan(s2v[i - 1]) and c_[i - 1] > s2v[i - 1]

        # ---------- manage open position ----------
        if sh > 0:
            bars_in = i - entry_i
            # scale-outs (checked before stop so profit-taking wins the day)
            if not scaled1 and hi >= entry_px + 1.0 * r_unit:
                px = max(o, entry_px + 1.0 * r_unit)
                sell(i, round(sh0 / 3), px, "SCALE 1/3 (+1R)",
                     "stop → breakeven")
                stop = max(stop, entry_px)
                scaled1 = True
            if sh > 0 and not scaled2 and hi >= entry_px + 2.0 * r_unit:
                px = max(o, entry_px + 2.0 * r_unit)
                sell(i, round(sh0 / 3), px, "SCALE 2/3 (+2R)",
                     "stop → entry +1R")
                stop = max(stop, entry_px + 1.0 * r_unit)
                scaled2 = True
            # trail on the remainder (trend mode)
            if sh > 0 and mode == "trend":
                new_trail = hi - atr_mult * av[i - 1]
                if new_trail > stop:
                    stop = new_trail
            # stop check
            if sh > 0 and lo <= stop:
                px = min(o, stop) if o > stop else o
                tag = ("BREAKEVEN STOP" if abs(stop - entry_px) < 1e-9
                       else "TRAIL STOP" if scaled1 else "STOP LOSS")
                sell(i, sh, max(px, 0.01), tag,
                     f"after {bars_in} bars")
                scaled1 = scaled2 = False
            # signal / rsi / time exits
            elif sh > 0 and mode == "trend" and sig[i - 1] == "SELL":
                sell(i, sh, o, "SIGNAL EXIT", "composite flipped to SELL")
                scaled1 = scaled2 = False
            elif sh > 0 and mode == "dip" and r2v[i - 1] > 65:
                sell(i, sh, o, "TARGET (RSI snap-back)",
                     f"RSI2={r2v[i-1]:.0f}")
                scaled1 = scaled2 = False
            elif sh > 0 and bars_in >= (20 if mode == "trend" else 10) \
                    and c_[i - 1] < entry_px:
                sell(i, sh, o, "TIME EXIT", "stale & unprofitable")
                scaled1 = scaled2 = False

        # ---------- entries ----------
        if sh == 0 and av[i - 1] > 0:
            enter = False
            why = ""
            if mode == "trend":
                if above200 and sig[i - 1] == "BUY" and bxl[i - 1] > 0 \
                        and bxr[i - 1]:
                    enter = True
                    why = (f"composite {score[i-1]:+.2f} BUY · B-X "
                           f"{bxl[i-1]:+.0f}↑ · above 200-SMA")
            else:
                pocket = (not np.isnan(fib_lo[i - 1]) and
                          fib_lo[i - 1] <= c_[i - 1] <= fib_hi[i - 1])
                if above200 and r2v[i - 1] < 10 and (pocket or bxbt[i - 1]):
                    enter = True
                    why = (f"RSI2={r2v[i-1]:.0f} panic · "
                           f"{'fib pocket' if pocket else 'B-X buy turn'} "
                           f"· uptrend intact")
            if enter:
                stop_d = atr_mult * av[i - 1]
                qty = int(min((cash * risk_pct / 100) / stop_d,
                              cash / (o * (1 + commission))))
                if qty * o >= 100:
                    cash -= qty * o * (1 + commission)
                    sh = sh0 = float(qty)
                    entry_px, entry_i = o, i
                    r_unit = stop_d
                    stop = o - stop_d
                    scaled1 = scaled2 = False
                    entry_ctx = {"score": float(score[i - 1]),
                                 "bx_long": float(bxl[i - 1]),
                                 "rsi2": float(r2.values[i - 1]),
                                 "invested": qty * o,
                                 "realized": 0.0}
                    log(i, f"ENTRY ({mode.upper()})", o, qty,
                        why + f" · stop ${stop:,.2f} · invested "
                        f"${qty * o:,.0f} ({qty * o / account * 100:.0f}% "
                        f"of account)")

        equity.append(cash + sh * c_[i])

    eq = pd.Series(equity, index=idx[1:])
    ev_df = pd.DataFrame(events)

    # ---- closed-trade stats from the event log ----
    realized = ev_df[ev_df["event"].str.contains(
        "STOP|EXIT|TARGET|SCALE")] if len(ev_df) else pd.DataFrame()

    # ---- live state ----
    if sh > 0:
        state = {
            "in_position": True,
            "shares": int(sh),
            "entry": round(entry_px, 2),
            "stop_now": round(stop, 2),
            "bars_held": int(len(df) - 1 - entry_i),
            "unrealized": round((c_[-1] - entry_px) * sh, 0),
            "scaled": f"{'⅓ taken' if scaled1 else 'full size'}"
                      f"{' + ⅔ taken' if scaled2 else ''}",
            "tomorrow": (f"HOLD {int(sh)} shares · stop at "
                         f"${stop:,.2f} · next scale at "
                         f"${entry_px + (2 if scaled1 else 1) * r_unit:,.2f}"),
        }
    else:
        nxt = "watch for "
        nxt += ("composite BUY + B-X rising above the 200-SMA"
                if mode == "trend" else
                "RSI2 < 10 inside the fib pocket of an uptrend")
        state = {"in_position": False, "tomorrow": f"FLAT · {nxt}"}

    r_ = eq.pct_change().dropna()
    n_years = max(len(eq) / 252, 1e-9)
    stats = {
        "Mode": mode.upper(),
        "Final equity $": round(float(eq.iloc[-1]), 0),
        "CAGR %": round(((eq.iloc[-1] / account) ** (1 / n_years) - 1) * 100, 1),
        "Sharpe": round(float(r_.mean() / r_.std() * np.sqrt(252)), 2)
        if r_.std() > 0 else 0.0,
        "Max DD %": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
        "Events logged": int(len(ev_df)),
    }
    closed_df = pd.DataFrame(closures)

    # ---- 🎓 self-learning: what did the machine learn on THIS ticker? -----
    lessons: list[str] = []
    if len(closed_df) >= 5:
        by_reason = closed_df.groupby("exit reason")["P&L $"].agg(
            ["count", "sum", "mean"]).sort_values("sum", ascending=False)
        best_r, worst_r = by_reason.index[0], by_reason.index[-1]
        lessons.append(f"Exit analysis: '{best_r}' exits made the most "
                       f"(${by_reason.loc[best_r,'sum']:,.0f} over "
                       f"{int(by_reason.loc[best_r,'count'])} trades); "
                       f"'{worst_r}' cost the most "
                       f"(${by_reason.loc[worst_r,'sum']:,.0f}).")
        hiBX = closed_df[closed_df["entry BX"] >= 25]
        loBX = closed_df[closed_df["entry BX"] < 25]
        if len(hiBX) >= 3 and len(loBX) >= 3:
            lessons.append(f"Entry-quality: trades opened with strong B-X "
                           f"(≥+25) won {float((hiBX['P&L $']>0).mean())*100:.0f}%"
                           f" vs {float((loBX['P&L $']>0).mean())*100:.0f}% "
                           f"with weak B-X → on {'this ticker' } demand the "
                           f"stronger confirmation.")
        wr = float((closed_df["P&L $"] > 0).mean() * 100)
        avg_roc = float(closed_df["return on capital %"].mean())
        lessons.append(f"Overall on this ticker: {len(closed_df)} closed "
                       f"round-trips · {wr:.0f}% won · avg "
                       f"{avg_roc:+.2f}% return on deployed capital per trade.")
        held_w = closed_df[closed_df["P&L $"] > 0]["bars"].mean()
        held_l = closed_df[closed_df["P&L $"] <= 0]["bars"].mean()
        if held_w and held_l:
            lessons.append(f"Time asymmetry: winners held "
                           f"{held_w:.0f} bars vs losers {held_l:.0f} — "
                           + ("healthy (cutting losers fast)."
                              if held_w > held_l else
                              "⚠️ losers held LONGER than winners — the "
                              "classic account-killer; trust the time stop."))
    elif len(closed_df):
        lessons.append(f"Only {len(closed_df)} closed trades — the machine "
                       f"needs ~5+ round-trips before its lessons mean much.")

    return {"events": ev_df, "equity": eq, "state": state, "stats": stats,
            "mode": mode, "closures": closed_df, "lessons": lessons}


# ===== quant/analyst.py =====
"""The Analyst — an autonomous briefing engine, not a chatbot.

One call runs the whole desk and writes the morning note a junior analyst
would hand you: market state, your book, today's setups, what to watch,
the news that touches YOUR names, and a weekly self-audit of the system's
own live edge. Every sentence is generated by rules over computed numbers —
nothing is ever 'imagined'.
"""


from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf








# ---------------------------------------------------------------------------
# News (per-ticker, shape-defensive across yfinance versions)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def ticker_news(tickers: tuple[str, ...], per_ticker: int = 3) -> pd.DataFrame:
    rows = []
    for t in tickers[:8]:                       # politeness cap
        try:
            items = yf.Ticker(t).news or []
        except Exception:
            continue
        for it in items[:per_ticker]:
            try:
                content = it.get("content", it)
                title = content.get("title") or it.get("title")
                if not title:
                    continue
                url = (content.get("canonicalUrl", {}) or {}).get("url") \
                    or it.get("link", "")
                pub = (content.get("provider", {}) or {}).get("displayName") \
                    or it.get("publisher", "")
                when = content.get("pubDate") or ""
                rows.append({"ticker": t, "headline": title,
                             "source": pub, "when": str(when)[:16],
                             "url": url})
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-audit — the system grades ITSELF on the live track record
# ---------------------------------------------------------------------------

def self_audit(blotter: pd.DataFrame, account: float) -> list[str]:
    notes = []
    closed = blotter[blotter["status"] == "CLOSED"] if len(blotter) else \
        pd.DataFrame()
    if len(closed) < 8:
        notes.append(f"📋 Track record: {len(closed)} closed trades — "
                     f"need ≥8 before the statistics mean anything. "
                     f"Keep recording every plan.")
        return notes
    hit = float((closed["P&L $"] > 0).mean() * 100)
    exp = float(closed["P&L $"].mean())
    notes.append(f"📋 Live record: {len(closed)} closed · hit rate "
                 f"{hit:.0f}% · expectancy ${exp:+,.0f}/trade.")
    bs = bootstrap_cagr(closed["P&L $"], starting=account)
    if "error" not in bs:
        notes.append(f"🔬 Bootstrap 90% CI on your LIVE trades: "
                     f"{bs['CI90_low_%']}% … {bs['CI90_high_%']}% — "
                     f"{bs['verdict']}")
        if not bs["excludes_zero"]:
            notes.append("⚠️ The live edge is NOT yet statistically proven "
                         "— trade the minimum size until the CI clears zero.")
    return notes


# ---------------------------------------------------------------------------
# The briefing
# ---------------------------------------------------------------------------

def morning_briefing(spy: pd.DataFrame,
                     universe_data: dict[str, pd.DataFrame],
                     blotter: pd.DataFrame | None,
                     account: float,
                     risk_pct: float,
                     event_gauge: dict | None = None) -> dict:
    """Run the desk, return a structured briefing with plain-language notes."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    brief = {"stamp": now, "market": [], "book": [], "setups": [],
             "watch": [], "audit": [], "setups_table": pd.DataFrame(),
             "news_tickers": []}

    # ---- 1. Market ---------------------------------------------------------
    reg = regime_quadrant(spy)
    brief["regime"] = reg
    brief["market"].append(f"Market regime: **{reg['regime']}** — "
                           f"{reg['playbook']}")
    if event_gauge:
        brief["market"].append(f"Macro event gauge: **{event_gauge['label']}**"
                               f" (score {event_gauge['score']:+.2f}).")

    # ---- 2. Your book -------------------------------------------------------
    open_pos = (blotter[blotter["status"] == "OPEN"]
                if blotter is not None and len(blotter) else pd.DataFrame())
    if len(open_pos):
        poss, rets_map = [], {}
        for _, r in open_pos.iterrows():
            t = r["ticker"]
            df_t = universe_data.get(t)
            if df_t is None or len(df_t) < 220:
                continue
            pb = build_playbook(df_t, account=account, risk_pct=risk_pct,
                                in_position=True, entry=float(r["entry"]),
                                stop=float(r["stop"]))
            brief["book"].append(f"**{t}** ({int(r['shares'])} sh, "
                                 f"{r['P&L %']:+.1f}%): {pb['urgency']} — "
                                 f"{pb['instruction']}")
            poss.append({"ticker": t, "shares": int(r["shares"]),
                         "entry": float(r["entry"]), "stop": float(r["stop"])})
            rets_map[t] = df_t["Close"].pct_change().dropna()
        pv = portfolio_var(poss, rets_map, account) if poss else {}
        ch = correlation_heat(poss, rets_map, account) if len(poss) > 1 else {}
        if pv:
            brief["book"].append(f"Book risk: 1-day VaR ${pv['VaR_$']:,.0f} "
                                 f"({pv['VaR_%']}%) · gross exposure "
                                 f"{pv['gross_exposure_%']}%.")
        if ch and ch.get("warning"):
            brief["book"].append(f"⚠️ Correlation alert: avg pairwise "
                                 f"{ch['avg_correlation']} — your positions "
                                 f"are effectively one trade.")
    else:
        brief["book"].append("No open positions — full dry powder.")

    # ---- 3. Today's setups ---------------------------------------------------
    setups = scan_setups(universe_data, account=account, risk_pct=risk_pct)
    brief["setups_table"] = setups
    if len(setups):
        enters = setups[setups["urgency"] == "🟢 ENTER"]
        fast = setups[setups["urgency"] == "🟡 FAST"]
        if len(enters):
            top = enters.iloc[0]
            brief["setups"].append(f"**{len(enters)} full entries** today — "
                                   f"strongest: **{top['ticker']}** "
                                   f"(score {top['score']}, BX {top['BX']}): "
                                   f"{top['shares']} sh ≈ ${top['price']:,.2f}"
                                   f", stop ${top['stop']:,.2f}.")
        if len(fast):
            brief["setups"].append(f"**{len(fast)} dip scalp(s)**: "
                                   + ", ".join(fast["ticker"].tolist()) +
                                   " — RSI2 panic inside uptrends; quick "
                                   "in/out on the snap-back.")
        stalk = setups[setups["urgency"] == "👀 STALK"]
        for _, s_ in stalk.head(3).iterrows():
            brief["watch"].append(f"**{s_['ticker']}** is one gate away "
                                  f"({s_['gates']}) with score {s_['score']}"
                                  f" — set an alert near ${s_['price']:,.2f}.")
    else:
        brief["setups"].append("Zero setups pass the gates today. The "
                               "correct trade is patience.")

    # news targets: holdings + top setups
    news_t = list(open_pos["ticker"]) if len(open_pos) else []
    if len(setups):
        news_t += setups["ticker"].head(5).tolist()
    brief["news_tickers"] = tuple(dict.fromkeys(news_t))[:8]

    # ---- 4. Self-audit ---------------------------------------------------------
    if blotter is not None and len(blotter):
        brief["audit"] = self_audit(blotter, account)
    return brief


# ===== quant/autotrader.py =====
"""AutoTrader — the bot that lives inside the website.

Architecture (honest by design):
  * The bot holds a MANDATE (universe, risk profile, caps) and a STATE file
    (cash, positions, orders, decision log) persisted like the journal.
  * Every activation ("tick") it CATCHES UP: for each day since its last
    tick it replays the mechanical rules on real bars — pending entries
    fill at the open, stops/scales/exits execute where price actually
    touched them. Watching live and catching up produce IDENTICAL results
    because the rules are bar-mechanical with next-open execution.
  * Every decision is logged with the model state and the reason — the
    same narration standard as the Runner.
  * 100% paper: real prices, fake money, building the record that decides
    whether this logic ever deserves a real broker API.
"""


import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd






BOT_PATH = "data/autotrader.json"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def default_state(account: float = 5000.0) -> dict:
    return {"enabled": False, "cash": account, "start_equity": account,
            "created": None, "last_tick": None,
            "mandate": {"risk_pct": 1.5, "max_positions": 4,
                        "universe": [], "atr_mult": 2.5},
            "positions": {},          # tkr -> {shares, entry, stop, entry_date, scaled1, scaled2, r_unit, sh0}
            "pending": [],            # orders for next open
            "log": [],                # decision feed
            "closed": []}             # deal ledger


def load_bot(path: str = BOT_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default_state()


def save_bot(state: dict, path: str = BOT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1, default=str)


def _log(state: dict, date, action: str, tkr: str, px: float, qty: int,
         why: str):
    state["log"].append({
        "date": str(date)[:10], "action": action, "ticker": tkr,
        "price": round(float(px), 2), "shares": int(qty),
        "cash": round(state["cash"], 0), "why": why,
        "stamp": datetime.now(timezone.utc).strftime("%H:%M UTC")})
    state["log"] = state["log"][-400:]


# ---------------------------------------------------------------------------
# The tick — catch-up + decide
# ---------------------------------------------------------------------------

def bot_tick(state: dict, data: dict[str, pd.DataFrame],
             spy: pd.DataFrame) -> dict:
    """Advance the bot through every unprocessed bar, then stage tomorrow."""
    if not state["enabled"]:
        return state
    md = state["mandate"]
    atr_mult = md.get("atr_mult", 2.5)

    # figure the calendar of unprocessed sessions from SPY's index
    last = pd.Timestamp(state["last_tick"]) if state["last_tick"] else None
    sessions = [d for d in spy.index if last is None or d > last]
    if not sessions:
        return state

    for day in sessions:
        # ---- 1. fill pending entries at today's open --------------------
        still_pending = []
        for od in state["pending"]:
            t = od["ticker"]
            df = data.get(t)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            o = float(row["Open"])
            cost = od["shares"] * o * 1.001
            if od["shares"] < 1 or cost > state["cash"]:
                continue
            if len(state["positions"]) >= md["max_positions"]:
                continue
            state["cash"] -= cost
            r_unit = atr_mult * od["atr"]
            state["positions"][t] = {
                "shares": od["shares"], "sh0": od["shares"], "entry": o,
                "stop": o - r_unit, "r_unit": r_unit,
                "entry_date": str(day)[:10],
                "scaled1": False, "scaled2": False,
                "invested": cost, "realized": 0.0,
                "why": od["why"]}
            _log(state, day, "🟢 BUY", t, o, od["shares"],
                 od["why"] + f" · invested ${cost:,.0f}")
        state["pending"] = still_pending

        # ---- 2. manage open positions on today's bar --------------------
        for t in list(state["positions"].keys()):
            p = state["positions"][t]
            df = data.get(t)
            if df is None or day not in df.index:
                continue
            row = df.loc[day]
            o, hi, lo = float(row["Open"]), float(row["High"]), float(row["Low"])
            i = df.index.get_loc(day)
            if i < 1:
                continue
            prev = df.iloc[i - 1]

            def sell(qty, px, tag, why):
                qty = min(qty, p["shares"])
                if qty <= 0:
                    return
                state["cash"] += qty * px * 0.999
                p["realized"] += qty * (px - p["entry"])
                p["shares"] -= qty
                _log(state, day, tag, t, px, qty, why)
                if p["shares"] == 0:
                    inv = p["invested"]
                    state["closed"].append({
                        "ticker": t, "opened": p["entry_date"],
                        "closed": str(day)[:10],
                        "entry $": round(p["entry"], 2),
                        "exit $": round(px, 2),
                        "chg %": round((px / p["entry"] - 1) * 100, 2),
                        "capital $": round(inv, 0),
                        "P&L $": round(p["realized"], 0),
                        "ROC %": round(p["realized"] / inv * 100, 2),
                        "reason": tag, "why_entered": p["why"]})
                    del state["positions"][t]

            # scale-outs
            if t in state["positions"] and not p["scaled1"] and \
                    hi >= p["entry"] + p["r_unit"]:
                px = max(o, p["entry"] + p["r_unit"])
                sell(round(p["sh0"] / 3), px, "💰 SCALE ⅓",
                     "+1R reached → bank a third, stop → breakeven")
                if t in state["positions"]:
                    p["stop"] = max(p["stop"], p["entry"])
                    p["scaled1"] = True
            if t in state["positions"] and p["scaled1"] and not p["scaled2"] \
                    and hi >= p["entry"] + 2 * p["r_unit"]:
                px = max(o, p["entry"] + 2 * p["r_unit"])
                sell(round(p["sh0"] / 3), px, "💰 SCALE ⅔",
                     "+2R reached → bank another third, stop → entry+1R")
                if t in state["positions"]:
                    p["stop"] = max(p["stop"], p["entry"] + p["r_unit"])
                    p["scaled2"] = True
            # trail
            if t in state["positions"]:
                prev_atr = float(atr(df.iloc[:i]).iloc[-1]) if i > 15 else 0
                if prev_atr > 0:
                    p["stop"] = max(p["stop"], hi - atr_mult * prev_atr)
            # stop
            if t in state["positions"] and lo <= p["stop"]:
                px = min(o, p["stop"]) if o > p["stop"] else o
                tag = "🟡 BREAKEVEN" if abs(p["stop"] - p["entry"]) < 1e-9 \
                    else "🛑 STOP"
                sell(p["shares"], max(px, 0.01), tag,
                     f"stop ${p['stop']:,.2f} touched (low ${lo:,.2f})")

        state["last_tick"] = str(day)[:10]

    # ---- 3. stage tomorrow's entries from today's setups ------------------
    if len(state["positions"]) < md["max_positions"]:
        setups = scan_setups(
            {t: d for t, d in data.items()
             if t not in state["positions"]},
            account=state["cash"] + sum(
                p["shares"] * p["entry"] for p in state["positions"].values()),
            risk_pct=md["risk_pct"])
        state["pending"] = []
        if len(setups):
            take = setups[setups["urgency"] == "🟢 ENTER"].head(
                md["max_positions"] - len(state["positions"]))
            for _, s in take.iterrows():
                df_t = data.get(s["ticker"])
                a_ = float(atr(df_t).iloc[-1]) if df_t is not None else 0
                if a_ <= 0:
                    continue
                state["pending"].append({
                    "ticker": s["ticker"], "shares": int(s["shares"]),
                    "atr": a_,
                    "why": (f"setup {s['setup']} · score {s['score']} · "
                            f"BX {s['BX']} · {s['gates']} gates")})
                _log(state, state["last_tick"], "📋 ORDER STAGED",
                     s["ticker"], s["price"], int(s["shares"]),
                     f"for next open: {s['setup']} (score {s['score']})")
    return state


def bot_equity(state: dict, data: dict[str, pd.DataFrame]) -> dict:
    """Mark the bot's book to the latest closes."""
    pos_val = 0.0
    rows = []
    for t, p in state["positions"].items():
        df = data.get(t)
        px = float(df["Close"].iloc[-1]) if df is not None and len(df) else \
            p["entry"]
        pos_val += p["shares"] * px
        rows.append({"ticker": t, "shares": p["shares"],
                     "entry": round(p["entry"], 2), "mark": round(px, 2),
                     "stop": round(p["stop"], 2),
                     "P&L $": round((px - p["entry"]) * p["shares"], 0),
                     "opened": p["entry_date"]})
    equity = state["cash"] + pos_val
    closed = pd.DataFrame(state["closed"])
    realized = float(closed["P&L $"].sum()) if len(closed) else 0.0
    return {"equity": round(equity, 0),
            "cash": round(state["cash"], 0),
            "return_pct": round((equity / state["start_equity"] - 1) * 100, 2),
            "open_table": pd.DataFrame(rows),
            "closed_table": closed,
            "realized": realized,
            "n_wins": int((closed["P&L $"] > 0).sum()) if len(closed) else 0}



# ===== MAIN APP =====
"""QuantSignal — neon-terminal quant desk for US stocks."""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st































st.set_page_config(page_title="QuantSignal", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# NEON TERMINAL styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(16,185,129,.13), transparent),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(59,130,246,.07), transparent),
    linear-gradient(180deg, #070b10 0%, #0b0f14 100%);
}
.stApp::before {
  content:""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(16,185,129,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(16,185,129,.03) 1px, transparent 1px);
  background-size: 42px 42px;
}

.hero-title {
  font-size: 3.2rem; font-weight: 900; letter-spacing: -2px; margin: 0;
  background: linear-gradient(90deg, #10b981 0%, #34d399 35%, #22d3ee 75%, #60a5fa 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 24px rgba(16,185,129,.35));
}
.hero-sub { color: #8b98a5; font-size: .95rem; margin-top: 4px; }
.chips { margin-top: 10px; }
.chip {
  display:inline-block; padding: 4px 12px; margin: 0 6px 6px 0;
  border-radius: 999px; font-size: .75rem; font-weight: 600;
  color: #6ee7b7; background: rgba(16,185,129,.08);
  border: 1px solid rgba(16,185,129,.35);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 6px; background: rgba(19,26,34,.6); padding: 6px;
  border-radius: 14px; border: 1px solid #1f2a36;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px; padding: 8px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(16,185,129,.25), rgba(34,211,238,.12)) !important;
  border: 1px solid rgba(16,185,129,.5);
  box-shadow: 0 0 18px rgba(16,185,129,.25);
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(23,32,42,.85) 0%, rgba(14,20,27,.9) 100%);
  border: 1px solid #1f2a36; border-radius: 16px; padding: 14px 16px;
  backdrop-filter: blur(8px); transition: all .25s ease;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(16,185,129,.6);
  box-shadow: 0 0 22px rgba(16,185,129,.18); transform: translateY(-2px);
}
[data-testid="stMetricValue"] { font-weight: 800; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { color: #8b98a5; }

div[data-testid="stExpander"] {
  border: 1px dashed rgba(16,185,129,.35); border-radius: 14px;
  background: rgba(15,21,29,.7);
}

.verdict {
  border-radius: 20px; padding: 30px 34px; margin: 10px 0 18px 0;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 10px 44px rgba(0,0,0,.45); backdrop-filter: blur(6px);
}
.verdict h2 { margin: 0; font-size: 2.5rem; font-weight: 900; letter-spacing: -1px; }
.verdict .sub { opacity: .85; font-size: .95rem; margin-top: 6px; }
.v-long  { background: linear-gradient(135deg, rgba(5,59,45,.95), rgba(6,95,70,.9) 60%, rgba(4,120,87,.85));
           border: 1px solid #10b981; box-shadow: 0 0 46px rgba(16,185,129,.30); }
.v-short { background: linear-gradient(135deg, rgba(95,20,20,.95), rgba(153,27,27,.9) 60%, rgba(185,28,28,.85));
           border: 1px solid #ef4444; box-shadow: 0 0 46px rgba(239,68,68,.30); }
.v-none  { background: linear-gradient(135deg, rgba(26,35,50,.95), rgba(43,54,72,.9) 60%, rgba(55,65,81,.85));
           border: 1px solid #6b7280; }

.reason-pro, .reason-con {
  border-radius: 10px; padding: 9px 14px; margin: 6px 0; font-size: .92rem;
}
.reason-pro { background: rgba(11,46,34,.8); border-left: 3px solid #10b981; }
.reason-con { background: rgba(46,15,15,.8); border-left: 3px solid #ef4444; }

.conv-wrap { background:#1f2a36; border-radius: 8px; height: 16px; width: 100%; overflow:hidden; }
.conv-bar  { height: 16px; border-radius: 8px; position: relative; }
.conv-bar::after {
  content:""; position:absolute; inset:0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent);
  animation: shimmer 2.2s infinite; transform: translateX(-100%);
}
@keyframes shimmer { 100% { transform: translateX(100%); } }

.regime-badge {
  display:inline-block; padding: 8px 18px; border-radius: 12px;
  font-weight: 800; font-size: 1.05rem; letter-spacing: .3px;
  background: rgba(19,26,34,.9); border: 1px solid #2a3644;
}
hr { border-color: #1f2a36; }

/* ---- terminal touches ---- */
.tape {
  overflow: hidden; white-space: nowrap; border-top: 1px solid #1f2a36;
  border-bottom: 1px solid #1f2a36; background: #0a0e13;
  font-family: 'JetBrains Mono', monospace; font-size: .85rem;
  padding: 6px 0; margin: 4px 0 10px 0;
}
.tape-inner { display: inline-block; animation: scroll 40s linear infinite; }
@keyframes scroll { 0% {transform: translateX(0)} 100% {transform: translateX(-50%)} }
.tape .up { color: #10b981; } .tape .dn { color: #ef4444; }
.tape .amber { color: #ffb000; font-weight: 700; }
div[data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #1f2a36; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(16,185,129,.5); }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<p class="hero-title">QuantSignal</p>
<p class="hero-sub">Institutional-grade quant desk — every model fused into one decision.
Educational tool, not financial advice.</p>
<div class="chips">
<span class="chip">7-model composite</span><span class="chip">B-Xtrender</span>
<span class="chip">Monte Carlo</span><span class="chip">Kelly sizing</span>
<span class="chip">EWMA vol</span><span class="chip">Fibonacci</span>
<span class="chip">Black-Scholes</span><span class="chip">IV surface</span>
<span class="chip">Max pain</span><span class="chip">Walk-forward</span>
</div>
''', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LIVE mode — one press arms a panel; it then self-refreshes on this pulse
# ---------------------------------------------------------------------------
ms = market_status()
lc1, lc2, lc3 = st.columns([1.1, 1, 3])
live_on = lc1.toggle("🔴 LIVE mode", value=False, key="live_on",
                     help="Armed panels (watchlist, Trade desk, Runner, "
                          "Track record) auto-refresh with live prices.")
live_every = lc2.selectbox("Pulse", [30, 60, 120], index=1,
                           key="live_every",
                           format_func=lambda s: f"every {s}s")
wl_input = lc2.text_input("Watchlist", value="SPY, QQQ, ^VIX",
                          key="wl_input", label_visibility="collapsed",
                          placeholder="Watchlist: SPY, QQQ, ^VIX ...")
st.session_state["watchlist"] = [t.strip().upper() for t in
                                 wl_input.split(",") if t.strip()][:6]
LIVE_EVERY = live_every if live_on else None
lc3.markdown(f"<div class='regime-badge'>{ms['emoji']} {ms['label']} · "
             f"{ms['detail']} · {ms['et_time']}</div>",
             unsafe_allow_html=True)
if live_on:
    st.caption("🔴 Live uses the free Yahoo feed on a shared IP — if data "
               "briefly shows '—' or a rate-limit note, that's throttling, "
               "not a crash. It recovers on the next pulse; a slower pulse "
               "helps.")


@st.fragment(run_every=LIVE_EVERY)
def _watchlist_strip():
    from datetime import datetime as _dt
    syms = st.session_state.get("watchlist", ["SPY", "QQQ", "^VIX"])[:6]
    armed = st.session_state.get("desk_params", {}).get("tkr")
    if armed and armed not in syms:
        syms.append(armed)
    cols = st.columns(len(syms) + 1)
    for col, s in zip(cols, syms):
        q = live_quote(s)
        if q:
            col.metric(s.replace("^", ""), f"{q['price']:,.2f}",
                       delta=f"{q['chg_pct']:+.2f}%")
        else:
            col.metric(s.replace("^", ""), "—")
    cols[-1].caption(f"{'🔴 LIVE' if LIVE_EVERY else '⏸ static'} · "
                     f"updated {_dt.now().strftime('%H:%M:%S')}")
    tape_syms = list(dict.fromkeys(
        st.session_state.get("watchlist", []) +
        ["SPY", "QQQ", "^VIX", "AAPL", "NVDA", "MSFT", "TSLA", "GLD"]))[:10]
    parts = []
    for ts_ in tape_syms:
        tq = live_quote(ts_)
        if tq:
            cls = "up" if tq["chg_pct"] >= 0 else "dn"
            arrow = "▲" if tq["chg_pct"] >= 0 else "▼"
            parts.append(f"<span class='amber'>{ts_.replace('^','')}</span> "
                         f"<span class='{cls}'>{tq['price']:,.2f} {arrow}"
                         f"{abs(tq['chg_pct']):.2f}%</span>")
    if parts:
        line = " &nbsp;·&nbsp; ".join(parts)
        st.markdown(f"<div class='tape'><div class='tape-inner'>{line}"
                    f" &nbsp;·&nbsp; {line}</div></div>",
                    unsafe_allow_html=True)


_watchlist_strip()

st.session_state.setdefault("memory", {})


def _remember(section: str, payload: dict):
    st.session_state["memory"][section] = payload


def _memory_chips():
    mem = st.session_state.get("memory", {})
    if not mem:
        return
    bits = []
    d = mem.get("desk")
    if d:
        bits.append(f"🎯 {d['ticker']}: {d['verdict']} ({d['conviction']})")
    o = mem.get("options")
    if o:
        bits.append(f"🌋 {o['ticker']}: {o.get('vol_state','')[:12]}…")
    e = mem.get("events")
    if e:
        bits.append(f"🌐 macro: {e['label']}")
    bt = mem.get("backtest")
    if bt:
        bits.append(f"🧪 {bt['ticker']}: {bt['mode']} Sharpe {bt['sharpe']}")
    if bits:
        st.caption("🧠 **Session memory (tabs share this):** " +
                   "  ·  ".join(bits))


_memory_chips()

# ---------------------------------------------------------------------------
# ⌨️ Terminal command line — type like a Bloomberg jockey
# ---------------------------------------------------------------------------
cmd = st.text_input("⌨️", placeholder="Command line — try: NVDA GO · AAPL PB · "
                    "TSLA VOL · SCAN", key="cmdline",
                    label_visibility="collapsed")
if cmd:
    parts = cmd.strip().upper().split()
    try:
        if parts[-1] == "GO" and len(parts) == 2:
            _t = parts[0]
            _q = live_quote(_t)
            _d = fetch_history(_t, period="1y")
            if _q and len(_d) > 220:
                _v = analyze(_d)
                cg = st.columns(5)
                cg[0].metric(_t, f"${_q['price']:,.2f}",
                             f"{_q['chg_pct']:+.2f}%")
                cg[1].metric("Verdict", _v["verdict"])
                cg[2].metric("Conviction", _v["conviction"])
                cg[3].metric("Score", f"{_v['score']:+.2f}")
                cg[4].metric("Signal Sharpe", _v["sharpe"])
            else:
                st.warning(f"{_t}: no data (throttled or bad ticker)")
        elif parts[-1] == "PB" and len(parts) == 2:
            _d = fetch_history(parts[0], period="1y")
            if len(_d) > 220:
                _pb = build_playbook(_d)
                st.info(f"**{parts[0]} · {_pb['urgency']}** — "
                        f"{_pb['instruction']}")
        elif parts[-1] == "VOL" and len(parts) == 2:
            _d = fetch_history(parts[0], period="2y")
            _e = ewma_vol(_d); _g = garch_forecast(_d)
            st.info(f"**{parts[0]} vol** — EWMA {_e['sigma_annual_pct']}% ann "
                    f"(±${_e['expected_move_1d']}/day)"
                    + (f" · GARCH {_g['sigma_annual_pct']}%" if _g else ""))
        elif parts[0] == "SCAN":
            st.info("→ open **🧬 Alpha engine** and hit **☀️ Scan today's "
                    "setups** — the full ranked list lives there.")
        else:
            st.caption("Commands: `TICKER GO` quote+verdict · `TICKER PB` "
                       "playbook · `TICKER VOL` volatility · `SCAN`")
    except Exception:
        st.warning("Command failed (data throttled?) — try again.")
st.write("")

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Inter", color="#e6edf3"))

(tab_analyst, tab_bot, tab_master, tab_journal, tab_runner, tab_desk,
 tab_screener, tab_backtest, tab_options, tab_pp, tab_events, tab_rl,
 tab_sizing) = st.tabs(
    ["🤵 Analyst", "🦾 AutoTrader", "🧬 Alpha engine", "📒 Track record",
     "⚙️ Runner", "🎯 Trade desk", "🔍 Screener", "🧪 Backtest",
     "🌋 Options / IV surface", "⚖️ Portfolio & Pairs", "🌐 Event radar",
     "🤖 RL lab", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#8a8a8a"}
VERDICT_CLASS = {"LONG": "v-long", "SHORT": "v-short", "NO TRADE": "v-none"}
VERDICT_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}
FIB_COLORS = {"0": "#6ee7b7", "0.236": "#34d399", "0.382": "#fbbf24",
              "0.5": "#f59e0b", "0.618": "#f97316", "0.786": "#ef4444",
              "1": "#dc2626"}



# ===========================================================================
# -1. THE ANALYST — runs the whole desk, writes you the note
# ===========================================================================
with tab_analyst:
    st.subheader("🤵 The Analyst — your desk, run for you")
    st.caption("One button executes the entire operation: market regime, "
               "macro odds, all 50 tickers through the setup gates, every "
               "open position through the playbook, book-level risk, the "
               "news touching your names, and a statistical self-audit of "
               "the live track record. Every sentence is computed, never "
               "imagined.")
    a1, a2, a3 = st.columns(3)
    an_acct = a1.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="an_acct")
    an_prof = a2.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                           index=1, key="an_prof")
    an_news = a3.toggle("Include news", value=True, key="an_news")

    if st.button("☕ Run my morning", type="primary", key="an_run"):
        _ap = RISK_PROFILES[an_prof]
        prog = st.progress(0, text="Market regime…")
        spy = fetch_history("SPY", period="2y")
        prog.progress(15, text="Macro event odds…")
        gauge = None
        try:
            ev = fetch_macro_markets()
            gauge = equity_risk_gauge(ev) if len(ev) else None
        except Exception:
            pass
        prog.progress(30, text="Scanning the universe…")
        data = fetch_many(tuple(DEFAULT_UNIVERSE), period="2y")
        prog.progress(60, text="Marking your book…")
        jj = load_journal()
        blotter = None
        if jj["positions"]:
            mtm = mark_to_market(
                jj, lambda t: patch_live_bar(fetch_history(t, period="1y"), t))
            save_journal(jj)
            blotter = mtm.get("blotter")
        prog.progress(80, text="Writing the note…")
        brief = morning_briefing(spy, data, blotter, float(an_acct),
                                 _ap["risk_pct"], gauge)
        prog.progress(100); prog.empty()
        st.session_state["briefing"] = brief

    if "briefing" in st.session_state:
        brief = st.session_state["briefing"]
        st.markdown(f"""<div style="border:1px solid #ffb000;border-radius:14px;
            padding:14px 20px;background:rgba(255,176,0,.05);
            font-family:'JetBrains Mono',monospace">
            <span style="color:#ffb000;font-weight:800">MORNING NOTE</span>
            · {brief['stamp']} · profile: {st.session_state.get('an_prof','')}
            </div>""", unsafe_allow_html=True)
        st.write("")

        sec_defs = [("🌍 Market", "market"), ("💼 Your book", "book"),
                    ("☀️ Today's trades", "setups"),
                    ("👀 Watch list", "watch"),
                    ("🔬 System self-audit", "audit")]
        for title, key in sec_defs:
            lines = brief.get(key) or []
            if not lines:
                continue
            st.markdown(f"#### {title}")
            for ln in lines:
                cls = "reason-con" if ("⚠️" in ln or "🔴" in ln) else "reason-pro"
                st.markdown(f"<div class='{cls}'>{ln}</div>",
                            unsafe_allow_html=True)
            st.write("")

        if len(brief["setups_table"]):
            with st.expander("📋 Full setups table (sized to your profile)"):
                st.dataframe(brief["setups_table"],
                             use_container_width=True, hide_index=True)

        if st.session_state.get("an_news") and brief["news_tickers"]:
            st.markdown("#### 📰 News on your names")
            nws = ticker_news(tuple(brief["news_tickers"]))
            if len(nws):
                for _, r in nws.iterrows():
                    st.markdown(f"<div class='reason-pro'>"
                                f"<b style='color:#ffb000'>{r['ticker']}</b> — "
                                f"<a href='{r['url']}' target='_blank' "
                                f"style='color:#e6edf3'>{r['headline']}</a> "
                                f"<span style='color:#8b98a5;font-size:.8rem'>"
                                f"({r['source']})</span></div>",
                                unsafe_allow_html=True)
            else:
                st.caption("News feed empty or throttled right now.")

        with st.expander("❓ What the Analyst is (and deliberately isn't)"):
            st.markdown("""
The Analyst is an **orchestrator**: it runs every engine on the site in sequence and converts the numbers into sentences by fixed rules. If it says your position is +1.4R and the stop should move — that came from the playbook math, checkable in the 🎯 desk. If it says the live edge isn't statistically proven yet — that's the bootstrap CI on your actual recorded trades.

What it deliberately **isn't**: a language model improvising opinions. In trading, confident-sounding text without verified computation behind it is how accounts die. Everything here is auditable — click into any tab and find the number behind the sentence. If we ever add a conversational layer, it will only be allowed to narrate what these engines computed.

**The routine**: ☕ every morning before the open (16:30 your time). Read the note top to bottom — market, your book's instructions, today's trades, the watch list. Execute through the Trade desk, record to the Track record. The self-audit keeps score of whether the whole thing is actually working — and it will tell you honestly if it isn't.
""")


# ===========================================================================
# -0.5 AUTOTRADER — the bot that lives here
# ===========================================================================
with tab_bot:
    st.subheader("🦾 AutoTrader — the in-website trading bot (paper)")
    st.caption("Give it a mandate once, flip it ON. It scans, stages orders "
               "for the open, fills them, banks profits at +1R/+2R, trails, "
               "stops out — and narrates every decision. While the site is "
               "closed it sleeps; on your next visit it CATCHES UP bar-by-"
               "bar, executing exactly what the rules dictated. Real "
               "prices, paper money — building the record that decides if "
               "this logic ever touches a real broker.")

    bot = load_bot()

    bc1, bc2, bc3, bc4 = st.columns([1, 1, 1, 1.6])
    with bc1:
        enabled = st.toggle("🦾 BOT ACTIVE", value=bot["enabled"],
                            key="bot_on")
    bot_prof = bc2.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                             index=1, key="bot_prof")
    bot_acct = bc3.number_input("Paper capital $", 1000, 1_000_000,
                                int(bot.get("start_equity", 5000)),
                                step=1000, key="bot_acct")
    _bp = RISK_PROFILES[bot_prof]
    bc4.caption(f"Mandate: {_bp['risk_pct']}%/trade · max "
                f"{_bp['max_pos']} positions · default 50-name universe · "
                f"entries only on 🟢 5/5-gate setups · scale ⅓ at +1R and "
                f"+2R · 2.5×ATR trail.")

    colA, colB = st.columns([1, 4])
    if colA.button("🔄 Reset bot (wipe paper history)", key="bot_reset"):
        bot = default_state(float(bot_acct))
        save_bot(bot)
        st.success("Bot reset — fresh paper account.")

    if enabled != bot["enabled"]:
        bot["enabled"] = enabled
        if enabled and not bot["created"]:
            bot = default_state(float(bot_acct))
            bot["enabled"] = True
            from datetime import datetime as _dtnow, timezone as _tz
            bot["created"] = _dtnow.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
        bot["mandate"]["risk_pct"] = _bp["risk_pct"]
        bot["mandate"]["max_positions"] = _bp["max_pos"]
        save_bot(bot)

    if bot["enabled"]:
        @st.fragment(run_every=LIVE_EVERY)
        def _bot_live():
            b = load_bot()
            with st.spinner("Bot tick — catching up & deciding…"):
                data_b = fetch_many(tuple(DEFAULT_UNIVERSE), period="2y")
                spy_b = fetch_history("SPY", period="2y")
                if spy_b.empty or not data_b:
                    st.warning("Data throttled — bot will retry next pulse.")
                    return
                b = bot_tick(b, data_b, spy_b)
                save_bot(b)
                eq = bot_equity(b, data_b)

            m = st.columns(6)
            m[0].metric("Paper equity", f"${eq['equity']:,.0f}",
                        f"{eq['return_pct']:+.2f}%")
            m[1].metric("Cash", f"${eq['cash']:,.0f}")
            m[2].metric("Open positions", len(eq["open_table"]))
            m[3].metric("Closed deals", len(eq["closed_table"]))
            m[4].metric("Wins", eq["n_wins"])
            m[5].metric("Realized P&L", f"${eq['realized']:+,.0f}")
            st.caption(f"Bot since {b.get('created','—')} · last processed "
                       f"session: {b.get('last_tick','—')} · "
                       f"{'🔴 watching live' if LIVE_EVERY else '⏸ tick on load only (turn LIVE on to watch it work)'}")

            if len(eq["open_table"]):
                st.markdown("**📈 Bot holdings (live marks):**")
                st.dataframe(eq["open_table"], use_container_width=True,
                             hide_index=True)
            if b["pending"]:
                st.markdown("**📋 Orders staged for next open:**")
                for od in b["pending"]:
                    st.markdown(f"<div class='reason-pro'>📋 BUY "
                                f"{od['shares']} {od['ticker']} at open — "
                                f"{od['why']}</div>", unsafe_allow_html=True)

            st.markdown("**🗣️ Bot decision feed (newest first):**")
            for L in list(reversed(b["log"]))[:12]:
                good = any(k in L["action"] for k in ("BUY", "SCALE", "ORDER"))
                st.markdown(f"<div class='{'reason-pro' if good else 'reason-con'}'"
                            f" style=\"font-family:'JetBrains Mono',monospace\">"
                            f"{L['action']} <b>{L['ticker']}</b> · "
                            f"{L['shares']} sh @ ${L['price']:,.2f} · "
                            f"{L['date']}<br>→ {L['why']}</div>",
                            unsafe_allow_html=True)

            if len(eq["closed_table"]):
                st.markdown("**🧾 Bot deal ledger:**")
                st.dataframe(eq["closed_table"].iloc[::-1],
                             use_container_width=True, hide_index=True,
                             height=280)
        _bot_live()
    else:
        st.info("Bot is OFF. Flip the toggle to give it the mandate — it "
                "starts scanning immediately and stages its first orders "
                "for the next open.")

    with st.expander("❓ How the bot works & why it's trustworthy"):
        st.markdown("""
**The loop**: every tick it (1) fills yesterday's staged orders at today's real open, (2) walks every open position through the bar — scale ⅓ at +1R (stop→breakeven), scale ⅓ at +2R (stop→+1R), trail the rest at 2.5×ATR, stop out where price actually touched — then (3) scans the universe and stages tomorrow's entries from 5/5-gate setups only.

**The catch-up trick**: the rules are bar-mechanical with next-open execution, so replaying missed days is *identical* to having watched them live. The bot being "asleep" while the site is closed costs nothing.

**Why paper first is non-negotiable**: this ledger becomes the statistical evidence (the 🔬 Validation Lab can audit it) that decides whether the logic deserves a real broker API. Bots don't get promoted on enthusiasm — they get promoted on a verified record. ⚠️ The free server wipes files on redeploy; like the journal, the bot's memory resets then — one more reason its early life is paper-only.
""")

# ===========================================================================
# 0. ALPHA ENGINE — the master algorithm
# ===========================================================================
with tab_master:
    st.subheader("🧬 The master algorithm — one answer: what to do now")
    with st.expander("❓ How this machine works (read once — 60 seconds)"):
        st.markdown("""
Four filters in a row; a stock must survive ALL of them to reach your plan:

**1️⃣ MARKET GATE** — is the overall market (SPY) healthy? Decides how much of your account may deploy at all (Bull·Calm = 100% → Bear·Storm = 15%). *Don't fight the tape.*

**2️⃣ CROSS-SECTIONAL RANK** — all 50 stocks scored on 6 published anomalies (momentum, 52-week high, anti-lottery, low-vol, low-beta, reversal) and ranked against each other. Top decile advances; bottom 5 become the avoid list.

**3️⃣ TIME-SERIES VERDICT** — survivors face the full 7-model engine (trend, B-X, MACD…) + a quick backtest of the signal on that exact ticker. Only LONG with real conviction passes. **This is the strictest gate — most days most names fail here. An empty plan means the machine is protecting you, not malfunctioning.**

**4️⃣ RISK SIZING** — equal risk per position, total portfolio heat capped. Then the plan prints: ticker, shares, entry, stop, target.

**☀️ Today's setups** below is the FAST lane: the same entry gates, no backtests — what's tradeable *right now*, in ~10 seconds.
""")

    prof_c1, prof_c2 = st.columns([1.2, 3])
    ma_profile = prof_c1.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                                   index=1, key="ma_prof")
    _pp = RISK_PROFILES[ma_profile]
    prof_c2.caption(f"**{ma_profile}**: {_pp['risk_pct']}% risk/position · "
                    f"up to {_pp['max_pos']} positions · "
                    f"{_pp['heat_cap']}% total heat · conviction ≥ "
                    f"{_pp['conviction_min']}. Aggressive ≈ 2-3× the P&L "
                    f"swing of Conservative — in BOTH directions. The 🛡️ "
                    f"risk-of-ruin stats in Backtest show what your profile "
                    f"survives.")

    c1, c2, c3 = st.columns(3)
    ma_acct = c1.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="ma_acct")
    ma_custom = c2.text_input("Universe (empty = default 50)",
                              placeholder="AAPL, NVDA ...", key="ma_uni")
    ma_aggfill = c3.toggle("Aggressive fill (never return empty)",
                           value=("Aggressive" in ma_profile),
                           help="If the strict gates pass <2 names, take the "
                                "top alpha names at HALF size, tagged as "
                                "lower-confidence.")
    ma_risk = _pp["risk_pct"]
    ma_maxpos = _pp["max_pos"]

    # ---- ☀️ TODAY'S SETUPS — the daily trades scanner --------------------
    if st.button("☀️ Scan today's setups (fast — all data, right now)",
                 key="scan_today"):
        uni_s = tuple(t.strip().upper() for t in ma_custom.split(",")
                      if t.strip()) or tuple(DEFAULT_UNIVERSE)
        with st.spinner(f"Playbook gates on {len(uni_s)} tickers…"):
            data_s = fetch_many(uni_s, period="2y")
            setups = scan_setups(data_s, account=float(ma_acct),
                                 risk_pct=ma_risk)
        if len(setups):
            n_enter = int((setups["urgency"] == "🟢 ENTER").sum())
            n_fast = int((setups["urgency"] == "🟡 FAST").sum())
            st.success(f"**Today: {n_enter} full entries · {n_fast} dip "
                       f"scalps · {len(setups)-n_enter-n_fast} stalking.** "
                       f"Sized at {ma_profile} ({ma_risk}%/trade).")
            st.dataframe(setups, use_container_width=True, hide_index=True)
            st.caption("🟢 ENTER = all 5 gates green, full playbook trade. "
                       "🟡 FAST = RSI2 panic in an uptrend — quick scalp, "
                       "exit on the snap-back. 👀 STALK = one gate away; "
                       "set an alert. Run any name through 🎯 Trade desk "
                       "for the full workup before pulling the trigger.")
        else:
            st.info("**Zero setups today.** The market isn't offering — "
                    "chasing anyway is how edges become donations. "
                    "Tomorrow is another scan.")

    st.markdown("---")

    if st.button("🚀 Run the machine", type="primary", key="ma_run"):
        uni = tuple(t.strip().upper() for t in ma_custom.split(",")
                    if t.strip()) or tuple(DEFAULT_UNIVERSE)
        prog = st.progress(0, text="Downloading universe…")
        data = fetch_many(uni, period="2y")
        prog.progress(30, text="Downloading SPY (market gate)…")
        spy = fetch_history("SPY", period="2y")
        prog.progress(45, text="Ranking cross-sectional anomalies…")
        res = run_master(data, spy, account=float(ma_acct),
                         risk_pct=ma_risk, max_positions=ma_maxpos,
                         heat_cap_pct=_pp["heat_cap"],
                         conviction_min=_pp["conviction_min"],
                         aggressive_fill=ma_aggfill)
        prog.progress(100, text="Done")
        prog.empty()
        st.session_state["master_res"] = res
        st.session_state["master_acct"] = float(ma_acct)

    if "master_res" in st.session_state:
        res = st.session_state["master_res"]
        if "error" in res:
            st.error(res["error"])
            st.stop()

        # --- Market gate ---------------------------------------------------
        reg = res["regime"]
        g1, g2, g3, g4 = st.columns([1.7, 1, 1, 1])
        g1.markdown(f"<div class='regime-badge'>{reg['regime']} (SPY)</div>"
                    f"<div style='color:#8b98a5;font-size:.85rem;margin-top:6px'>"
                    f"{reg['playbook']}</div>", unsafe_allow_html=True)
        g2.metric("Capital allowed to deploy", f"{res['exposure_pct']}%")
        g3.metric("Portfolio risk if all stops hit",
                  f"${res['total_risk']:,.0f} ({res['total_risk_pct']}%)")
        g4.metric("Cash left", f"${res['cash']:,.0f} ({res['cash_pct']}%)")

        # --- THE PLAN --------------------------------------------------------
        st.markdown("## 📋 The plan")
        if len(res["plan"]):
            st.dataframe(res["plan"], use_container_width=True,
                         hide_index=True)
            if res["plan"]["action"].str.contains("½").any():
                st.caption("*½size = aggressive fill: strong alpha rank but "
                           "the verdict engine wasn't fully convinced — "
                           "taken at HALF risk so a good rank can't hurt "
                           "you at full weight.")
            st.success(f"**Do this:** open the {len(res['plan'])} position(s) "
                       f"above with the exact share counts, place the stops "
                       f"immediately, keep ${res['cash']:,.0f} in cash. "
                       f"If a stop hits — you're out, no negotiating with it.")
        else:
            st.info("**The machine says: do nothing.** No candidate passed all "
                    "four gates (anomaly rank → verdict → regime → risk). "
                    "Cash is a position; the next setup will come to you.")

        if len(res["plan"]):
            if st.button("📒 Record this plan to the track record",
                         key="rec_plan"):
                jj = load_journal()
                jj, n_added = record_plan(jj, res["plan"],
                                          res["regime"]["regime"],
                                          st.session_state.get("master_acct",
                                                               5000.0))
                save_journal(jj)
                st.success(f"Recorded {n_added} position(s) with UTC "
                           f"timestamp, model version and regime stamp. "
                           f"See the 📒 Track record tab.")

        st.warning(f"**Honesty layer (McLean & Pontiff 2016):** published "
                   f"anomalies earn ~{res['haircut_pct']}% LESS after "
                   f"publication as arbitrageurs crowd in. Whatever edge this "
                   f"engine finds, assume roughly half survives in live "
                   f"trading. That is why risk caps matter more than signal "
                   f"strength.")

        # --- Anomaly ranking table ---------------------------------------------
        st.markdown("### 🏆 Cross-sectional anomaly ranking")
        st.dataframe(
            res["ranks"].style.background_gradient(subset=["alpha"],
                                                   cmap="RdYlGn"),
            use_container_width=True, height=480)
        st.caption(f"Bottom of the table = research says avoid: "
                   f"{', '.join(res['avoid'])}")

        with st.expander("📚 The research behind each column (SSRN / journals)"):
            for key, (name, paper, desc) in ANOMALY_INFO.items():
                st.markdown(f"- **{name}** (`{key}`) — *{paper}*: {desc}")
            st.markdown("""
---
**Meta-research this engine is built on:**
- *Jensen, Kelly & Pedersen (2023, Journal of Finance)* — most published factors replicate, cluster into 13 themes, and work out-of-sample across 93 countries. Anomaly investing is real.
- *McLean & Pontiff (2016, Journal of Finance)* — but returns are ~26% lower out-of-sample and ~58% lower post-publication. Hence the haircut above.
- *Bali, Brown, Murray & Tang (2017)* — lottery demand (MAX) largely subsumes the beta anomaly, which is why MAX gets a heavy weight here.

**How the fusion works:** each anomaly is z-scored ACROSS the universe (cross-sectional, exactly as defined in the papers), weighted, and summed into `alpha`. The top decile then has to survive the 7-model time-series verdict, the SPY regime gate, and the portfolio heat cap. Four independent filters — most stocks fail at least one, and that's the point.
""")

        # --- Considered but rejected ------------------------------------------
        rej = [v for v in res["considered"] if v["verdict"] != "LONG"]
        if rej:
            st.markdown("### 🚫 High-alpha names the verdict engine rejected")
            rej_df = pd.DataFrame([{
                "ticker": v["ticker"], "alpha rank %": v["pct_rank"],
                "verdict": v["verdict"], "conviction": v["conviction"],
                "why (top reason against)": (v["reasons_con"][0]
                                             if v["reasons_con"] else "—"),
            } for v in rej])
            st.dataframe(rej_df, use_container_width=True, hide_index=True)
            st.caption("Good anomaly scores, bad timing — the whole point of "
                       "layering time-series checks on top of cross-sectional "
                       "ranks.")


# ===========================================================================
# 0b. TRACK RECORD — the fund fact sheet
# ===========================================================================
with tab_journal:
    st.subheader("📒 Track record — verified paper-trading journal")
    st.caption("Append-only journal: every plan stamped with UTC time, model "
               "version and market regime at entry. Stops & targets enforced "
               "mechanically on real daily bars (first touch; stop wins ties). "
               "This is how a strategy earns trust before real money.")

    @st.fragment(run_every=LIVE_EVERY)
    def _journal_live():
        jj = load_journal()
        n_pos = len(jj["positions"])

        ctop = st.columns([1, 1, 2])
        if n_pos:
            ctop[0].download_button("⬇️ Export journal (CSV)",
                                    journal_to_csv(jj), "quantsignal_journal.csv",
                                    "text/csv")
        up = ctop[1].file_uploader("Restore from CSV", type="csv",
                                   label_visibility="collapsed")
        if up is not None:
            jj = journal_from_csv(up.getvalue().decode())
            save_journal(jj)
            st.success(f"Journal restored — {len(jj['positions'])} positions.")
            n_pos = len(jj["positions"])
        ctop[2].info("⚠️ Free hosting wipes local files on redeploy — export "
                     "after every session. The CSV is your custody.")

        if not n_pos:
            st.info("No positions recorded yet. Run the 🧬 Alpha engine and hit "
                    "**Record this plan** — the clock starts there.")
        else:
            with st.spinner("Marking positions to market…"):
                mtm = mark_to_market(
                jj, lambda t: patch_live_bar(fetch_history(t, period="1y"), t))
                save_journal(jj)   # persist any auto-closed stops/targets

            if mtm.get("data_issues"):
                st.warning("Data issues: " + ", ".join(mtm["data_issues"]))

            s = mtm["stats"]
            head = st.columns(6)
            head[0].metric("Paper equity", f"${s.get('Equity $', 0):,.0f}")
            head[1].metric("Total return", f"{s.get('Total return %', 0)}%")
            head[2].metric("vs SPY", f"{s.get('Alpha vs SPY %', '—')}%"
                           if "Alpha vs SPY %" in s else "—")
            head[3].metric("Live Sharpe", s.get("Sharpe (live)", "—"))
            head[4].metric("Hit rate", f"{s.get('Hit rate %', '—')}%"
                           if "Hit rate %" in s else "—")
            head[5].metric("Open heat", f"${s.get('Heat (risk if all stops hit) $', 0):,.0f}")

            meta1, meta2, meta3 = st.columns(3)
            meta1.caption(f"Inception: {jj['meta'].get('inception', '—')}")
            meta2.caption(f"Model: {jj['meta'].get('version', '—')}")
            meta3.caption(f"Positions: {s.get('Open / Closed', '—')} "
                          f"(open/closed) · Max DD {s.get('Max DD %', '—')}%")

            # ---- 🛡️ RISK DESK — live book risk, the quant way -------------
            open_b = mtm["blotter"][mtm["blotter"]["status"] == "OPEN"]
            if len(open_b):
                st.markdown("### 🛡️ Risk desk — the open book")
                poss = [{"ticker": r["ticker"], "shares": int(r["shares"]),
                         "entry": float(r["entry"]), "stop": float(r["stop"])}
                        for _, r in open_b.iterrows()]
                rets_map = {}
                for p_ in poss:
                    try:
                        h_ = fetch_history(p_["ticker"], period="1y")
                        rets_map[p_["ticker"]] = \
                            h_["Close"].pct_change().dropna()
                    except Exception:
                        continue
                acct_ = float(jj["meta"].get("account", 5000.0))
                pv = portfolio_var(poss, rets_map, acct_)
                ch = correlation_heat(poss, rets_map, acct_)
                rk1, rk2, rk3, rk4, rk5 = st.columns(5)
                if pv:
                    rk1.metric("1-day VaR 95%",
                               f"${pv['VaR_$']:,.0f} ({pv['VaR_%']}%)",
                               help="On a normal bad day (1 in 20), expect "
                                    "to lose up to this much.")
                    rk2.metric("1-day CVaR 95%",
                               f"${pv['CVaR_$']:,.0f} ({pv['CVaR_%']}%)",
                               help="When that bad day happens, this is the "
                                    "AVERAGE loss — the tail number desks "
                                    "size by.")
                    rk3.metric("Gross exposure",
                               f"{pv['gross_exposure_%']}%")
                if ch:
                    rk4.metric("Heat: naive → corr-adj",
                               f"${ch['naive_heat_$']:,.0f} → "
                               f"${ch['corr_adj_heat_$']:,.0f}")
                    rk5.metric("Avg pairwise corr", ch["avg_correlation"],
                               delta="⚠️ crowded book" if ch["warning"]
                               else "diversified",
                               delta_color="inverse" if ch["warning"]
                               else "normal")
                if ch and ch["warning"]:
                    st.warning("Your open positions are highly correlated — "
                               "effectively ONE big trade. A single market "
                               "move can hit every stop together. Consider "
                               "trimming or diversifying sectors.")
                with st.expander("❓ Reading the risk desk"):
                    st.markdown("""
- **VaR 95%** — parametric 1-day Value-at-Risk from the actual covariance of your holdings: "on a normal bad day, expect up to this."
- **CVaR** — the average loss *given* that bad day happened. Desks size by CVaR, not VaR, because tails are where accounts die.
- **Correlation-adjusted heat** — summing per-position risk pretends positions are independent. When names move together (avg corr > 0.6), your true worst case approaches the naive sum — the diversification you think you have is an illusion. The gap between the numbers = your real diversification benefit.
""")

            if len(mtm["equity"]) > 1:
                fige = go.Figure()
                fige.add_trace(go.Scatter(x=mtm["equity"].index, y=mtm["equity"],
                                          name="Portfolio",
                                          line=dict(color="#10b981", width=2)))
                if len(mtm["bench"]) > 1:
                    fige.add_trace(go.Scatter(x=mtm["bench"].index, y=mtm["bench"],
                                              name="SPY (same $)",
                                              line=dict(color="#8b98a5", width=1.5,
                                                        dash="dot")))
                fige.update_layout(height=380, yaxis_title="Equity $",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(fige, use_container_width=True)

            st.markdown("#### Blotter")
            st.dataframe(mtm["blotter"], use_container_width=True,
                         hide_index=True, height=360)

            if len(mtm["monthly"]):
                st.markdown("#### Monthly returns")
                st.dataframe(mtm["monthly"], use_container_width=True,
                             hide_index=True)

            with st.expander("❓ Why this is the feature that matters most"):
                st.markdown("""
    Backtests can be (accidentally) curve-fit. A **forward paper record** cannot: the timestamps prove every pick was made *before* the outcome. This is exactly how allocators evaluate new managers — months of verified process before a dollar moves. Rules of the game: record every plan (no cherry-picking), let stops do their job, export the CSV after each session, and judge nothing before ~20 closed trades. If after months the record shows an edge over SPY — it's real. If it doesn't — the site just saved you real money.
    """)


    _journal_live()

# ===========================================================================
# 0c. RUNNER — the trade lifecycle machine
# ===========================================================================
with tab_runner:
    st.subheader("⚙️ Runner — the machine trades it, you read the log")
    st.caption("Bar-by-bar lifecycle on real history: entries (trend or dip, "
               "auto-picked by Hurst), scale-outs at +1R and +2R, breakeven "
               "jumps, chandelier trail, time exits — every event logged with "
               "the model state and the reason. Ends with today's live "
               "status and tomorrow's order.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    rn_tkr = c1.text_input("Ticker", value="NVDA", key="rn").upper().strip()
    rn_tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="rnp")
    rn_acct = c3.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="rn_acct")
    rn_mode = c4.selectbox("Mode", ["auto", "trend", "dip"], key="rn_mode")

    if st.button("▶️ Run the machine", type="primary", key="rn_run"):
        st.session_state["rn_params"] = dict(
            rn_tkr=rn_tkr, rn_tf=rn_tf, rn_acct=float(rn_acct),
            rn_mode=rn_mode)

    @st.fragment(run_every=LIVE_EVERY)
    def _runner_live():
        if "rn_params" not in st.session_state:
            return
        _p = st.session_state["rn_params"]
        rn_tkr, rn_tf = _p["rn_tkr"], _p.get("rn_tf", "Daily")
        rn_acct, rn_mode = _p["rn_acct"], _p["rn_mode"]
        _rm = tf_meta(rn_tf)
        with st.spinner(f"Replaying every {rn_tf} bar through the models…"):
            df = fetch_tf(rn_tkr, rn_tf)
            if rn_tf == "Daily":
                df = patch_live_bar(df, rn_tkr)
            if len(df) < _rm["min_bars"]:
                st.error(f"Need at least ~{_rm['min_bars']} {rn_tf} bars.")
                return
            res = run_machine(df, account=float(rn_acct), mode=rn_mode)
            if "error" in res:
                st.error(res["error"])
                return

        # live state card
        stt = res["state"]
        if stt["in_position"]:
            st.markdown(f"""
            <div class="verdict v-long">
              <div>
                <h2>🟢 IN POSITION — {rn_tkr}</h2>
                <div class="sub">{stt['shares']} shares @ ${stt['entry']:,.2f}
                 · held {stt['bars_held']} bars · {stt['scaled']} ·
                 unrealized ${stt['unrealized']:,.0f}</div>
              </div>
              <div style="text-align:right">
                <div style="font-size:.85rem;opacity:.8">Active stop</div>
                <div style="font-size:1.6rem;font-weight:800">
                  ${stt['stop_now']:,.2f}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict v-none">
              <div><h2>⚪ FLAT — {rn_tkr}</h2>
              <div class="sub">machine mode: {res['mode'].upper()}</div></div>
            </div>""", unsafe_allow_html=True)
        st.info(f"**Tomorrow's order:** {stt['tomorrow']}")

        sc = st.columns(len(res["stats"]))
        for col, (k, v) in zip(sc, res["stats"].items()):
            col.metric(k, v)

        # price chart with event markers
        ev = res["events"]
        figrn = go.Figure()
        figrn.add_trace(go.Candlestick(
            x=df.index[-504:], open=df["Open"][-504:], high=df["High"][-504:],
            low=df["Low"][-504:], close=df["Close"][-504:], name=rn_tkr))
        if len(ev):
            ev_plot = ev[pd.to_datetime(ev["date"]).isin(df.index[-504:])]
            marker_map = [("ENTRY", "triangle-up", "#10b981"),
                          ("SCALE", "diamond", "#22d3ee"),
                          ("TARGET", "star", "#6ee7b7"),
                          ("STOP", "x", "#ef4444"),
                          ("EXIT", "circle", "#f59e0b")]
            for key, sym, col_ in marker_map:
                sub = ev_plot[ev_plot["event"].str.contains(key)]
                if len(sub):
                    figrn.add_trace(go.Scatter(
                        x=pd.to_datetime(sub["date"]), y=sub["price"],
                        mode="markers", name=key.title(),
                        marker=dict(symbol=sym, size=11, color=col_,
                                    line=dict(width=1, color="#0b0f14"))))
        figrn.update_layout(height=520, xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=10, t=30, b=10),
                            **PLOTLY_LAYOUT)
        st.plotly_chart(figrn, use_container_width=True)

        # ---- 🗣️ Decision feed — the machine narrates itself -----------------
        st.markdown("#### 🗣️ Decision feed — the machine explains every move")
        ev_feed = res["events"].iloc[::-1].head(10)
        ICONS = {"ENTRY": "🟢", "SCALE": "💰", "TARGET": "🎯",
                 "STOP": "🛑", "BREAKEVEN": "🟡", "TRAIL": "📉",
                 "SIGNAL": "🔵", "TIME": "⏰"}
        for _, e_ in ev_feed.iterrows():
            ic = next((v for k, v in ICONS.items() if k in e_["event"]), "•")
            is_entry = "ENTRY" in e_["event"]
            cls = "reason-pro" if is_entry or "SCALE" in e_["event"] \
                or "TARGET" in e_["event"] else "reason-con"
            st.markdown(
                f"<div class='{cls}' style=\"font-family:'JetBrains Mono',"
                f"monospace\">{ic} <b>{e_['date']}</b> · {e_['event']} · "
                f"{e_['shares']} sh @ ${e_['price']:,.2f}<br>"
                f"<span style='color:#8b98a5'>saw: score {e_['score']:+.2f} · "
                f"BX {e_['bx']} · RSI2 {e_.get('rsi2','—')}</span><br>"
                f"→ {e_['note']}</div>", unsafe_allow_html=True)

        # ---- 🧾 Deal ledger — full closure reports ---------------------------
        if len(res.get("closures", [])):
            st.markdown("#### 🧾 Deal ledger — every round-trip, fully "
                        "accounted")
            cl = res["closures"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Round-trips", len(cl))
            k2.metric("Won", f"{float((cl['P&L $']>0).mean())*100:.0f}%")
            k3.metric("Avg return on capital",
                      f"{float(cl['return on capital %'].mean()):+.2f}%")
            k4.metric("Total realized P&L",
                      f"${float(cl['P&L $'].sum()):+,.0f}")
            st.dataframe(cl.iloc[::-1], use_container_width=True,
                         hide_index=True, height=320)

        # ---- 🎓 What the machine learned -------------------------------------
        if res.get("lessons"):
            st.markdown("#### 🎓 What the machine learned on this ticker")
            for L in res["lessons"]:
                cls = "reason-con" if "⚠️" in L else "reason-pro"
                st.markdown(f"<div class='{cls}'>🎓 {L}</div>",
                            unsafe_allow_html=True)
            with st.expander("❓ How the self-learning works"):
                st.markdown("""
After every run the machine audits its own closed trades on this ticker and derives lessons **from data, not vibes**: which exit types made vs cost money, whether entries with strong B-X confirmation outperformed weak ones, the win rate and average return **on deployed capital** per round-trip, and the time-asymmetry check — if losers are held longer than winners, that's the single most common way traders bleed, and the machine will flag itself for it. Every lesson traces to rows you can see in the deal ledger above. This is what "self-learning" honestly means at this scale: measured feedback, not magic.
""")

        st.markdown("#### 📜 The log — every decision, with its reason")
        if len(ev):
            st.dataframe(ev.iloc[::-1], use_container_width=True,
                         hide_index=True, height=420)
        else:
            st.info("The machine never found an entry it liked on this "
                    "ticker — that is a valid (and cheap) outcome.")

        figeq = go.Figure(go.Scatter(x=res["equity"].index, y=res["equity"],
                                     line=dict(color="#10b981", width=2),
                                     name="Machine equity"))
        figeq.update_layout(height=260, yaxis_title="Equity $",
                            margin=dict(l=10, r=10, t=30, b=10),
                            **PLOTLY_LAYOUT)
        st.plotly_chart(figeq, use_container_width=True)

        with st.expander("❓ How the machine decides"):
            st.markdown("""
**Entry** — trend mode: composite BUY + B-Xtrender rising & positive + above the 200-SMA. Dip mode: RSI(2) < 10 panic *inside the Fibonacci 0.382–0.786 pocket* of an uptrend. Mode auto-picked by the ticker's Hurst exponent.

**The lifecycle** — at **+1R**: sell ⅓, stop jumps to breakeven (the trade can no longer lose). At **+2R**: sell another ⅓, stop jumps to entry+1R (profit is locked). The last third rides a 2.5×ATR chandelier trail as far as the trend goes. Stale unprofitable trades get time-stopped.

**Why scale-outs** — they resolve the eternal "take profit vs let it run" fight by doing both: the win rate rises (thirds get banked), the tail stays open (the runner catches the big moves). It costs a little expectancy vs all-or-nothing in pure trends — and buys a smoother equity curve and a calmer trader. That's usually the right trade.
""")

    _runner_live()

# ===========================================================================
# 1. TRADE DESK
# ===========================================================================
with tab_desk:
    c1, c2, c3, c4, c5 = st.columns([1.7, 1, 1, 1, 1.3])
    tkr = c1.text_input("Ticker", value="NVDA", key="desk").upper().strip()
    tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="desk_tf")
    account = c3.number_input("Account $", 500, 1_000_000, 5000, step=500)
    risk_pct = c4.slider("Risk/trade %", 0.5, 3.0, 1.0, 0.25)
    use_opts = c5.toggle("Include options skew (slower)", value=False)

    if st.button("Run desk analysis", type="primary", key="deskrun"):
        st.session_state["desk_params"] = dict(
            tkr=tkr, account=float(account), risk_pct=float(risk_pct),
            use_opts=bool(use_opts), tf=tf)

    @st.fragment(run_every=LIVE_EVERY)
    def _desk_live():
        if "desk_params" not in st.session_state:
            return
        _p = st.session_state["desk_params"]
        tkr, account = _p["tkr"], _p["account"]
        risk_pct, use_opts = _p["risk_pct"], _p["use_opts"]
        tf = _p.get("tf", "Daily")
        _m = tf_meta(tf)
        with st.spinner(f"Crunching 7 models on {tf} bars…"):
            df = fetch_tf(tkr, tf)
            if tf == "Daily":
                df = patch_live_bar(df, tkr)
            if df.empty or len(df) < _m["min_bars"]:
                st.error(f"Not enough {tf} bars for {tkr} "
                         f"(need ~{_m['min_bars']}).")
                return
            skew = None
            flow_share = None
            if use_opts:
                try:
                    _, chain_d = fetch_chains(tkr, max_expiries=4)
                    skew = skew_25(chain_d)
                    uf_d = unusual_flow(chain_d, top_n=30)
                    if len(uf_d):
                        callp = uf_d.loc[uf_d["type"] == "C", "premium_$"].sum()
                        totp = uf_d["premium_$"].sum()
                        flow_share = float(callp / totp) if totp > 0 else None
                except Exception:
                    skew = None
            v = analyze(df, account=account, risk_pct=risk_pct, skew=skew,
                        flow_call_share=flow_share)
            h = hurst(df)
            fib = fib_levels(df)
            reg = regime_quadrant(df)
            vol = ewma_vol(df)
            sr = support_resistance(df)
            direction = 1 if v["verdict"] != "SHORT" else -1
            paths = simulate(df, days=30, n_paths=2000)
            odds = trade_odds(paths, v["entry"], v["stop"], v["target"],
                              direction)
            bands = cone(paths)
            p_win = (odds["p_target_first"] /
                     max(odds["p_target_first"] + odds["p_stop_first"], 1e-9))
            kel = kelly(p_win, v["rr"])

        # ---- 📖 PLAYBOOK — the WHEN engine -----------------------------------
        pb = build_playbook(df, account=account, risk_pct=risk_pct)
        urg_color = {"🟢 ACTIONABLE": "#10b981", "🟡 FAST SETUP": "#f59e0b",
                     "🟡 WATCH": "#f59e0b", "⚪ NO TRADE": "#6b7280",
                     "🟢 CALM": "#10b981", "🟡 SOON": "#f59e0b",
                     "🟠 TODAY": "#f97316", "🔴 IMMEDIATE": "#ef4444"}.get(
            pb["urgency"], "#8b98a5")
        st.markdown(f"""
        <div style="border:1px solid {urg_color};border-radius:16px;
                    padding:18px 22px;margin-bottom:14px;
                    background:linear-gradient(135deg,rgba(19,26,34,.95),
                    rgba(11,15,20,.95));box-shadow:0 0 24px {urg_color}22">
          <div style="font-size:.8rem;color:{urg_color};font-weight:800;
                      letter-spacing:1px">📖 PLAYBOOK · {pb['urgency']} ·
                      {pb['greens']}/5 gates green</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:6px;
                      font-family:'JetBrains Mono',monospace">
                      {pb['instruction']}</div>
        </div>""", unsafe_allow_html=True)
        gc = st.columns(5)
        for col, (name, ok, detail) in zip(gc, pb["gates"]):
            col.markdown(f"{'✅' if ok else '⛔'} **{name.split('(')[0]}**")
            col.caption(detail)
        with st.expander("❓ The playbook — when to enter, manage, exit"):
            st.markdown("""
The playbook runs the exact five gates the backtest engine trades, live:
**ENTER** when all 5 are green (with shares/stop/scale levels printed). **DIP SETUP** when RSI(2) panics inside an intact uptrend — the fast scalp lane. **STALK** at 3–4 greens: it names what's blocking, you set an alert. **STAND DOWN** below that — no setup exists, and forcing one is how accounts bleed.

Once you're in a trade, re-run with your entry/stop (Runner tracks this automatically) and the playbook switches to management: **PROTECT** at +1R (stop → breakeven), **SCALE** at +2R (bank a third), **TIGHTEN** when B-X rolls over, **EXIT** on a composite flip or stop violation — each with an urgency color. It's the same lifecycle the ⚙️ Runner trades historically, pointed at *right now*.
""")

        # ---- Regime + vol forecast row --------------------------------------
        garch = garch_forecast(df) if tf == "Daily" else {}
        rg1, rg2, rg3, rg4, rg5 = st.columns([1.6, 1, 1, 1, 1])
        rg1.markdown(f"<div class='regime-badge'>{reg['regime']}</div>"
                     f"<div style='color:#8b98a5;font-size:.85rem;margin-top:6px'>"
                     f"{reg['playbook']}</div>", unsafe_allow_html=True)
        rg2.metric("EWMA vol (annual)", f"{vol['sigma_annual_pct']}%")
        rg3.metric("Expected move (1 day)", f"±${vol['expected_move_1d']:,.2f}")
        rg4.metric("GARCH(1,1) 1-day move",
                   f"±${garch['move_1d']:,.2f}" if garch else "—",
                   delta=f"persistence {garch['persistence']}" if garch else None,
                   delta_color="off")
        rg5.metric("Hurst exponent", h,
                   delta="trending" if h > 0.55 else
                   "mean-reverting" if h < 0.45 else "random walk",
                   delta_color="off")
        with st.expander("❓ Regime, EWMA volatility & Hurst — why they matter"):
            st.markdown("""
- **Regime quadrant** — price vs its 200-day average (bull/bear) × current volatility vs its own history (calm/storm). Each quadrant has a playbook; most losing streaks come from running a bull-calm playbook in a bear-storm.
- **EWMA vol (RiskMetrics λ=0.94)** — the industry-standard forecast of tomorrow's volatility, weighting recent days most. The ± number is the *expected* one-day move: intraday wiggles inside it are noise, not signal.
- **Hurst exponent** — the ticker's memory. >0.5 moves tend to continue (trust trend models), <0.5 they reverse (trust mean-reversion), ≈0.5 random walk.
""")

        _remember("desk", {"ticker": tkr, "verdict": v["verdict"],
                           "conviction": v["conviction"],
                           "garch": garch.get("sigma_annual_pct") if garch else None,
                           "tf": tf})
        _mem_o = st.session_state["memory"].get("options")
        if _mem_o and _mem_o.get("ticker") == tkr:
            st.caption(f"🧠 From your options run: {_mem_o['vol_state']} · "
                       f"skew {_mem_o.get('skew','—')} — factored into how "
                       f"you should express this view (see 🌋 Edge finder).")

        # ---- Verdict banner ---------------------------------------------------
        cls = VERDICT_CLASS[v["verdict"]]
        conv_color = ("#10b981" if v["verdict"] == "LONG"
                      else "#ef4444" if v["verdict"] == "SHORT" else "#6b7280")
        st.markdown(f"""
        <div class="verdict {cls}">
          <div>
            <h2>{VERDICT_EMOJI[v['verdict']]} {v['verdict']} — {tkr}</h2>
            <div class="sub">Composite {v['score']:+.2f} · {v['agree']}/7 models aligned ·
              signal Sharpe on {tkr}: {v['sharpe']} ({v['n_trades']} trades)</div>
          </div>
          <div style="min-width:230px">
            <div style="font-size:.85rem;opacity:.8;margin-bottom:4px">
              Conviction {v['conviction']}/100</div>
            <div class="conv-wrap"><div class="conv-bar"
              style="width:{v['conviction']}%;background-color:{conv_color}"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("❓ What is the verdict & conviction?"):
            st.markdown("""
The verdict fuses seven models (trend, momentum, B-Xtrender, MACD, RSI, mean-reversion, volume) into one composite score, then demands: models agreeing, a calm volatility regime, a *proven* historical edge on this exact ticker, and risk/reward ≥ 1.3. Fail any → **NO TRADE**. Standing aside is a position.

**Conviction (0–100)**: 40% signal strength + 25% model agreement + 15% volatility regime + 20% historical edge (± options skew). Above 55 = tradeable.
""")

        # ---- Levels -------------------------------------------------------------
        if v["verdict"] != "NO TRADE":
            m = st.columns(6)
            m[0].metric("Entry", f"${v['entry']:,.2f}")
            m[1].metric("Stop", f"${v['stop']:,.2f}")
            m[2].metric("Target", f"${v['target']:,.2f}")
            m[3].metric("Risk : Reward", f"1 : {v['rr']}")
            m[4].metric("Shares", v["shares"])
            m[5].metric("$ at risk", f"${v['risk_dollars']:,.0f}")
            if v["verdict"] == "SHORT":
                st.warning("Shorting needs margin approval at your broker; if "
                           "unavailable, treat SHORT as **avoid / exit longs**.")
            if v["verdict"] == "LONG":
                if st.button(f"📒 Track this trade ({tkr}: {v['shares']} sh, "
                             f"stop ${v['stop']:,.2f})", key="desk_rec"):
                    _plan1 = pd.DataFrame([{
                        "ticker": tkr, "shares": v["shares"],
                        "entry ~": v["entry"], "stop": v["stop"],
                        "target": v["target"],
                        "conviction": v["conviction"]}])
                    _jj = load_journal()
                    _jj, _n = record_plan(_jj, _plan1, reg["regime"],
                                          float(account))
                    save_journal(_jj)
                    st.success(f"Recorded {tkr} to the 📒 Track record "
                               f"(UTC-stamped, regime: {reg['regime']}). "
                               f"Export the CSV after your session!")
            with st.expander("❓ How are these levels computed?"):
                st.markdown("""
- **Stop** = entry ± 2.5×ATR — wide enough to survive noise, tight enough to cap damage.
- **Target** = the 63-day swing level, or a 2R measured move on breakouts to new highs.
- **Shares** = sized so a stop-out loses exactly your chosen % of the account.
""")
        else:
            st.info("**Standing aside is a position.** The desk found no edge "
                    "worth risking money on right now — that's the system "
                    "working, not failing.")

        # ---- Reasons -------------------------------------------------------------
        colp, colc = st.columns(2)
        with colp:
            st.markdown("**✅ For**")
            for r in v["reasons_pro"]:
                st.markdown(f"<div class='reason-pro'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_pro"]:
                st.markdown("<div class='reason-con'>Nothing working in favour "
                            "right now</div>", unsafe_allow_html=True)
        with colc:
            st.markdown("**⚠️ Against**")
            for r in v["reasons_con"]:
                st.markdown(f"<div class='reason-con'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_con"]:
                st.markdown("<div class='reason-pro'>No red flags detected</div>",
                            unsafe_allow_html=True)

        st.markdown("---")

        # ---- 🎲 Monte Carlo + Kelly ------------------------------------------------
        st.markdown("### 🎲 Monte Carlo — 2,000 simulated futures (30 days)")
        mc = st.columns(6)
        mc[0].metric("P(hit target first)", f"{odds['p_target_first']}%")
        mc[1].metric("P(hit stop first)", f"{odds['p_stop_first']}%")
        mc[2].metric("P(profitable at day 30)", f"{odds['p_profit_end']}%")
        mc[3].metric("95% CVaR / share", f"${odds['cvar95_share']:,.2f}")
        mc[4].metric("Kelly optimal size", f"{kel['kelly_pct']}%" if
                     kel["edge_positive"] else "No edge")
        mc[5].metric("Half-Kelly (use this)", f"{kel['half_kelly_pct']}%" if
                     kel["edge_positive"] else "—")

        if odds["p_target_first"] + odds["p_stop_first"] > 10:
            _wr = odds["p_target_first"] / max(
                odds["p_target_first"] + odds["p_stop_first"], 1e-9)
            _ror = risk_of_ruin(win_rate=_wr, avg_win=v["rr"], avg_loss=1.0,
                                risk_per_trade_pct=risk_pct)
            if _ror:
                st.caption(f"🛡️ **Risk of a 30% drawdown** trading this "
                           f"setup repeatedly at {risk_pct}% risk: "
                           f"**{_ror['prob_of_ruin_%']}%** "
                           f"({_ror['verdict']}) · expectancy "
                           f"{_ror['expectancy_R']:+.2f}R per trade "
                           f"(5,000 simulated careers).")

        x = list(range(paths.shape[1]))
        figmc = go.Figure()
        figmc.add_trace(go.Scatter(x=x + x[::-1],
                                   y=list(bands[95]) + list(bands[5])[::-1],
                                   fill="toself", fillcolor="rgba(16,185,129,.10)",
                                   line=dict(width=0), name="5–95%"))
        figmc.add_trace(go.Scatter(x=x + x[::-1],
                                   y=list(bands[75]) + list(bands[25])[::-1],
                                   fill="toself", fillcolor="rgba(16,185,129,.22)",
                                   line=dict(width=0), name="25–75%"))
        figmc.add_trace(go.Scatter(x=x, y=bands[50], name="Median path",
                                   line=dict(color="#6ee7b7", width=2)))
        rng = np.random.default_rng(1)
        for i in rng.choice(paths.shape[0], 12, replace=False):
            figmc.add_trace(go.Scatter(x=x, y=paths[i], showlegend=False,
                                       line=dict(width=.7,
                                                 color="rgba(230,237,243,.25)")))
        for lvl, name, color in ((v["stop"], "Stop", "#ef4444"),
                                 (v["target"], "Target", "#10b981")):
            figmc.add_hline(y=lvl, line_dash="dot", line_color=color,
                            annotation_text=f"{name} ${lvl:,.2f}",
                            annotation_font_color=color)
        figmc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title=f"{tf} bars ahead",
                            yaxis_title="Price $", **PLOTLY_LAYOUT)
        st.plotly_chart(figmc, use_container_width=True)

        with st.expander("❓ Monte Carlo & Kelly — how to read this"):
            st.markdown(f"""
2,000 alternative futures simulated with Geometric Brownian Motion calibrated to {tkr}'s own recent drift & volatility.

- **The cone** — dark band = middle 50% of futures, light = 90%. Reality outside the cone = the market changed character.
- **P(target first) vs P(stop first)** — your *simulated win rate* for this exact trade setup.
- **CVaR 95%** — the average loss in the worst 5% of futures. The "how bad is bad" number desks size by.
- **Kelly criterion** — the mathematically optimal fraction of capital: f* = p − (1−p)/RR. Full Kelly is a wild ride; **half-Kelly** keeps ~75% of the growth at half the pain, which is why pros use it. "No edge" = the simulated odds don't justify the trade at all.
""")

        st.markdown("---")

        # ---- ⚡ B-Xtrender -----------------------------------------------------
        st.markdown("### ⚡ B-Xtrender — institutional edition")
        bx = bxtrender(df)
        div = detect_divergence(df)
        wk = weekly_alignment(df)
        bx_last = bx.iloc[-1]

        bm = st.columns(5)
        bm[0].metric("Short oscillator", f"{bx_last['short_osc']:+.1f}",
                     delta="rising" if bx_last['t3_rising'] else "falling",
                     delta_color="normal" if bx_last['t3_rising'] else "inverse")
        bm[1].metric("Long oscillator (trend)", f"{bx_last['long_osc']:+.1f}")
        wk_txt = ("—" if wk["weekly_osc"] is None else
                  f"{wk['weekly_osc']:+.1f} " + ("↑" if wk["weekly_rising"] else "↓"))
        bm[2].metric("Weekly oscillator (MTF)", wk_txt)
        aligned = (wk["weekly_osc"] is not None and
                   np.sign(wk["weekly_osc"]) == np.sign(bx_last["long_osc"]))
        bm[3].metric("Timeframes aligned", "YES ✅" if aligned else "NO ⚠️")
        div_txt = ("🐻 Bearish" if div["bearish"] else
                   "🐂 Bullish" if div["bullish"] else "None")
        bm[4].metric("Divergence", div_txt)
        if div["detail"]:
            st.caption(f"Divergence detail: {div['detail']}")

        w = df.index[-252:]
        bxw = bx.loc[w]
        so, t3l = bxw["short_osc"], bxw["t3"]
        rising_now = so > so.shift(1)
        colors_s = np.where(so > 0, np.where(rising_now, "#22ff44", "#228B22"),
                            np.where(rising_now, "#ff5555", "#8B0000"))
        lo = bxw["long_osc"]
        colors_l = np.where(lo > 0, np.where(lo > lo.shift(1), "#22ff44", "#228B22"),
                            np.where(lo > lo.shift(1), "#ff5555", "#8B0000"))

        figbx = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.06,
                              subplot_titles=("Short-term oscillator + T3 signal",
                                              "Long-term oscillator (trend)"))
        figbx.add_trace(go.Bar(x=w, y=so, marker_color=colors_s,
                               name="Short osc"), row=1, col=1)
        figbx.add_trace(go.Scatter(x=w, y=t3l, name="T3 signal",
                                   line=dict(color="#e6edf3", width=2.5)),
                        row=1, col=1)
        buys_bx = w[bxw["buy_turn"].values]
        sells_bx = w[bxw["sell_turn"].values]
        figbx.add_trace(go.Scatter(x=buys_bx, y=t3l.loc[buys_bx],
                                   mode="markers", name="Buy turn",
                                   marker=dict(color="#22ff44", size=9)),
                        row=1, col=1)
        figbx.add_trace(go.Scatter(x=sells_bx, y=t3l.loc[sells_bx],
                                   mode="markers", name="Sell turn",
                                   marker=dict(color="#ff5555", size=9)),
                        row=1, col=1)
        figbx.add_trace(go.Bar(x=w, y=lo, marker_color=colors_l,
                               name="Long osc", showlegend=False), row=2, col=1)
        figbx.add_hline(y=0, line_color="#2a3644", row=1, col=1)
        figbx.add_hline(y=0, line_color="#2a3644", row=2, col=1)
        figbx.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                            bargap=0.15, **PLOTLY_LAYOUT)
        st.plotly_chart(figbx, use_container_width=True)

        st.markdown("**📊 Event study — what actually happened after each "
                    f"signal on {tkr}:**")
        st.dataframe(event_study(df), use_container_width=True, hide_index=True)

        st.markdown("**🔬 BX Lab — probability-calibrated states "
                    f"({tkr}, 5-bar horizon):**")
        sp_tab = state_probabilities(df)
        st.dataframe(sp_tab, use_container_width=True, hide_index=True)
        st.caption(f"Current state: **{sp_tab.attrs.get('current_state','—')}**"
                   " — find it in the table for its historical odds.")
        if st.button("🧪 Run BX parameter sweep (8 presets, OOS-validated)",
                     key="bx_sweep"):
            with st.spinner("Testing 8 parameter sets, out-of-sample…"):
                sw = parameter_sweep(df)
            st.dataframe(sw, use_container_width=True, hide_index=True)
            best = sw.iloc[0]
            st.caption(f"Best OOS: **{best['preset']}** (Sharpe "
                       f"{best['OOS Sharpe']}). Watch the **overfit gap** "
                       f"column — a preset that shines in-sample and dies "
                       f"OOS is curve-fitting, and this table catches it "
                       f"in the act.")

        with st.expander("❓ What is B-Xtrender & the upgrades here?"):
            st.markdown("""
**B-Xtrender** (Bharat Jhunjhunwala, IFTA Journal) — an RSI applied to the *spread between two EMAs*, filtered by a Tillson T3 line. Faster than MACD, cleaner than raw RSI.

- **Short oscillator (top)** — bright = accelerating, dark = decelerating. Dots on the white T3 line = turn signals.
- **Long oscillator (bottom)** — the trend filter. Institutional rule: only take buy turns when it's above zero.
- **MTF alignment** — the same oscillator on weekly bars. Daily signals against the weekly trend are how retail gets chopped.
- **Divergence** — price at new highs while the oscillator makes lower highs = momentum quietly leaving.
- **Event study** — every historical turn on this exact ticker, measured: average forward return and win rate 5/10/20 days later. Trust data, not paint.
""")

        st.markdown("---")

        # ---- 📐 Price structure ---------------------------------------------------
        st.markdown("### 📐 Price structure — Fibonacci, support & resistance")
        comp = composite(df)
        figp = go.Figure()
        figp.add_trace(go.Candlestick(
            x=df.index[-252:], open=df["Open"][-252:], high=df["High"][-252:],
            low=df["Low"][-252:], close=df["Close"][-252:], name=tkr))
        for key, price in fib["levels"].items():
            figp.add_hline(y=price, line_dash="dot", line_width=1,
                           line_color=FIB_COLORS.get(key, "#8b98a5"),
                           annotation_text=f"Fib {key} — ${price:,.2f}",
                           annotation_font_size=10,
                           annotation_font_color=FIB_COLORS.get(key, "#8b98a5"))
        for lv in sr:
            col = "#22d3ee" if lv["kind"] == "resistance" else "#a78bfa"
            figp.add_hline(y=lv["price"], line_width=2, line_color=col,
                           opacity=.7,
                           annotation_text=f"{lv['kind'].upper()} "
                                           f"${lv['price']:,.2f} "
                                           f"({lv['touches']} touches)",
                           annotation_font_color=col,
                           annotation_font_size=10)
        if v["verdict"] != "NO TRADE":
            for lvl, name, color in ((v["stop"], "STOP", "#ef4444"),
                                     (v["target"], "TARGET", "#10b981")):
                figp.add_hline(y=lvl, line_width=2, line_color=color,
                               annotation_text=name,
                               annotation_font_color=color)
        buys = [d for d in comp[comp["signal"] == "BUY"].index[-252:]
                if d in df.index[-252:]]
        sells = [d for d in comp[comp["signal"] == "SELL"].index[-252:]
                 if d in df.index[-252:]]
        figp.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Low"] * 0.985,
                                  mode="markers", name="BUY zone",
                                  marker=dict(symbol="triangle-up", size=7,
                                              color="#10b981")))
        figp.add_trace(go.Scatter(x=sells, y=df.loc[sells, "High"] * 1.015,
                                  mode="markers", name="SELL zone",
                                  marker=dict(symbol="triangle-down", size=7,
                                              color="#ef4444")))
        figp.update_layout(height=580, xaxis_rangeslider_visible=False,
                           margin=dict(l=10, r=10, t=30, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figp, use_container_width=True)

        swing_dir = "upswing" if fib["up_swing"] else "downswing"
        with st.expander("❓ Fibonacci + support/resistance — how to use"):
            st.markdown(f"""
Dominant swing detected: **{swing_dir}** from ${fib['swing_low']:,.2f} to ${fib['swing_high']:,.2f}.

- **Fibonacci retracements** — how much of a move typically gets "given back": 0.382/0.5 = shallow (strong trend), **0.618 = the golden pocket** where healthy pullbacks end, 0.786 = last defence.
- **Support/resistance** (cyan/purple lines) — price zones where swing pivots *clustered*; the touch count shows how battle-tested each level is. Fib level + S/R level + your stop all in one zone = a level that actually matters.
""")

        st.markdown("---")

        # ---- 📅 Seasonality + 🏦 Fundamentals (from the feed) ------------------
        colS, colF = st.columns([1.3, 1])
        with colS:
            st.markdown("### 📅 Seasonality — Detrick-style stats")
            seas = monthly_seasonality(fetch_history(tkr, period="10y"))
            if len(seas):
                import datetime as _dt
                cur_m = _dt.date.today().strftime("%b")
                def _hl(row):
                    return ["background-color: rgba(16,185,129,.18)"
                            if row.name == cur_m else "" for _ in row]
                st.dataframe(seas.style.apply(_hl, axis=1)
                             .background_gradient(subset=["avg return %"],
                                                  cmap="RdYlGn"),
                             use_container_width=True, height=460)
                if cur_m in seas.index:
                    r = seas.loc[cur_m]
                    st.caption(f"**{cur_m} historically:** up "
                               f"{r['win rate %']:.0f}% of years, average "
                               f"{r['avg return %']:+.2f}% "
                               f"({int(r['years'])} years of data).")
            else:
                st.info("Not enough history for seasonality stats.")
        with colF:
            st.markdown("### 🏦 Fundamentals check")
            st.caption("\"Price follows growth & margin & free cash flow\"")
            fund = fundamental_snapshot(tkr)
            if fund and fund.get("quality_score"):
                f1, f2 = st.columns(2)
                rg = fund.get("revenue_growth")
                gm = fund.get("gross_margin")
                om = fund.get("op_margin")
                fy = fund.get("fcf_yield")
                pe = fund.get("fwd_pe")
                f1.metric("Revenue growth (yoy)",
                          f"{rg*100:.1f}%" if rg is not None else "—")
                f2.metric("Gross margin",
                          f"{gm*100:.1f}%" if gm is not None else "—")
                f1.metric("Operating margin",
                          f"{om*100:.1f}%" if om is not None else "—")
                f2.metric("FCF yield",
                          f"{fy*100:.1f}%" if fy is not None else "—")
                f1.metric("Forward P/E",
                          f"{pe:.1f}" if pe else "—")
                f2.metric("Quality score", fund["quality_score"])
            else:
                st.info("Fundamentals unavailable for this ticker (ETFs "
                        "have none; some names return partial data).")
            with st.expander("❓ What are these & the thresholds?"):
                st.markdown("""
The 4-point quality check (1 point each): revenue growth ≥ 10%, gross margin ≥ 40%, operating margin ≥ 15%, FCF yield ≥ 3%. Quality growth compounders tend to score 3-4; melting ice cubes and story-stocks score 0-1. A great technical setup on a 0/4 business deserves a smaller size and a shorter leash. Seasonality: the highlighted row is the current month — a headwind or tailwind stat, never a signal by itself.
""")

        st.markdown("---")

        # ---- 🧠 Model breakdown
        st.markdown("### 🧠 Model breakdown")
        last = comp.iloc[-1]
        sub = last[["trend", "momentum", "bxtrender", "macd", "rsi",
                    "meanrev", "volume"]]
        sub_df = pd.DataFrame(
            {"model": [str(k) for k in sub.index],
             "score": [float(xv) for xv in sub.values]}).set_index("model")
        st.bar_chart(sub_df, height=220)
        with st.expander("❓ What does each model measure?"):
            st.markdown("""
- **trend** — price vs 50 & 200-day averages + golden/death cross. The big picture.
- **momentum** — the academic "12-1" factor: past-year performance excluding the last month. Winners keep winning.
- **bxtrender** — double-smoothed momentum (RSI of an EMA spread, T3-filtered). Less lag than MACD.
- **macd** — momentum of momentum; catches acceleration early.
- **rsi** — which side is dominant right now (above/below 50), fading extremes.
- **meanrev** — Bollinger z-score; stretched rubber bands snap back.
- **volume** — do volume surges confirm the move? Moves without volume are suspect.

Bars pointing the same way = quality signal. Bars fighting = chop = usually NO TRADE.
""")

    _desk_live()

# ===========================================================================
# 2. SCREENER
# ===========================================================================
with tab_screener:
    st.subheader("Scan the market")
    col1, col2 = st.columns([3, 1])
    with col1:
        custom = st.text_input(
            "Tickers (comma-separated) — empty = default 50-name universe",
            placeholder="AAPL, NVDA, SPY ...")
    with col2:
        min_score = st.slider("Min |score| filter", 0.0, 0.8, 0.0, 0.05)

    universe = tuple(t.strip().upper() for t in custom.split(",")
                     if t.strip()) or tuple(DEFAULT_UNIVERSE)

    if st.button("Run scan", type="primary"):
        with st.spinner(f"Downloading & scoring {len(universe)} tickers…"):
            data = fetch_many(universe, period="2y")
            spy_ret3m = None
            try:
                spy = fetch_history("SPY", period="1y")
                spy_ret3m = float(spy["Close"].pct_change(63).iloc[-1] * 100)
            except Exception:
                pass
            rows = []
            for sym, dfr in data.items():
                try:
                    snap = latest_snapshot(dfr)
                    snap["ticker"] = sym
                    if spy_ret3m is not None and snap.get("ret_3m") is not None:
                        snap["rs_vs_spy"] = round(snap["ret_3m"] - spy_ret3m, 1)
                    rows.append(snap)
                except Exception:
                    continue
        if not rows:
            st.error("No data returned — check tickers or try again shortly.")
        else:
            table = pd.DataFrame(rows).set_index("ticker")
            table = table[abs(table["score"]) >= min_score]
            st.session_state["scan"] = table.sort_values("score",
                                                         ascending=False)

    if "scan" in st.session_state:
        table = st.session_state["scan"]
        n_buy = int((table["signal"] == "BUY").sum())
        n_sell = int((table["signal"] == "SELL").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("BUY signals", n_buy)
        c2.metric("SELL signals", n_sell)
        c3.metric("HOLD", int(len(table)) - n_buy - n_sell)

        def _style(vv):
            return f"color: {SIGNAL_COLORS.get(vv, '')}; font-weight: 700"

        st.dataframe(
            table.style.map(_style, subset=["signal"]).background_gradient(
                subset=["score"], cmap="RdYlGn", vmin=-0.6, vmax=0.6),
            use_container_width=True, height=560)
        with st.expander("❓ How to read this table"):
            st.markdown("""
- **score** — the fused 7-model output, −1 to +1, dampened in vol storms. ≥ +0.25 BUY zone, ≤ −0.25 SELL zone.
- **rs_vs_spy** — 3-month return minus SPY's: is this name actually *beating the market*, or just floating with it? Institutions buy leaders, not laggards.
- **off_52w_high** — distance from the 52-week high. Research says strength near highs (0 to −10%) keeps working; −40% "bargains" usually aren't.
- **atr** — daily range in $; wilder stock = smaller position for the same risk.

Workflow: scan → pick extremes → **🎯 Trade desk** for the full verdict. Never trade off the scan alone.
""")

# ===========================================================================
# 3. BACKTEST
# ===========================================================================
with tab_backtest:
    st.subheader("Backtest the signal on any ticker")
    c1, c2, c3, c4, c5 = st.columns(5)
    bt_tkr = c1.text_input("Ticker", value="AAPL", key="bt").upper().strip()
    bt_tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="bt_tf")
    bt_cash = c3.number_input("Starting cash $", 500, 1_000_000, 5000,
                              step=500)
    bt_risk = c4.slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)
    bt_mode = c5.selectbox("Engine", ["auto", "core", "trend", "dip",
                                      "blend"], index=0,
                           help="auto picks per ticker by Hurst exponent")
    bt_short = st.checkbox("Allow shorts (trend mode, below 200-SMA only)",
                           value=False, key="bt_short")

    if st.button("Run backtest", type="primary", key="btrun"):
        _bm = tf_meta(bt_tf)
        df = fetch_tf(bt_tkr, bt_tf)
        if len(df) < _bm["min_bars"]:
            st.error(f"Not enough {bt_tf} bars (need ~{_bm['min_bars']}).")
        else:
            cfg = BTConfig(starting_cash=float(bt_cash),
                           risk_per_trade=bt_risk / 100, mode=bt_mode,
                           allow_short=bt_short,
                           bars_per_year=_bm["bars_per_year"])
            res = run_backtest(df, cfg)
            st.caption(f"Engine: **{res.mode_used.upper()}** on **{bt_tf}** "
                       f"bars — Sharpe/CAGR annualized with "
                       f"{_bm['bars_per_year']} bars/yr"
                       f"{' · mode picked by Hurst' if bt_mode == 'auto' else ''}")
            _remember("backtest", {"ticker": bt_tkr,
                                   "mode": res.mode_used,
                                   "sharpe": res.metrics.get("Sharpe"),
                                   "tf": bt_tf})
            cols = st.columns(5)
            for col, (k, val) in zip(cols * 2, res.metrics.items()):
                col.metric(k, val if val is not None else "—")

            with st.expander("📉 Why is my Sharpe low? (read this once)"):
                st.markdown("""
Three usual suspects, in order of impact:
1. **Cash drag** — risking 1%/trade with a 2.5×ATR stop deploys only ~15-25% of capital; the rest earns nothing while buy & hold is 100% invested. **Fix: the CORE engine** — near-fully invested while the regime is healthy (above 200-SMA + B-Xtrender positive), cash when it breaks. B&H-like CAGR in bulls, a fraction of the drawdown in bears.
2. **Single-name concentration** — one stock's noise dominates. Sharpe rises with diversification faster than with better signals; that's what the 🧬 Alpha engine's multi-position plan is for.
3. **Trend systems are streaky** — 40% win rates with occasional big winners produce lumpy equity. The DIP engine smooths it (77%+ win rate, small wins). A **blend** of core + dip is how small systematic accounts actually maximize Sharpe.
""")

            with st.expander("⚙️ What's inside the v2 engine?"):
                st.markdown("""
Two strategies, auto-selected per ticker by its **Hurst exponent**:
- **TREND** — follows the 7-model composite signal. For tickers whose moves *continue* (H > 0.5).
- **DIP** — Connors-style RSI(2) pullback buyer: buys short-term panic **inside an uptrend**, exits on the snap-back or a strict time limit — now with a **Fibonacci pocket filter**: panic is only bought when price sits inside the 0.382–0.786 retracement zone of the dominant swing (or on a B-Xtrender buy-turn). Confluence, not just oversold.
- **CORE** — improved buy & hold: ~fully invested while price > 200-SMA **and** the B-Xtrender long oscillator is positive; steps to cash when either breaks. Fixes the cash-drag problem that makes signal strategies lose to B&H in bull markets.
- **Shorts (optional)** — trend mode can short below the 200-SMA when the composite says SELL and B-Xtrender confirms (falling & negative). Symmetric stops, breakeven and time exits.
- **B-Xtrender confirmation** — trend longs now require the long oscillator positive AND the T3 rising. Fewer, better entries.

Risk mechanics on every trade:
- **Regime gate (Faber 2007)** — longs only above the 200-day SMA. No knife-catching.
- **Breakeven stop after +1R** — once the trade is one risk-unit in profit, the stop jumps to entry. Converts would-be losers into scratches → directly raises win rate.
- **Time stop** — not working within 10–20 bars? Out. Dead capital is a cost.
- **Volatility-targeted sizing (Moreira & Muir 2017)** — risk shrinks automatically when the stock's ATR% is elevated vs its own history.
- Chandelier 2.5×ATR trail on trend trades, next-bar-open execution, commissions included.
""")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     name="Strategy",
                                     line=dict(width=2, color="#10b981")))
            fig.add_trace(go.Scatter(x=res.bh_equity.index, y=res.bh_equity,
                                     name="Buy & Hold",
                                     line=dict(width=1.5, dash="dot",
                                               color="#8b98a5")))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Equity $", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown + rolling Sharpe
            dd = res.equity / res.equity.cummax() - 1
            roll = res.equity.pct_change().rolling(63)
            rsharpe = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
            cA, cB = st.columns(2)
            with cA:
                figd = go.Figure(go.Scatter(x=dd.index, y=dd * 100,
                                            fill="tozeroy",
                                            line=dict(color="#ef4444"),
                                            name="Drawdown %"))
                figd.update_layout(height=280, title="Drawdown %",
                                   margin=dict(l=10, r=10, t=40, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figd, use_container_width=True)
            with cB:
                figs2 = go.Figure(go.Scatter(x=rsharpe.index, y=rsharpe,
                                             line=dict(color="#22d3ee"),
                                             name="Rolling Sharpe"))
                figs2.add_hline(y=0, line_color="#2a3644")
                figs2.update_layout(height=280, title="Rolling Sharpe (3m)",
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(figs2, use_container_width=True)

            # Monthly returns heatmap
            mrets = res.equity.resample("ME").last().pct_change().dropna()
            if len(mrets) >= 6:
                hm = pd.DataFrame({
                    "year": mrets.index.year,
                    "month": mrets.index.strftime("%b"),
                    "ret": mrets.values * 100})
                order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot = hm.pivot_table(index="year", columns="month",
                                       values="ret").reindex(columns=order)
                fighm = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns,
                    y=[str(y) for y in pivot.index],
                    colorscale="RdYlGn", zmid=0,
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text}", showscale=False))
                fighm.update_layout(height=90 + 45 * len(pivot),
                                    title="Monthly returns % (strategy)",
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(fighm, use_container_width=True)

            with st.expander("❓ What do these metrics & charts mean?"):
                st.markdown("""
- **CAGR** — the "interest rate" your strategy earned. **Sharpe** — return per unit of risk (~1 decent, 2+ = check for bugs 😉). **Sortino** — punishes only downside wiggle. **Max Drawdown** — worst peak-to-valley; the number that decides if you'd *survive* running this.
- **Drawdown chart** — how deep and how *long* the underwater periods were. Long flat red = the psychological killer.
- **Rolling Sharpe** — is the edge steady or was it one lucky quarter?
- **Monthly heatmap** — seasonality and consistency at a glance. A good system is boringly green, not one giant month.
""")

            st.markdown("#### Walk-forward check (4 sequential folds)")
            st.dataframe(walk_forward(df, cfg), use_container_width=True)

            # ---- 🔬 Statistical validation ------------------------------
            st.markdown("### 🔬 Is this edge REAL? (institutional validation)")
            st.caption("The tests that separate genuine edges from the "
                       "thousands you'd find by data-mining. This is what "
                       "makes a quant trust — or discard — a backtest.")
            N_TRIALS = 20   # we test ~20 models across the site; be honest about it
            rets_bt = res.equity.pct_change().dropna()
            sharpe_bt = res.metrics.get("Sharpe", 0) or 0
            tstat_bt = (sharpe_bt * np.sqrt(len(rets_bt) / 252)
                        if len(rets_bt) > 252 else sharpe_bt)

            vc1, vc2 = st.columns(2)
            with vc1:
                dsr = deflated_sharpe(sharpe_bt, N_TRIALS, len(rets_bt))
                if "error" not in dsr:
                    st.markdown(f"**Deflated Sharpe** — {dsr['verdict']}")
                    st.caption(f"Observed {dsr['observed_sharpe']} vs "
                               f"noise-benchmark {dsr['deflated_benchmark_ann']} "
                               f"· P(real) = {dsr['DSR_probability']}")
                perm = permutation_test(rets_bt)
                if "error" not in perm:
                    st.markdown(f"**Permutation test** — {perm['verdict']}")
                    st.caption(f"Real Sharpe {perm['actual_sharpe']} vs luck's "
                               f"95th pct {perm['perm_sharpe_95pct']} · "
                               f"p = {perm['perm_p_value']}")
            with vc2:
                hc = haircut_pvalue(tstat_bt, N_TRIALS)
                st.markdown(f"**Multiple-testing haircut** — {hc['verdict']}")
                st.caption(f"Raw p {hc['raw_p']} → after correcting for "
                           f"{N_TRIALS} models: {hc['bonferroni_p']}")
                if len(res.trades):
                    bs = bootstrap_cagr(res.trades["pnl"],
                                        starting=float(bt_cash))
                    if "error" not in bs:
                        st.markdown(f"**Bootstrap 90% CI** — {bs['verdict']}")
                        st.caption(f"Return CI: {bs['CI90_low_%']}% … "
                                   f"{bs['median_return_%']}% … "
                                   f"{bs['CI90_high_%']}%")

            with st.expander("❓ Why these four tests are the real grade"):
                st.markdown("""
Anyone can produce a pretty backtest — try 1,000 parameter combos and *one* will look brilliant by pure chance. These tests fight that:

- **Deflated Sharpe** (Bailey & López de Prado 2014) — lowers your Sharpe to account for how many strategies were tried. A Sharpe of 1.5 from testing 20 models is worth far less than 1.5 from testing one.
- **Multiple-testing haircut** (Harvey, Liu & Zhu 2016) — a t-stat of 2 (the classic "significant") is NOT significant when mined across 20 signals. This corrects it.
- **Permutation test** — randomly flips the sign of each day's return 500× to build the distribution of "luck." If your real Sharpe isn't clearly above that cloud, you have nothing.
- **Bootstrap CI** — resamples your trades 1,000× for a 90% confidence band on returns. If the band straddles zero, you genuinely don't know if the strategy works — and that knowledge is worth more than false confidence.

**If a strategy passes all four, it's in rarer air than 99% of what retail traders trade on.** If it fails — the app just saved you from a mirage. That honesty is the highest-value thing this whole site does.
""")
            with st.expander("❓ Why walk-forward matters"):
                st.markdown("""
The same rules re-run on 4 separate sequential periods. A real edge shows in most folds; a curve-fit illusion shines in one and dies in the rest. The single best overfitting detector available to a retail quant.
""")

            if len(res.trades):
                st.markdown("#### 🗺️ Every trade on the chart — "
                            "entry, exit, price, reason")
                tr = res.trades.copy()
                figtr = go.Figure()
                figtr.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name=bt_tkr,
                    increasing_line_color="#2a3644",
                    decreasing_line_color="#1a222c",
                    increasing_fillcolor="#2a3644",
                    decreasing_fillcolor="#1a222c"))
                figtr.add_trace(go.Scatter(
                    x=pd.to_datetime(tr["entry_date"]), y=tr["entry"],
                    mode="markers", name="Entry",
                    marker=dict(symbol="triangle-up", size=12,
                                color="#10b981",
                                line=dict(width=1, color="#0b0f14")),
                    customdata=tr[["entry"]].values,
                    hovertemplate="ENTRY @ $%{y:.2f}<br>%{x|%Y-%m-%d}"
                                  "<extra></extra>"))
                exit_colors = tr["reason"].map(
                    {"stop": "#ef4444", "breakeven": "#f59e0b",
                     "time": "#8b98a5", "signal": "#22d3ee",
                     "target(rsi)": "#6ee7b7"}).fillna("#e6edf3")
                figtr.add_trace(go.Scatter(
                    x=pd.to_datetime(tr["exit_date"]), y=tr["exit"],
                    mode="markers", name="Exit",
                    marker=dict(symbol="triangle-down", size=12,
                                color=exit_colors,
                                line=dict(width=1, color="#0b0f14")),
                    customdata=np.stack([tr["reason"], tr["pnl"]], axis=-1),
                    hovertemplate="EXIT @ $%{y:.2f}<br>reason: %{customdata[0]}"
                                  "<br>P&L: $%{customdata[1]}<br>%{x|%Y-%m-%d}"
                                  "<extra></extra>"))
                # connect entry->exit with a thin win/loss colored line
                for _, t_ in tr.iterrows():
                    figtr.add_trace(go.Scatter(
                        x=[pd.to_datetime(t_["entry_date"]),
                           pd.to_datetime(t_["exit_date"])],
                        y=[t_["entry"], t_["exit"]], mode="lines",
                        line=dict(width=1.2,
                                  color="rgba(16,185,129,.55)" if t_["pnl"] > 0
                                  else "rgba(239,68,68,.55)"),
                        showlegend=False, hoverinfo="skip"))
                figtr.update_layout(height=520,
                                    xaxis_rangeslider_visible=False,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(figtr, use_container_width=True)
                st.caption("▲ green = entry · ▼ exit colored by reason "
                           "(🔴 stop, 🟠 breakeven, 🔵 signal, 🟢 RSI target, "
                           "⚪ time) · connecting line green = winner, "
                           "red = loser. Hover any marker for exact price, "
                           "date, reason and P&L.")

                colR, colM = st.columns(2)
                with colR:
                    if "R" in res.trades:
                        figR = go.Figure(go.Histogram(
                            x=res.trades["R"], nbinsx=24,
                            marker_color=np.where(
                                np.histogram(res.trades["R"], bins=24)[1][:-1]
                                >= 0, "#10b981", "#ef4444")))
                        figR.add_vline(x=0, line_color="#8b98a5")
                        figR.update_layout(height=300,
                                           title="R-multiple distribution",
                                           xaxis_title="R", yaxis_title="trades",
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           **PLOTLY_LAYOUT)
                        st.plotly_chart(figR, use_container_width=True)
                with colM:
                    if "MAE_R" in res.trades:
                        figM = go.Figure(go.Scatter(
                            x=res.trades["MAE_R"], y=res.trades["MFE_R"],
                            mode="markers",
                            marker=dict(size=9,
                                        color=np.where(res.trades["pnl"] > 0,
                                                       "#10b981", "#ef4444"))))
                        figM.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                       line=dict(color="#8b98a5", dash="dot"))
                        figM.update_layout(height=300,
                                           title="MAE vs MFE (per trade, in R)",
                                           xaxis_title="Max pain (MAE, R)",
                                           yaxis_title="Max gain (MFE, R)",
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           **PLOTLY_LAYOUT)
                        st.plotly_chart(figM, use_container_width=True)
                with st.expander("❓ R-distribution & MAE/MFE — pro trade forensics"):
                    st.markdown("""
- **R-multiple distribution** — every trade's P&L in risk units. A healthy system: losses clustered at −1R (stops doing their job), a right tail of +2R/+3R winners. Losses beyond −1R = slippage/gap problem; no right tail = you're cutting winners.
- **MAE vs MFE** — each dot is one trade: how far it went AGAINST you (x) vs FOR you (y). Green dots high-left = ideal (little pain, much gain). Red dots that reached high MFE = winners you gave back → tighten trailing. Many reds with tiny MAE = stops too tight; they died without ever being wrong.
This is the same trade-forensics workflow a Bloomberg BTST user runs.
""")

                st.markdown("#### Trade log")
                st.dataframe(res.trades, use_container_width=True, height=300)

                # ---- 🛡️ Risk-of-ruin on THIS strategy's actual stats ------
                wins_ = res.trades[res.trades["pnl"] > 0]["pnl"]
                losses_ = -res.trades[res.trades["pnl"] < 0]["pnl"]
                if len(wins_) >= 3 and len(losses_) >= 3:
                    ror = risk_of_ruin(
                        win_rate=len(wins_) / len(res.trades),
                        avg_win=float(wins_.mean()),
                        avg_loss=float(losses_.mean()),
                        risk_per_trade_pct=bt_risk)
                    if ror:
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Risk of 30% drawdown",
                                  f"{ror['prob_of_ruin_%']}%",
                                  delta=ror["verdict"], delta_color="off")
                        r2.metric("Payoff ratio (avg W/L)",
                                  ror["payoff_ratio"])
                        r3.metric("Expectancy per trade",
                                  f"{ror['expectancy_R']:+.2f}R")
                        kl = kelly_ladder(len(wins_) / len(res.trades),
                                          ror["payoff_ratio"])
                        r4.metric("Kelly (full/half/¼)",
                                  f"{kl['full_kelly_%']}/{kl['half_kelly_%']}"
                                  f"/{kl['quarter_kelly_%']}%"
                                  if kl["edge"] else "No edge")
                        st.caption("Risk of ruin: 5,000 Monte Carlo careers "
                                   "of 200 trades each with THIS strategy's "
                                   "real win rate and payoff, at your chosen "
                                   "risk %. The single most important "
                                   "number on this page.")

# ===========================================================================
# 4. OPTIONS / IV SURFACE
# ===========================================================================
with tab_options:
    st.subheader("Implied volatility surface")
    c1, c2 = st.columns([2, 1])
    opt_tkr = c1.text_input("Ticker (must have listed options)", value="SPY",
                            key="opt").upper().strip()
    n_exp = c2.slider("Expiries to load", 3, 12, 8)

    if st.button("Build surface", type="primary", key="optrun"):
        with st.spinner(f"Downloading {n_exp} option chains for {opt_tkr}…"):
            spot, chain = fetch_chains(opt_tkr, max_expiries=n_exp)

        if chain.empty:
            st.error(f"No usable options data for {opt_tkr}. Try SPY, QQQ, "
                     "AAPL, NVDA or another liquid name.")
        else:
            ts = atm_term_structure(chain)
            atm_iv = float(ts["iv"].iloc[0]) if len(ts) else None
            skew = skew_25(chain)
            pcr = put_call_ratio(chain)
            near_exp = sorted(chain["expiry"].unique())[0]
            mp = max_pain(chain, near_exp)
            exp_move = None
            if atm_iv:
                t_yrs = float(ts["dte"].iloc[0]) / 365.0
                exp_move = 0.8 * spot * (atm_iv / 100) * np.sqrt(t_yrs)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Spot", f"${spot:,.2f}")
            m2.metric("ATM IV", f"{atm_iv:.1f}%" if atm_iv else "—")
            m3.metric("Expected move", f"±${exp_move:,.2f}" if exp_move else "—")
            m4.metric("Skew (95P−105C)",
                      f"{skew:+.1f}" if skew is not None else "—")
            m5.metric("Put/Call OI ratio", pcr if pcr is not None else "—")
            m6.metric(f"Max pain ({near_exp[5:]})",
                      f"${mp:,.0f}" if mp else "—")

            with st.expander("❓ What do these numbers mean?"):
                st.markdown(f"""
- **ATM IV** — the market's own volatility forecast, backed by real money.
- **Expected move** — Black-Scholes straddle approximation (≈ 0.8 × price × IV × √time): how far the market prices {opt_tkr} to move by the nearest expiry. Targets outside it = betting against the options market.
- **Skew** — downside puts vs upside calls. Large positive = crash insurance in demand.
- **Put/Call OI ratio** — total put open interest ÷ calls. Extremes are contrarian: >1.2 = fear (often near bottoms), <0.6 = greed.
- **Max pain** — the strike where option *holders* lose the most at expiry. Price often gravitates toward it into expiration week (dealers hedging), a real but modest effect.
""")


            # ---- 💡 EDGE FINDER — our models vs the options market ------------
            st.markdown("### 💡 Edge finder — what WE forecast vs what "
                        "OPTIONS price")
            df_u = fetch_history(opt_tkr, period="2y")
            g_u = garch_forecast(df_u) if len(df_u) > 260 else {}
            ew_u = ewma_vol(df_u) if len(df_u) > 60 else {}
            vr = vrp(atm_iv, g_u.get("sigma_annual_pct"),
                     ew_u.get("sigma_annual_pct")) if atm_iv else {}
            rich = iv_richness(df_u, atm_iv) if atm_iv else {}

            mem_d = st.session_state.get("memory", {}).get("desk", {})
            if mem_d.get("ticker") == opt_tkr:
                direction = mem_d["verdict"]
                dir_src = f"🧠 from your Trade-desk run (conviction {mem_d['conviction']})"
            else:
                try:
                    v_q = analyze(df_u)
                    direction = v_q["verdict"]
                    dir_src = "computed fresh by the 7-model verdict engine"
                except Exception:
                    direction, dir_src = "NO TRADE", "unavailable"

            e1, e2, e3, e4 = st.columns(4)
            if vr:
                e1.metric("IV vs our vol forecast",
                          f"{vr['iv']}% vs {vr['forecast_vol']}%",
                          delta=f"VRP {vr['vrp_pts']:+.1f} pts",
                          delta_color="off")
            if rich:
                e2.metric("IV richness percentile", f"{rich['iv_pctile']}%",
                          help="Where today's IV sits vs this ticker's own "
                               "1-year realized-vol distribution.")
            e3.metric("Directional view", direction, delta=dir_src,
                      delta_color="off")
            mvm = {}
            try:
                paths_o = simulate(df_u, days=int(ts["dte"].iloc[0]),
                                   n_paths=2000)
                mvm = move_vs_model(exp_move, paths_o, spot,
                                    int(ts["dte"].iloc[0]))
            except Exception:
                pass
            if mvm:
                e4.metric("Move: market vs model",
                          f"±${mvm['market_move']} vs ±${mvm['model_move']}",
                          delta=mvm["read"], delta_color="off")

            if vr:
                st.markdown(f"**Vol verdict: {vr['state']}**")
                sug = suggest_structure(direction, vr["state"], chain,
                                        near_exp, spot, bs_greeks)
                st.success(f"**🎯 Suggested structure: {sug['name']}**  \n"
                           f"`{sug['legs']}`  \n{sug['logic']}")
                _remember("options", {"ticker": opt_tkr,
                                      "vol_state": vr["state"],
                                      "skew": skew,
                                      "atm_iv": atm_iv})
            with st.expander("❓ Where the options edge actually comes from"):
                st.markdown("""
The one durable, research-backed edge in listed options is the **variance risk premium** (Carr & Wu 2009): implied vol *persistently* overprices realized vol, because the world pays up for insurance. Everything in this panel is that comparison, done properly:

- **IV vs our forecast** — ATM implied vol against a GARCH(1,1) + EWMA blend forecast of what vol will actually be. Gap > +4 pts = the market is overpaying for options → *selling* structures have tailwind. Negative gap = options are statistically cheap → *own* them.
- **IV richness percentile** — level lies, rank doesn't. 90th percentile IV on a boring stock beats 40% IV on a meme stock.
- **Move: market vs model** — the straddle's expected move against our own 2,000-path Monte Carlo at the same horizon. Disagreement = someone is wrong; the panel tells you which side to take.
- **The structure suggester** fuses your **directional view** (from the Trade desk — the tabs share memory) with the **vol state** into one concrete trade with delta-picked strikes: bullish+rich IV → put credit spread (get *paid* to be long); bullish+cheap IV → call debit spread (own the move at a discount); no direction+rich IV → iron condor (harvest the premium). Direction, vol, and structure must all agree — that's the whole edge.

⚠️ Honesty: strikes are delta-suggestions from delayed data — always check live quotes, and spreads on Blink need options approval. Defined-risk structures only; never naked short options on a $5K account.
""")

            st.markdown("---")

            # ---- 🐋 Dealer gamma & whale flow ---------------------------------
            st.markdown("### 🐋 Dealer gamma exposure (GEX) & whale flow")
            prof = gex_profile(chain, spot)
            summ = gex_summary(prof, spot)
            if summ:
                gm1, gm2, gm3, gm4, gm5 = st.columns(5)
                gm1.metric("Net GEX", f"${summ['net_gex_m']:,.0f}M")
                gm2.metric("Regime", summ["regime"].split(" ")[0] + " " +
                           summ["regime"].split(" ")[1])
                gm3.metric("Call wall", f"${summ['call_wall']:,.0f}")
                gm4.metric("Put wall", f"${summ['put_wall']:,.0f}")
                gm5.metric("Gamma flip",
                           f"${summ['flip']:,.0f}" if summ["flip"] else "—",
                           delta=summ["spot_vs_flip"], delta_color="off")

                colors_gex = np.where(prof["gex_m"] >= 0, "#10b981", "#ef4444")
                figg = go.Figure(go.Bar(x=prof["strike"], y=prof["gex_m"],
                                        marker_color=colors_gex,
                                        name="Net GEX $M"))
                figg.add_vline(x=spot, line_dash="dot", line_color="#e6edf3",
                               annotation_text=f"spot ${spot:,.0f}")
                if summ["flip"]:
                    figg.add_vline(x=summ["flip"], line_dash="dash",
                                   line_color="#f59e0b",
                                   annotation_text="flip")
                figg.update_layout(height=380, xaxis_title="Strike",
                                   yaxis_title="Net GEX ($M per 1% move)",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figg, use_container_width=True)

                with st.expander("❓ GEX, walls & the flip — the flow-trader playbook"):
                    st.markdown("""
Dealers who sell options must hedge them by trading the stock — mechanically, without opinion. **GEX** estimates how much:

- **Positive net GEX (🧲 pinning)** — dealers buy dips and sell rips. Price gets *magnetized* between the walls; ranges, boring days, failed breakouts.
- **Negative net GEX (⛽ vol fuel)** — dealers must sell INTO drops and buy INTO rallies, amplifying every move. Crashes and face-rippers live here.
- **Call wall** — the strike with peak positive gamma; acts as resistance/pin magnet into expiry.
- **Put wall** — peak negative gamma; the air-pocket level where support turns to acceleration.
- **Gamma flip** — the spot level where the regime changes. Above it = stable zone, below = unstable. Watch what happens when price approaches it.

This is the same math behind the paid flow dashboards — computed from public open interest.
""")

            st.markdown("### 🔥 Unusual activity (fresh positioning)")
            uf = unusual_flow(chain)
            if len(uf):
                st.dataframe(uf, use_container_width=True, hide_index=True,
                             height=380)
                with st.expander("❓ How to read the flow table"):
                    st.markdown("""
- **vol/oi > 1** 🔥 — more contracts traded today than existed before: someone is OPENING a fresh position, not closing an old one. That's the signature flow-traders hunt.
- **premium $** — the actual money behind it. A million in premium is conviction; $5K is noise.
- Caveats the paid services rarely mention: you can't see if it's a buy or a sell from delayed data, and big prints are often hedges or spread legs. Treat as *context*, never as a signal alone.
""")
            else:
                st.info("No contracts with meaningful volume right now "
                        "(quiet session or delayed data).")

            strikes, dtes, grid = build_surface(chain)
            if grid.size:
                fig = go.Figure(data=[go.Surface(
                    x=strikes, y=dtes, z=grid, colorscale="Jet",
                    colorbar=dict(title="IV %"), connectgaps=True)])
                fig.update_layout(
                    height=620,
                    scene=dict(xaxis_title="Strike",
                               yaxis_title="Days to expiry",
                               zaxis_title="Implied vol (%)",
                               camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.7))),
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"{opt_tkr} IV surface — spot ${spot:,.2f}",
                    **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("❓ How to read the 3D surface"):
                    st.markdown("""
Every point = one option's implied vol (strike × expiry). Pros look for: the **smile/smirk** across strikes (steep left wing = crash fear), the **term slope** across expiries (inverted = near-term event fear), and **bumps** at specific expiries = event risk priced exactly there.
""")

            colA, colB = st.columns(2)
            with colA:
                exps = sorted(chain["expiry"].unique())
                pick = st.selectbox("Smile for expiry", exps, index=0)
                sm = chain[chain["expiry"] == pick]
                figs = go.Figure()
                for side, color in (("P", "#ef4444"), ("C", "#10b981")):
                    s = sm[sm["type"] == side].groupby("strike")["iv"].median()
                    figs.add_trace(go.Scatter(
                        x=s.index, y=s.values, mode="lines+markers",
                        name="Puts" if side == "P" else "Calls",
                        line=dict(color=color)))
                figs.add_vline(x=spot, line_dash="dot", annotation_text="spot")
                figs.update_layout(height=360, xaxis_title="Strike",
                                   yaxis_title="IV %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figs, use_container_width=True)
            with colB:
                st.markdown("**ATM term structure**")
                figt = go.Figure(go.Scatter(x=ts["dte"], y=ts["iv"],
                                            mode="lines+markers",
                                            line=dict(color="#6ee7b7")))
                figt.update_layout(height=360, xaxis_title="Days to expiry",
                                   yaxis_title="ATM IV %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figt, use_container_width=True)
                st.caption("Upward slope = calm; inverted = near-term event fear.")

            st.markdown("#### Option chain + Greeks (selected expiry)")
            sel = chain[chain["expiry"] == pick].copy()
            g = bs_greeks(spot, sel["strike"].values,
                          sel["dte"].values / 365.0,
                          sel["iv"].values / 100.0,
                          (sel["type"] == "C").values)
            sel = pd.concat([sel.reset_index(drop=True), g], axis=1)
            show_cols = ["type", "strike", "last", "bid", "ask", "iv",
                         "volume", "oi", "delta", "gamma", "vega", "theta"]
            st.dataframe(sel[show_cols].sort_values(["type", "strike"]),
                         use_container_width=True, height=380)
            with st.expander("❓ Greeks cheat-sheet"):
                st.markdown("""
- **Delta** — $ change per $1 stock move; also ≈ probability of expiring in the money.
- **Gamma** — how fast delta changes; high gamma = behaviour flips fast near the strike.
- **Vega** — $ change per IV point; post-earnings IV crush is vega bleeding.
- **Theta** — $ lost per day to time decay; the rent you pay to hold.

Data ~15 min delayed (Yahoo). Educational, not advice.
""")



# ===========================================================================
# 7. PORTFOLIO & PAIRS (awesome-quant: PyPortfolioOpt, statsmodels)
# ===========================================================================
with tab_pp:
    st.subheader("⚖️ Portfolio optimizer")
    st.caption("PyPortfolioOpt: Max-Sharpe (Markowitz), Min-Vol, and HRP — "
               "Hierarchical Risk Parity (López de Prado 2016), the robust "
               "one that needs no return forecasts.")
    pp_in = st.text_input("Tickers (comma-separated, 3+)",
                          value="AAPL, MSFT, NVDA, JPM, XOM, GLD",
                          key="pp_in")
    pp_acct = st.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="pp_acct")
    if st.button("Optimize", type="primary", key="pp_run"):
        tks = tuple(t.strip().upper() for t in pp_in.split(",") if t.strip())
        with st.spinner("Downloading & optimizing…"):
            data = fetch_many(tks, period="2y")
            px = build_prices(data)
            res = optimize(px, account=float(pp_acct))
        if "error" in res:
            st.error(res["error"])
        else:
            colw = st.columns(3)
            for col, key, title in zip(
                    colw, ("hrp", "max_sharpe", "min_vol"),
                    ("🌳 HRP (recommended)", "🎯 Max Sharpe", "🛡️ Min Vol")):
                r = res.get(key, {})
                with col:
                    st.markdown(f"**{title}**")
                    if "error" in r:
                        st.warning(r["error"])
                    else:
                        st.caption(f"exp. ret {r['ret']}% · vol {r['vol']}% "
                                   f"· Sharpe {r['sharpe']}")
                        wdf = pd.DataFrame(
                            {"weight %": {k: round(v * 100, 1)
                                          for k, v in r["weights"].items()
                                          if v > 0.001}})
                        st.dataframe(wdf, use_container_width=True)
            if res.get("frontier"):
                figf = go.Figure()
                figf.add_trace(go.Scatter(
                    x=[p[0] for p in res["frontier"]],
                    y=[p[1] for p in res["frontier"]],
                    mode="lines", name="Efficient frontier",
                    line=dict(color="#10b981", width=2)))
                for v_, r_, t_ in res.get("assets", []):
                    figf.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers+text",
                                              text=[t_], textposition="top center",
                                              showlegend=False,
                                              marker=dict(size=9,
                                                          color="#8b98a5")))
                figf.update_layout(height=380, xaxis_title="Volatility %",
                                   yaxis_title="Expected return %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figf, use_container_width=True)
            if res.get("allocation") and "shares" in res.get("allocation", {}):
                st.markdown("**🧾 Discrete allocation for your account "
                            "(HRP weights):**")
                st.json(res["allocation"])
            with st.expander("❓ Which one should I use?"):
                st.markdown("""
- **HRP** — clusters assets by how they move together and splits risk down the tree. No return forecasts, no unstable matrix math → the weights barely change when the data wiggles. What quants actually deploy.
- **Max Sharpe** — the textbook optimum, but it *inhales* estimation error: tiny changes in expected returns swing the weights wildly. We cap any single name at 35% to tame it.
- **Min Vol** — pure defense. Also the sneaky one: low-vol portfolios historically beat their risk-adjusted expectations (the low-volatility anomaly from the Alpha engine).
""")

    st.markdown("---")
    st.subheader("🔗 Pairs lab — cointegration (Engle-Granger)")
    st.caption("Two stocks whose spread is mean-reverting = a market-neutral "
               "trade: long the cheap one, short the rich one, profit on "
               "convergence — regardless of market direction.")
    p1, p2 = st.columns(2)
    pa = p1.text_input("Ticker A", value="KO", key="pa").upper().strip()
    pb = p2.text_input("Ticker B", value="PEP", key="pb").upper().strip()
    if st.button("Test the pair", type="primary", key="pair_run"):
        with st.spinner("Testing cointegration…"):
            da = fetch_history(pa, period="2y")
            db = fetch_history(pb, period="2y")
            pr = pairs_analysis(da, db)
        if "error" in pr:
            st.error(pr["error"])
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cointegration p-value", pr["pvalue"],
                      delta="cointegrated ✓" if pr["cointegrated"]
                      else "borderline" if pr["borderline"] else "not a pair",
                      delta_color="off")
            k2.metric("Hedge ratio", pr["hedge_ratio"],
                      help=f"1 share {pa} ≈ {pr['hedge_ratio']} shares {pb}")
            k3.metric("Spread z-score", pr["z"])
            k4.metric("Half-life (days)", pr["half_life_days"] or "—")
            st.info(f"**Signal:** {pr['signal']}")

            z = pr["z_series"]
            figz = go.Figure()
            figz.add_trace(go.Scatter(x=z.index, y=z, name="spread z",
                                      line=dict(color="#22d3ee")))
            for lvl, col_ in ((2, "#ef4444"), (-2, "#10b981"), (0, "#8b98a5")):
                figz.add_hline(y=lvl, line_dash="dot", line_color=col_)
            figz.update_layout(height=320, yaxis_title="z-score",
                               margin=dict(l=10, r=10, t=30, b=10),
                               **PLOTLY_LAYOUT)
            st.plotly_chart(figz, use_container_width=True)
            with st.expander("❓ How to read this"):
                st.markdown(f"""
- **p-value ≤ 0.05** — the spread between {pa} and {pb} is statistically mean-reverting (Engle-Granger test). Above 0.10: whatever the chart looks like, it's not a pair.
- **z-score** — how stretched the spread is right now. The classic playbook: enter at |z| ≥ 2 (long the cheap leg, short the rich leg, sized by the hedge ratio), exit near z = 0.
- **Half-life** — how fast the spread typically closes half its gap. 5–30 days = tradeable; 100+ days = your capital will die of boredom.
- Caveat: shorting requires margin; if unavailable, the pair still works as a *relative-value tell* for which of the two names to prefer long.
""")


# ===========================================================================
# 8. EVENT RADAR — Polymarket odds as information (never traded)
# ===========================================================================
with tab_events:
    st.subheader("🌐 Event radar — real-money macro odds")
    st.caption("Live Polymarket probabilities on the events that move US "
               "equities: Fed, recession, CPI, shutdowns, tariffs, elections. "
               "**Information source only — we read these markets, we never "
               "trade them.**")

    if st.button("Scan macro markets", type="primary", key="ev_run"):
        with st.spinner("Reading Polymarket odds…"):
            ev = fetch_macro_markets()
        if ev.empty:
            st.warning("Couldn't reach the Polymarket API right now (or no "
                       "macro markets matched). Try again in a minute.")
        else:
            g = equity_risk_gauge(ev)
            if g:
                _remember("events", {"label": g["label"],
                                     "score": g["score"]})
                c1, c2 = st.columns([1, 2.5])
                c1.metric("Equity event gauge", g["label"],
                          delta=f"score {g['score']:+.2f}",
                          delta_color="off")
                with c2:
                    st.markdown("**Top drivers (real-money odds):**")
                    for q, p, d in g["drivers"]:
                        arrow = "🟢" if d > 0 else "🔴"
                        st.markdown(f"<div class='reason-{'pro' if d>0 else 'con'}'>"
                                    f"{arrow} {q} — **{p:.0f}%**</div>",
                                    unsafe_allow_html=True)

            st.markdown("#### All macro/finance markets (by volume)")
            show = ev.copy()
            show["yes %"] = show["yes %"].astype(float)
            st.dataframe(
                show.style.background_gradient(subset=["yes %"],
                                               cmap="RdYlGn_r",
                                               vmin=0, vmax=100),
                use_container_width=True, height=480, hide_index=True)

            with st.expander("❓ How a stock trader uses prediction markets"):
                st.markdown("""
Prediction-market odds are **real-money consensus** — people betting actual dollars, updated in real time. For an equities desk they answer one question: *what event risk is already priced?*

- **Fed cut at 80%** — a cut that happens is a non-event (priced); a *hold* would be the shock. Trade the surprise, not the event.
- **Recession odds climbing week over week** — tighten stops, favor the defensive side of the screener, respect the regime gate.
- **Shutdown/tariff odds jumping** — expect vol regime shifts; the Trade Desk's EWMA/GARCH will confirm with a lag, this leads.
- The 🧲/⛽ GEX regime + this gauge together tell you *both* how the market is positioned and *what* it's positioned for.

Idea credit where due: the repo you sent reads **market skew as crowd positioning** before entering — that's exactly what this tab does, pointed at macro instead of 5-minute BTC. And per your rule: read-only. We inform the stock process; we don't touch the markets themselves.
""")

# ===========================================================================
# 6. RL LAB — TradeMaster-inspired
# ===========================================================================
with tab_rl:
    st.subheader("🤖 RL lab — a learning agent, evaluated honestly")
    st.caption("Inspired by TradeMaster (NTU, NeurIPS 2023): agent + market "
               "dynamics modeling + PRUDEX-style multi-axis evaluation. "
               "Trained on the first 70% of history, judged ONLY on the "
               "unseen last 30%.")
    c1, c2 = st.columns([2, 1])
    rl_tkr = c1.text_input("Ticker", value="AAPL", key="rl").upper().strip()
    rl_period = c2.selectbox("History", ["5y", "10y"], index=0, key="rlp")

    if st.button("Train & evaluate agent", type="primary", key="rlrun"):
        with st.spinner("Training agent on the first 70%, testing on the rest…"):
            df = fetch_history(rl_tkr, period=rl_period)
            if len(df) < 400:
                st.error("Need at least ~400 bars of history.")
                st.stop()
            res = train_agent(df)
            if "error" in res:
                st.error(res["error"])
                st.stop()
            md = market_dynamics(df)

        # --- current decision ------------------------------------------------
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Current market state", res["current_state"])
        act_color = "🟢" if res["current_action"] == "LONG" else "⚪"
        d2.metric("Agent says", f"{act_color} {res['current_action']}")
        d3.metric("Edge estimate (bps/day)", res["current_confidence"])
        d4.metric("OOS exposure", f"{res['oos_exposure_pct']}%")

        # --- OOS equity ------------------------------------------------------
        st.markdown(f"### Out-of-sample test (from {res['split_date']} — "
                    "data the agent never saw)")
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=res["oos_equity"].index,
                                  y=res["oos_equity"], name="Agent",
                                  line=dict(color="#10b981", width=2)))
        figr.add_trace(go.Scatter(x=res["oos_bh"].index, y=res["oos_bh"],
                                  name="Buy & Hold",
                                  line=dict(color="#8b98a5", width=1.5,
                                            dash="dot")))
        figr.update_layout(height=380, yaxis_title="Growth of $1",
                           margin=dict(l=10, r=10, t=30, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figr, use_container_width=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Agent (out-of-sample)**")
            st.json(res["oos_stats"])
        with s2:
            st.markdown("**Buy & Hold (same period)**")
            st.json(res["bh_stats"])

        # --- PRUDEX radar -----------------------------------------------------
        st.markdown("### 🧭 PRUDEX-style evaluation compass")
        ax_a = prudex_scores(res["oos_equity"],
                             exposure_pct=res["oos_exposure_pct"])
        ax_b = prudex_scores(res["oos_bh"], exposure_pct=100)
        cats = list(ax_a.keys())
        figc = go.Figure()
        figc.add_trace(go.Scatterpolar(r=[ax_a[k] for k in cats] + [ax_a[cats[0]]],
                                       theta=cats + [cats[0]], fill="toself",
                                       name="Agent",
                                       line=dict(color="#10b981")))
        figc.add_trace(go.Scatterpolar(r=[ax_b[k] for k in cats] + [ax_b[cats[0]]],
                                       theta=cats + [cats[0]], fill="toself",
                                       name="Buy & Hold",
                                       line=dict(color="#8b98a5")))
        figc.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 100])),
                           margin=dict(l=40, r=40, t=30, b=30),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figc, use_container_width=True)

        # --- Policy table -------------------------------------------------------
        st.markdown("### 🧠 What the agent learned (full policy — nothing hidden)")
        st.dataframe(res["policy"], use_container_width=True, hide_index=True)

        # --- Market dynamics strip ----------------------------------------------
        st.markdown("### 🌍 Market dynamics modeling (TradeMaster MDM concept)")
        recent = md.iloc[-504:]
        figm = go.Figure()
        for s_i, (style, color) in enumerate(zip(MDM_STYLES, MDM_COLORS)):
            mask = recent["style"] == s_i
            if mask.any():
                figm.add_trace(go.Bar(x=recent.index[mask],
                                      y=np.ones(int(mask.sum())),
                                      marker_color=color, name=style,
                                      marker_line_width=0))
        figm.update_layout(height=140, barmode="stack", bargap=0,
                           yaxis=dict(visible=False),
                           margin=dict(l=10, r=10, t=10, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figm, use_container_width=True)
        st.caption(f"Current market style: **{md['label'].iloc[-1]}**")

        with st.expander("❓ What is this & why it's honest"):
            st.markdown("""
**The agent** learns the expected next-day return for each of 12 market states (trend × B-Xtrender × RSI) from the first 70% of history, with statistical shrinkage — it only acts on states where evidence clears a hurdle. This is the *contextual-bandit* form of reinforcement learning: since our tiny orders don't move the market, estimating the conditional edge IS the optimal policy — and unlike deep RL, it can't hallucinate patterns the data can't support.

**Why the honesty obsession:** TradeMaster (NeurIPS 2023) and its PRUDEX-Compass benchmark exist because most published FinRL results don't survive out-of-sample testing. So this lab shows you ONLY out-of-sample performance, the full learned policy with sample counts and t-stats, and the compass comparison vs plain buy & hold. If the agent doesn't beat B&H on your ticker — that's the data talking, believe it.

**Market dynamics strip:** the last 2 years labeled into 5 styles. Agents (and humans) trained mostly on bull data will be over-optimistic in bears — check what diet your agent grew up on.
""")

# ===========================================================================
# 5. POSITION SIZING
# ===========================================================================
with tab_sizing:
    st.subheader("How many shares for your account?")
    c1, c2, c3 = st.columns(3)
    acct = c1.number_input("Account $", 100, 1_000_000, 5000, step=100,
                           key="ps_acct")
    risk_p = c2.slider("Max loss per trade %", 0.5, 3.0, 1.0, 0.25,
                       key="ps_risk")
    ps_tkr = c3.text_input("Ticker", value="MSFT", key="ps").upper().strip()

    if st.button("Calculate", type="primary", key="pscalc"):
        df = fetch_history(ps_tkr, period="6mo")
        if df.empty:
            st.error(f"No data for {ps_tkr}")
        else:
            price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            stop_dist = 2.5 * a
            risk_dollars = acct * risk_p / 100
            shares = int(min(risk_dollars / stop_dist, acct / price))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${price:,.2f}")
            c2.metric("Stop distance (2.5×ATR)", f"${stop_dist:,.2f}")
            c3.metric("Shares to buy", shares)
            c4.metric("Position value", f"${shares * price:,.0f}")
            st.info(f"Stop-loss ≈ **${price - stop_dist:,.2f}**. If hit, you "
                    f"lose ≈ **${shares * stop_dist:,.0f}** ({risk_p}% of the "
                    f"account).")
            with st.expander("❓ Why size by risk instead of by dollars?"):
                st.markdown("""
Buying "$1,000 of each stock" gives a calm stock and a wild stock totally different risk. Sizing by **risk** (account % ÷ stop distance) equalises it. With a small account, ruin-avoidance *is* the strategy — a 50% drawdown needs +100% just to break even. For the mathematically optimal size per trade, see the **Kelly** number in the Trade desk's Monte Carlo section — and then use half of it, like the pros.
""")


st.markdown("---")
st.caption("QuantSignal v25 · data: Yahoo Finance (delayed) · educational "
           "tool, not financial advice · every model documented in its ❓ "
           "expander · built for one very persistent trader 🇮🇱")
