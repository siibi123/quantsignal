"""Data layer — OHLCV from Yahoo Finance with caching.

Price convention (important for correctness):
  auto_adjust=False  -> Close is the REAL traded price, matching your broker
                        and TradingView. This is what we chart, set stops/
                        targets on, and quote. 'Adj Close' is kept separately
                        for return calculations that need dividend adjustment.
"""
from __future__ import annotations

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
