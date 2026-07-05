"""Data layer — OHLCV from Yahoo Finance with caching.

Price convention (important for correctness):
  auto_adjust=False  -> Close is the REAL traded price, matching your broker
                        and TradingView. This is what we chart, set stops/
                        targets on, and quote. 'Adj Close' is kept separately
                        for return calculations that need dividend adjustment.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

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
    df = yf.Ticker(ticker).history(period=period, interval=interval,
                                   auto_adjust=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)          # Open/High/Low/Close/Adj Close/Volume
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        df["AdjClose"] = df["Adj Close"]
        cols = cols + ["AdjClose"]
    return df[cols].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_many(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_history(t, period=period)
            if len(df) >= 60:
                out[t] = df
        except Exception:
            continue
    return out
