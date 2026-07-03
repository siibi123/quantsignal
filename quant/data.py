"""Data layer — fetches OHLCV data from Yahoo Finance with caching."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

# A liquid, tradeable-through-any-US-broker universe (S&P 500 leaders + popular ETFs).
DEFAULT_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD", "CRM",
    # Financials / industrials / energy
    "JPM", "BAC", "GS", "V", "MA", "CAT", "GE", "XOM", "CVX", "COP",
    # Healthcare / consumer
    "UNH", "LLY", "JNJ", "PG", "KO", "COST", "WMT", "MCD", "NKE", "DIS",
    # Semis / software
    "TSM", "INTC", "MU", "QCOM", "ORCL", "ADBE", "NOW", "PLTR", "SMCI", "PANW",
    # ETFs (good for a $5K account — instant diversification)
    "SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "SMH", "GLD", "TLT",
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history for one ticker. Cached for 1 hour."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return df
    df = df.rename(columns=str.title)  # Open/High/Low/Close/Volume
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_many(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Download several tickers. Returns {ticker: df}. Skips failures silently."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = fetch_history(t, period=period)
            if len(df) >= 60:
                out[t] = df
        except Exception:
            continue
    return out
