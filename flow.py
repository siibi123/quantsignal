"""Timeframes — one selector, correct math on every horizon.

Indicator windows are in BARS (standard practice): a 50-bar average on
weekly bars is a ~1-year trend measure, on hourly bars a ~2-week one.
Annualization factors keep Sharpe/CAGR honest per timeframe.
"""
from __future__ import annotations

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
