"""Seasonality stats (Detrick-style) + fundamental quality snapshot."""
from __future__ import annotations

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
