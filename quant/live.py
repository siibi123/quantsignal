"""Live engine — market clock, near-real-time quotes, live-bar patching.

The trick that makes the whole site "live": every model runs on daily bars,
so we fetch the current quote (cached ~20s) and PATCH it into today's bar.
Every downstream number — composite score, verdict, stops, GEX distance,
track-record marks — then moves with the market automatically.
"""
from __future__ import annotations

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


@st.cache_data(ttl=20, show_spinner=False)
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
