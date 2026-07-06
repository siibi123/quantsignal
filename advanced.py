"""Event radar — Polymarket macro odds as an INFORMATION source for stocks.

We do NOT trade prediction markets here. We read them: real-money odds on
Fed decisions, recessions, CPI, shutdowns and elections are among the best
live estimates of macro event risk — the stuff that moves US equities.

Data: Polymarket Gamma API (public, read-only, no key).
"""
from __future__ import annotations

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
