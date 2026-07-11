"""London Strategic Edge adapter — free-data integration, engineered honestly.

LSE (londonstrategicedge.com) offers a free-key HTTP API: candles at 14
resolutions, options chains WITH greeks, options flow, macro series, bond
yields. If it works as advertised it removes our Yahoo rate-limit ceiling
for $0.

Because their docs are JS-rendered (unreadable from the build sandbox),
this adapter does NOT hardcode guessed endpoints. Instead:
  * base URL + key are user-configurable,
  * probe() tests a set of common REST patterns and reports what responds,
  * fetch_candles() normalizes any of several likely response shapes to our
    standard OHLCV frame,
  * everything fails gracefully back to Yahoo.
Once one real example request from their docs is pasted in, the working
pattern is locked via session config — no code change needed.
"""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

DEFAULT_BASES = [
    "https://api.londonstrategicedge.com",
    "https://londonstrategicedge.com/api",
    "https://londonstrategicedge.com/api/v1",
]

CANDLE_PATTERNS = [
    "{base}/candles?symbol={sym}&resolution={res}&api_key={key}",
    "{base}/candles/{sym}?resolution={res}&api_key={key}",
    "{base}/v1/candles?symbol={sym}&resolution={res}&api_key={key}",
    "{base}/history?symbol={sym}&resolution={res}&api_key={key}",
    "{base}/candles?ticker={sym}&interval={res}&apikey={key}",
]

HEADER_STYLES = [
    {},                                   # key in query string
    {"Authorization": "Bearer {key}"},
    {"X-API-Key": "{key}"},
]


def _try(url: str, headers: dict, key: str, timeout: int = 8):
    h = {k: v.format(key=key) for k, v in headers.items()}
    try:
        r = requests.get(url, headers=h, timeout=timeout)
        return r.status_code, r
    except Exception as exc:
        return None, str(exc)


def probe(key: str, symbol: str = "AAPL", resolution: str = "1d",
          extra_base: str | None = None) -> list[dict]:
    """Test candidate endpoint shapes; return what each returned."""
    bases = ([extra_base] if extra_base else []) + DEFAULT_BASES
    results = []
    for base in bases:
        for pat in CANDLE_PATTERNS:
            for hs in HEADER_STYLES:
                url = pat.format(base=base.rstrip("/"), sym=symbol,
                                 res=resolution, key=key if not hs else "")
                code, resp = _try(url, hs, key)
                ok = code == 200
                note = ""
                if ok:
                    try:
                        j = resp.json()
                        note = f"JSON keys: {list(j)[:6]}" if isinstance(j, dict) \
                            else f"list[{len(j)}]"
                    except Exception:
                        note = resp.headers.get("content-type", "")[:40]
                results.append({"status": code, "ok": ok,
                                "auth": list(hs.keys())[0] if hs else "query",
                                "url": url.replace(key, "***") if key else url,
                                "note": note})
                if ok:
                    st.session_state["lse_working"] = {
                        "url_pattern": pat, "base": base,
                        "headers": hs}
                    return results          # first hit wins
    return results


def _normalize(j) -> pd.DataFrame:
    """Coerce likely candle shapes into our OHLCV frame."""
    rows = None
    if isinstance(j, dict):
        for k in ("candles", "data", "results", "bars", "ohlcv"):
            if k in j and isinstance(j[k], list):
                rows = j[k]
                break
        if rows is None and all(k in j for k in ("t", "o", "h", "l", "c")):
            df = pd.DataFrame({"Open": j["o"], "High": j["h"],
                               "Low": j["l"], "Close": j["c"],
                               "Volume": j.get("v", [0] * len(j["t"]))})
            ts = pd.Series(j["t"])
            idx = pd.to_datetime(ts, unit="s", errors="coerce") \
                if pd.api.types.is_numeric_dtype(ts) and ts.max() > 1e9 \
                else pd.to_datetime(ts, errors="coerce")
            df.index = pd.DatetimeIndex(idx).tz_localize(None)
            return df.astype(float).dropna().sort_index()
    elif isinstance(j, list):
        rows = j
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    colmap = {}
    for want, alts in {"Open": ("open", "o"), "High": ("high", "h"),
                       "Low": ("low", "l"), "Close": ("close", "c"),
                       "Volume": ("volume", "v", "vol")}.items():
        for a in alts:
            if a in df.columns:
                colmap[a] = want
                break
    tcol = next((c for c in ("ts", "t", "time", "timestamp", "date",
                             "datetime") if c in df.columns), None)
    if not tcol or len(colmap) < 4:
        return pd.DataFrame()
    df = df.rename(columns=colmap)
    ts = df[tcol]
    idx = pd.to_datetime(ts, unit="s", errors="coerce") \
        if pd.api.types.is_numeric_dtype(ts) and ts.max() > 1e9 \
        else pd.to_datetime(ts, errors="coerce")
    df.index = pd.DatetimeIndex(idx).tz_localize(None)
    if "Volume" not in df:
        df["Volume"] = 0.0
    out = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return out.dropna().sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_candles_lse(symbol: str, resolution: str = "1d") -> pd.DataFrame:
    """Fetch via the pattern probe() discovered. Empty df if unconfigured."""
    cfg = st.session_state.get("lse_working")
    key = st.session_state.get("lse_key", "")
    if not cfg or not key:
        return pd.DataFrame()
    url = cfg["url_pattern"].format(base=cfg["base"].rstrip("/"), sym=symbol,
                                    res=resolution,
                                    key=key if not cfg["headers"] else "")
    code, resp = _try(url, cfg["headers"], key, timeout=12)
    if code != 200:
        return pd.DataFrame()
    try:
        return _normalize(resp.json())
    except Exception:
        return pd.DataFrame()
