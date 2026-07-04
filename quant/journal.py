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
from __future__ import annotations

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
