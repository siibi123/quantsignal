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
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .advanced import regime_quadrant
from .bxtrender import bxtrender
from .scanner import scan_setups
from .signals import atr, composite, rsi, sma

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
