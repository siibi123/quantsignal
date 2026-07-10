"""RUNNER — the trade lifecycle machine.

Replays a ticker bar-by-bar and manages positions the way a desk does,
logging EVERY event with its reason and the model state at that moment:

  ENTRY       — trend (composite BUY + B-X confirm + regime gate) or
                dip (RSI2 panic inside the fib pocket of an uptrend);
                auto-picked per ticker by Hurst.
  SCALE 1/3   — at +1R: take a third off, stop jumps to breakeven.
  SCALE 2/3   — at +2R: take another third, stop jumps to entry +1R.
  TRAIL       — the runner rides a 2.5xATR chandelier under the rest.
  STOP / BE   — mechanical, first touch.
  TIME EXIT   — unprofitable and stale -> out.
  SIGNAL EXIT — the composite flips to SELL.

Ends with the CURRENT live state: in/out, remaining size, active stop,
and tomorrow's order. This is the machine actually running the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _hurst_quick
from .bxtrender import bxtrender
from .signals import atr, composite, rsi, sma


def run_machine(df: pd.DataFrame, account: float = 5000.0,
                risk_pct: float = 1.0, atr_mult: float = 2.5,
                mode: str = "auto", commission: float = 0.001) -> dict:
    if len(df) < 300:
        return {"error": "Need at least ~300 bars."}

    if mode == "auto":
        h = _hurst_quick(df["Close"])
        mode = "trend" if h >= 0.5 else "dip"

    comp = composite(df)
    bx = bxtrender(df)
    a = atr(df)
    s200 = sma(df["Close"], 200)
    r2 = rsi(df["Close"], 2)

    roll_hi = df["High"].rolling(126, min_periods=40).max()
    roll_lo = df["Low"].rolling(126, min_periods=40).min()
    rng_ = roll_hi - roll_lo
    fib_lo = (roll_hi - 0.786 * rng_).values
    fib_hi = (roll_hi - 0.382 * rng_).values

    o_, h_, l_, c_ = (df[k].values for k in ("Open", "High", "Low", "Close"))
    sig = comp["signal"].values
    score = comp["score"].values
    bxl = bx["long_osc"].values
    bxr = bx["t3_rising"].values
    bxbt = bx["buy_turn"].values
    av = a.values
    s2v = s200.values
    r2v = r2.values
    idx = df.index

    cash = account
    sh = 0.0                    # current shares
    sh0 = 0.0                   # original size
    entry_px = stop = r_unit = 0.0
    entry_i = 0
    scaled1 = scaled2 = False
    events: list[dict] = []
    equity = []

    closures: list[dict] = []
    entry_ctx: dict = {}

    def log(i, ev, px, qty, note):
        events.append({
            "date": idx[i].date(), "event": ev, "price": round(float(px), 2),
            "shares": int(qty), "position_after": int(sh),
            "equity": round(cash + sh * c_[i], 0),
            "score": round(float(score[i - 1]), 2),
            "bx": f"{bxl[i-1]:+.0f}{'↑' if bxr[i-1] else '↓'}",
            "rsi2": round(float(r2.values[i - 1])),
            "note": note,
        })

    def sell(i, qty, px, ev, note):
        nonlocal cash, sh
        qty = min(qty, sh)
        if qty <= 0:
            return
        cash += qty * px * (1 - commission)
        sh -= qty
        if entry_ctx:
            entry_ctx["realized"] += qty * (px - entry_px)
        log(i, ev, px, qty, note)
        if sh == 0 and entry_ctx:
            invested = entry_ctx["invested"]
            pnl = entry_ctx["realized"]
            closures.append({
                "opened": idx[entry_i].date(), "closed": idx[i].date(),
                "bars": int(i - entry_i),
                "entry $": round(entry_px, 2),
                "final exit $": round(px, 2),
                "price chg %": round((px / entry_px - 1) * 100, 2),
                "capital in $": round(invested, 0),
                "P&L $": round(pnl, 0),
                "return on capital %": round(pnl / invested * 100, 2)
                if invested else 0,
                "R": round(pnl / (sh0 * r_unit), 2) if r_unit > 0 else 0,
                "exit reason": ev,
                "entry score": round(entry_ctx["score"], 2),
                "entry BX": round(entry_ctx["bx_long"], 0),
                "entry RSI2": round(entry_ctx["rsi2"], 0),
            })

    for i in range(1, len(df)):
        o, hi, lo = o_[i], h_[i], l_[i]
        above200 = not np.isnan(s2v[i - 1]) and c_[i - 1] > s2v[i - 1]

        # ---------- manage open position ----------
        if sh > 0:
            bars_in = i - entry_i
            # scale-outs (checked before stop so profit-taking wins the day)
            if not scaled1 and hi >= entry_px + 1.0 * r_unit:
                px = max(o, entry_px + 1.0 * r_unit)
                sell(i, round(sh0 / 3), px, "SCALE 1/3 (+1R)",
                     "stop → breakeven")
                stop = max(stop, entry_px)
                scaled1 = True
            if sh > 0 and not scaled2 and hi >= entry_px + 2.0 * r_unit:
                px = max(o, entry_px + 2.0 * r_unit)
                sell(i, round(sh0 / 3), px, "SCALE 2/3 (+2R)",
                     "stop → entry +1R")
                stop = max(stop, entry_px + 1.0 * r_unit)
                scaled2 = True
            # trail on the remainder (trend mode)
            if sh > 0 and mode == "trend":
                new_trail = hi - atr_mult * av[i - 1]
                if new_trail > stop:
                    stop = new_trail
            # stop check
            if sh > 0 and lo <= stop:
                px = min(o, stop) if o > stop else o
                tag = ("BREAKEVEN STOP" if abs(stop - entry_px) < 1e-9
                       else "TRAIL STOP" if scaled1 else "STOP LOSS")
                sell(i, sh, max(px, 0.01), tag,
                     f"after {bars_in} bars")
                scaled1 = scaled2 = False
            # signal / rsi / time exits
            elif sh > 0 and mode == "trend" and sig[i - 1] == "SELL":
                sell(i, sh, o, "SIGNAL EXIT", "composite flipped to SELL")
                scaled1 = scaled2 = False
            elif sh > 0 and mode == "dip" and r2v[i - 1] > 65:
                sell(i, sh, o, "TARGET (RSI snap-back)",
                     f"RSI2={r2v[i-1]:.0f}")
                scaled1 = scaled2 = False
            elif sh > 0 and bars_in >= (20 if mode == "trend" else 10) \
                    and c_[i - 1] < entry_px:
                sell(i, sh, o, "TIME EXIT", "stale & unprofitable")
                scaled1 = scaled2 = False

        # ---------- entries ----------
        if sh == 0 and av[i - 1] > 0:
            enter = False
            why = ""
            if mode == "trend":
                if above200 and sig[i - 1] == "BUY" and bxl[i - 1] > 0 \
                        and bxr[i - 1]:
                    enter = True
                    why = (f"composite {score[i-1]:+.2f} BUY · B-X "
                           f"{bxl[i-1]:+.0f}↑ · above 200-SMA")
            else:
                pocket = (not np.isnan(fib_lo[i - 1]) and
                          fib_lo[i - 1] <= c_[i - 1] <= fib_hi[i - 1])
                if above200 and r2v[i - 1] < 10 and (pocket or bxbt[i - 1]):
                    enter = True
                    why = (f"RSI2={r2v[i-1]:.0f} panic · "
                           f"{'fib pocket' if pocket else 'B-X buy turn'} "
                           f"· uptrend intact")
            if enter:
                stop_d = atr_mult * av[i - 1]
                qty = int(min((cash * risk_pct / 100) / stop_d,
                              cash / (o * (1 + commission))))
                if qty * o >= 100:
                    cash -= qty * o * (1 + commission)
                    sh = sh0 = float(qty)
                    entry_px, entry_i = o, i
                    r_unit = stop_d
                    stop = o - stop_d
                    scaled1 = scaled2 = False
                    entry_ctx = {"score": float(score[i - 1]),
                                 "bx_long": float(bxl[i - 1]),
                                 "rsi2": float(r2.values[i - 1]),
                                 "invested": qty * o,
                                 "realized": 0.0}
                    log(i, f"ENTRY ({mode.upper()})", o, qty,
                        why + f" · stop ${stop:,.2f} · invested "
                        f"${qty * o:,.0f} ({qty * o / account * 100:.0f}% "
                        f"of account)")

        equity.append(cash + sh * c_[i])

    eq = pd.Series(equity, index=idx[1:])
    ev_df = pd.DataFrame(events)

    # ---- closed-trade stats from the event log ----
    realized = ev_df[ev_df["event"].str.contains(
        "STOP|EXIT|TARGET|SCALE")] if len(ev_df) else pd.DataFrame()

    # ---- live state ----
    if sh > 0:
        state = {
            "in_position": True,
            "shares": int(sh),
            "entry": round(entry_px, 2),
            "stop_now": round(stop, 2),
            "bars_held": int(len(df) - 1 - entry_i),
            "unrealized": round((c_[-1] - entry_px) * sh, 0),
            "scaled": f"{'⅓ taken' if scaled1 else 'full size'}"
                      f"{' + ⅔ taken' if scaled2 else ''}",
            "tomorrow": (f"HOLD {int(sh)} shares · stop at "
                         f"${stop:,.2f} · next scale at "
                         f"${entry_px + (2 if scaled1 else 1) * r_unit:,.2f}"),
        }
    else:
        nxt = "watch for "
        nxt += ("composite BUY + B-X rising above the 200-SMA"
                if mode == "trend" else
                "RSI2 < 10 inside the fib pocket of an uptrend")
        state = {"in_position": False, "tomorrow": f"FLAT · {nxt}"}

    r_ = eq.pct_change().dropna()
    n_years = max(len(eq) / 252, 1e-9)
    stats = {
        "Mode": mode.upper(),
        "Final equity $": round(float(eq.iloc[-1]), 0),
        "CAGR %": round(((eq.iloc[-1] / account) ** (1 / n_years) - 1) * 100, 1),
        "Sharpe": round(float(r_.mean() / r_.std() * np.sqrt(252)), 2)
        if r_.std() > 0 else 0.0,
        "Max DD %": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
        "Events logged": int(len(ev_df)),
    }
    closed_df = pd.DataFrame(closures)

    # ---- 🎓 self-learning: what did the machine learn on THIS ticker? -----
    lessons: list[str] = []
    if len(closed_df) >= 5:
        by_reason = closed_df.groupby("exit reason")["P&L $"].agg(
            ["count", "sum", "mean"]).sort_values("sum", ascending=False)
        best_r, worst_r = by_reason.index[0], by_reason.index[-1]
        lessons.append(f"Exit analysis: '{best_r}' exits made the most "
                       f"(${by_reason.loc[best_r,'sum']:,.0f} over "
                       f"{int(by_reason.loc[best_r,'count'])} trades); "
                       f"'{worst_r}' cost the most "
                       f"(${by_reason.loc[worst_r,'sum']:,.0f}).")
        hiBX = closed_df[closed_df["entry BX"] >= 25]
        loBX = closed_df[closed_df["entry BX"] < 25]
        if len(hiBX) >= 3 and len(loBX) >= 3:
            lessons.append(f"Entry-quality: trades opened with strong B-X "
                           f"(≥+25) won {float((hiBX['P&L $']>0).mean())*100:.0f}%"
                           f" vs {float((loBX['P&L $']>0).mean())*100:.0f}% "
                           f"with weak B-X → on {'this ticker' } demand the "
                           f"stronger confirmation.")
        wr = float((closed_df["P&L $"] > 0).mean() * 100)
        avg_roc = float(closed_df["return on capital %"].mean())
        lessons.append(f"Overall on this ticker: {len(closed_df)} closed "
                       f"round-trips · {wr:.0f}% won · avg "
                       f"{avg_roc:+.2f}% return on deployed capital per trade.")
        held_w = closed_df[closed_df["P&L $"] > 0]["bars"].mean()
        held_l = closed_df[closed_df["P&L $"] <= 0]["bars"].mean()
        if held_w and held_l:
            lessons.append(f"Time asymmetry: winners held "
                           f"{held_w:.0f} bars vs losers {held_l:.0f} — "
                           + ("healthy (cutting losers fast)."
                              if held_w > held_l else
                              "⚠️ losers held LONGER than winners — the "
                              "classic account-killer; trust the time stop."))
    elif len(closed_df):
        lessons.append(f"Only {len(closed_df)} closed trades — the machine "
                       f"needs ~5+ round-trips before its lessons mean much.")

    return {"events": ev_df, "equity": eq, "state": state, "stats": stats,
            "mode": mode, "closures": closed_df, "lessons": lessons}
