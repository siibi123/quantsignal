"""QuantSignal — quant trade desk, screener & backtester for US stocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.backtest import BTConfig, run_backtest, walk_forward
from quant.bxtrender import (bxtrender, detect_divergence, event_study,
                             weekly_alignment)
from quant.data import DEFAULT_UNIVERSE, fetch_history, fetch_many
from quant.levels import fib_levels, hurst
from quant.montecarlo import cone, simulate, trade_odds
from quant.options import (atm_term_structure, bs_greeks, build_surface,
                           fetch_chains, skew_25)
from quant.signals import BUY_TH, SELL_TH, atr, composite, latest_snapshot
from quant.verdict import analyze

st.set_page_config(page_title="QuantSignal", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Global styling — terminal-grade dark UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
  font-size: 3rem; font-weight: 900; letter-spacing: -1.5px; margin: 0;
  background: linear-gradient(90deg, #10b981 0%, #34d399 40%, #6ee7b7 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #8b98a5; font-size: .95rem; margin-top: 2px; }

[data-testid="stMetric"] {
  background: linear-gradient(180deg, #141c26 0%, #10161e 100%);
  border: 1px solid #1f2a36; border-radius: 14px; padding: 14px 16px;
  transition: border-color .2s;
}
[data-testid="stMetric"]:hover { border-color: #10b98155; }
[data-testid="stMetricValue"] { font-weight: 800; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { color: #8b98a5; }

div[data-testid="stExpander"] {
  border: 1px dashed #2a3644; border-radius: 12px; background: #0f151d;
}

.verdict {
  border-radius: 18px; padding: 28px 32px; margin: 8px 0 16px 0;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 8px 32px rgba(0,0,0,.35);
}
.verdict h2 { margin: 0; font-size: 2.4rem; font-weight: 900; letter-spacing: -1px; }
.verdict .sub { opacity: .85; font-size: .95rem; margin-top: 6px; }
.v-long  { background: linear-gradient(135deg,#053b2d,#065f46 60%,#047857); border:1px solid #10b981; }
.v-short { background: linear-gradient(135deg,#5f1414,#991b1b 60%,#b91c1c); border:1px solid #ef4444; }
.v-none  { background: linear-gradient(135deg,#1a2332,#2b3648 60%,#374151); border:1px solid #6b7280; }

.reason-pro, .reason-con {
  border-radius: 10px; padding: 9px 14px; margin: 6px 0; font-size: .92rem;
}
.reason-pro { background:#0b2e22; border-left: 3px solid #10b981; }
.reason-con { background:#2e0f0f; border-left: 3px solid #ef4444; }
.conv-wrap { background:#1f2a36; border-radius: 8px; height: 16px; width: 100%; }
.conv-bar  { height: 16px; border-radius: 8px;
  background-image: linear-gradient(90deg, transparent 0, rgba(255,255,255,.18) 50%, transparent 100%);
  background-size: 40px 100%; }
hr { border-color: #1f2a36; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hero-title">QuantSignal</p>'
            '<p class="hero-sub">Trend · Momentum · Mean-reversion · '
            'Volatility regimes · Monte Carlo · Fibonacci · Black-Scholes — '
            'fused into one decision. Educational tool, not financial advice.'
            '</p>', unsafe_allow_html=True)
st.write("")

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Inter", color="#e6edf3"))

tab_desk, tab_screener, tab_backtest, tab_options, tab_sizing = st.tabs(
    ["🎯 Trade desk", "🔍 Screener", "🧪 Backtest",
     "🌋 Options / IV surface", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#8a8a8a"}
VERDICT_CLASS = {"LONG": "v-long", "SHORT": "v-short", "NO TRADE": "v-none"}
VERDICT_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}
FIB_COLORS = {"0": "#6ee7b7", "0.236": "#34d399", "0.382": "#fbbf24",
              "0.5": "#f59e0b", "0.618": "#f97316", "0.786": "#ef4444",
              "1": "#dc2626"}

# ===========================================================================
# 1. TRADE DESK
# ===========================================================================
with tab_desk:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4])
    tkr = c1.text_input("Ticker", value="NVDA", key="desk").upper().strip()
    account = c2.number_input("Account $", 500, 1_000_000, 5000, step=500)
    risk_pct = c3.slider("Risk/trade %", 0.5, 3.0, 1.0, 0.25)
    use_opts = c4.toggle("Include options skew (slower)", value=False)

    if st.button("Run desk analysis", type="primary", key="deskrun"):
        with st.spinner("Crunching signals, edge history, Monte Carlo & levels…"):
            df = fetch_history(tkr, period="2y")
            if df.empty or len(df) < 260:
                st.error(f"Not enough data for {tkr} (need ~1y of history).")
                st.stop()
            skew = None
            if use_opts:
                try:
                    _, chain_d = fetch_chains(tkr, max_expiries=4)
                    skew = skew_25(chain_d)
                except Exception:
                    skew = None
            v = analyze(df, account=account, risk_pct=risk_pct, skew=skew)
            h = hurst(df)
            fib = fib_levels(df)
            direction = 1 if v["verdict"] != "SHORT" else -1
            paths = simulate(df, days=30, n_paths=2000)
            odds = trade_odds(paths, v["entry"], v["stop"], v["target"],
                              direction)
            bands = cone(paths)

        # ---- Verdict banner ------------------------------------------------
        cls = VERDICT_CLASS[v["verdict"]]
        conv_color = ("#10b981" if v["verdict"] == "LONG"
                      else "#ef4444" if v["verdict"] == "SHORT" else "#6b7280")
        st.markdown(f"""
        <div class="verdict {cls}">
          <div>
            <h2>{VERDICT_EMOJI[v['verdict']]} {v['verdict']} — {tkr}</h2>
            <div class="sub">Composite {v['score']:+.2f} · {v['agree']}/7 models aligned ·
              signal Sharpe on {tkr}: {v['sharpe']} ({v['n_trades']} trades) ·
              Hurst {h}</div>
          </div>
          <div style="min-width:230px">
            <div style="font-size:.85rem;opacity:.8;margin-bottom:4px">
              Conviction {v['conviction']}/100</div>
            <div class="conv-wrap"><div class="conv-bar"
              style="width:{v['conviction']}%;background-color:{conv_color}"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("❓ What is the verdict & conviction?"):
            st.markdown(f"""
- **The verdict** fuses seven models (trend, momentum, B-Xtrender, MACD, RSI, mean-reversion, volume) into one composite score, then demands: models agreeing, a calm volatility regime, a *proven* historical edge on this exact ticker, and risk/reward ≥ 1.3. Fail any → **NO TRADE**. Standing aside is a position.
- **Conviction (0–100)**: 40% signal strength + 25% model agreement + 15% volatility regime + 20% historical edge (± options skew). Above 55 = tradeable.
- **Hurst exponent ({h})**: measures the ticker's *character*. Above 0.5 → moves tend to **continue** (trust trend/momentum models). Below 0.5 → moves tend to **reverse** (trust mean-reversion). Near 0.5 → random walk, no memory.
""")

        # ---- Levels ---------------------------------------------------------
        if v["verdict"] != "NO TRADE":
            m = st.columns(6)
            m[0].metric("Entry", f"${v['entry']:,.2f}")
            m[1].metric("Stop", f"${v['stop']:,.2f}")
            m[2].metric("Target", f"${v['target']:,.2f}")
            m[3].metric("Risk : Reward", f"1 : {v['rr']}")
            m[4].metric("Shares", v["shares"])
            m[5].metric("$ at risk", f"${v['risk_dollars']:,.0f}")
            if v["verdict"] == "SHORT":
                st.warning("Shorting needs margin approval at your broker; if "
                           "unavailable, treat SHORT as **avoid / exit longs**.")
            with st.expander("❓ How are these levels computed?"):
                st.markdown("""
- **Stop** = entry ± 2.5×ATR (Average True Range — the stock's typical daily wiggle). Wide enough to survive noise, tight enough to cap damage.
- **Target** = the 63-day swing high/low, or a 2R measured move on breakouts to new highs (no overhead resistance to aim at).
- **Shares** = sized so a stop-out loses exactly your chosen % of the account. This is the single most important number on this page.
""")
        else:
            st.info("**Standing aside is a position.** The desk found no edge "
                    "worth risking money on right now — that's the system "
                    "working, not failing.")

        # ---- Reasons ---------------------------------------------------------
        colp, colc = st.columns(2)
        with colp:
            st.markdown("**✅ For**")
            for r in v["reasons_pro"]:
                st.markdown(f"<div class='reason-pro'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_pro"]:
                st.markdown("<div class='reason-con'>Nothing working in "
                            "favour right now</div>", unsafe_allow_html=True)
        with colc:
            st.markdown("**⚠️ Against**")
            for r in v["reasons_con"]:
                st.markdown(f"<div class='reason-con'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_con"]:
                st.markdown("<div class='reason-pro'>No red flags "
                            "detected</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---- 🎲 Monte Carlo -------------------------------------------------
        st.markdown("### 🎲 Monte Carlo — 2,000 simulated futures (30 days)")
        mc = st.columns(6)
        mc[0].metric("P(hit target first)", f"{odds['p_target_first']}%")
        mc[1].metric("P(hit stop first)", f"{odds['p_stop_first']}%")
        mc[2].metric("P(neither in 30d)", f"{odds['p_neither']}%")
        mc[3].metric("P(profitable at day 30)", f"{odds['p_profit_end']}%")
        mc[4].metric("95% VaR / share", f"${odds['var95_share']:,.2f}")
        mc[5].metric("95% CVaR / share", f"${odds['cvar95_share']:,.2f}")

        x = list(range(paths.shape[1]))
        figmc = go.Figure()
        figmc.add_trace(go.Scatter(x=x + x[::-1],
                                   y=list(bands[95]) + list(bands[5])[::-1],
                                   fill="toself", fillcolor="rgba(16,185,129,.10)",
                                   line=dict(width=0), name="5–95%"))
        figmc.add_trace(go.Scatter(x=x + x[::-1],
                                   y=list(bands[75]) + list(bands[25])[::-1],
                                   fill="toself", fillcolor="rgba(16,185,129,.22)",
                                   line=dict(width=0), name="25–75%"))
        figmc.add_trace(go.Scatter(x=x, y=bands[50], name="Median path",
                                   line=dict(color="#6ee7b7", width=2)))
        rng = np.random.default_rng(1)
        for i in rng.choice(paths.shape[0], 12, replace=False):
            figmc.add_trace(go.Scatter(x=x, y=paths[i], showlegend=False,
                                       line=dict(width=.7,
                                                 color="rgba(230,237,243,.25)")))
        for lvl, name, color in ((v["stop"], "Stop", "#ef4444"),
                                 (v["target"], "Target", "#10b981")):
            figmc.add_hline(y=lvl, line_dash="dot", line_color=color,
                            annotation_text=f"{name} ${lvl:,.2f}",
                            annotation_font_color=color)
        figmc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                            xaxis_title="Trading days ahead",
                            yaxis_title="Price $", **PLOTLY_LAYOUT)
        st.plotly_chart(figmc, use_container_width=True)

        with st.expander("❓ What is Monte Carlo & how do I read this?"):
            st.markdown(f"""
This simulates **2,000 alternative futures** for {tkr} using Geometric Brownian Motion, calibrated to *its own* recent drift and volatility (recent days weighted heavier).

- **The cone**: dark band = the middle 50% of futures, light band = 90%. A price outside the cone in reality = the market changed character.
- **P(hit target first) vs P(hit stop first)**: which level do the simulations touch first? This is your *simulated win rate* for this exact trade — compare it with the risk/reward. A 40% win rate is great when RR is 2:1.
- **VaR 95%**: in the worst 5% of futures, you lose at least this much per share. **CVaR**: the *average* loss inside that worst 5% — the "how bad is bad" number professional desks size by.
""")

        st.markdown("---")

        # ---- 📐 Chart with Fibonacci + levels --------------------------------
        st.markdown("### 📐 Price structure — Fibonacci retracements & levels")
        comp = composite(df)
        figp = go.Figure()
        figp.add_trace(go.Candlestick(
            x=df.index[-252:], open=df["Open"][-252:], high=df["High"][-252:],
            low=df["Low"][-252:], close=df["Close"][-252:], name=tkr))
        for key, price in fib["levels"].items():
            figp.add_hline(y=price, line_dash="dot", line_width=1,
                           line_color=FIB_COLORS.get(key, "#8b98a5"),
                           annotation_text=f"Fib {key} — ${price:,.2f}",
                           annotation_font_size=10,
                           annotation_font_color=FIB_COLORS.get(key, "#8b98a5"))
        if v["verdict"] != "NO TRADE":
            for lvl, name, color in ((v["stop"], "STOP", "#ef4444"),
                                     (v["target"], "TARGET", "#10b981")):
                figp.add_hline(y=lvl, line_width=2, line_color=color,
                               annotation_text=name,
                               annotation_font_color=color)
        buys = [d for d in comp[comp["signal"] == "BUY"].index[-252:]
                if d in df.index[-252:]]
        sells = [d for d in comp[comp["signal"] == "SELL"].index[-252:]
                 if d in df.index[-252:]]
        figp.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Low"] * 0.985,
                                  mode="markers", name="BUY zone",
                                  marker=dict(symbol="triangle-up", size=7,
                                              color="#10b981")))
        figp.add_trace(go.Scatter(x=sells, y=df.loc[sells, "High"] * 1.015,
                                  mode="markers", name="SELL zone",
                                  marker=dict(symbol="triangle-down", size=7,
                                              color="#ef4444")))
        figp.update_layout(height=560, xaxis_rangeslider_visible=False,
                           margin=dict(l=10, r=10, t=30, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figp, use_container_width=True)

        swing_dir = "upswing" if fib["up_swing"] else "downswing"
        with st.expander("❓ What are Fibonacci retracements?"):
            st.markdown(f"""
The dominant swing detected in the last ~6 months: **{swing_dir}** from ${fib['swing_low']:,.2f} to ${fib['swing_high']:,.2f}.

Fibonacci retracements mark how much of a move typically gets "given back" before the trend resumes. The levels traders actually watch:
- **0.382 / 0.5** — shallow pullback: strong trend, buyers stepping in early
- **0.618** — the "golden pocket": the classic spot where healthy pullbacks end. Bounce here = trend intact
- **0.786** — last defence; below it the swing is usually considered broken

They work partly because *so many traders watch them* (self-fulfilling). Use them as **zones to expect a reaction**, not as magic lines — best combined with the verdict above.
""")

        st.markdown("---")

        # ---- ⚡ B-Xtrender (institutional) -----------------------------------
        st.markdown("### ⚡ B-Xtrender — institutional edition")
        bx = bxtrender(df)
        div = detect_divergence(df)
        wk = weekly_alignment(df)
        bx_last = bx.iloc[-1]

        bm = st.columns(5)
        bm[0].metric("Short oscillator", f"{bx_last['short_osc']:+.1f}",
                     delta="rising" if bx_last['t3_rising'] else "falling",
                     delta_color="normal" if bx_last['t3_rising'] else "inverse")
        bm[1].metric("Long oscillator (trend)", f"{bx_last['long_osc']:+.1f}")
        wk_txt = ("—" if wk["weekly_osc"] is None else
                  f"{wk['weekly_osc']:+.1f} " + ("↑" if wk["weekly_rising"] else "↓"))
        bm[2].metric("Weekly oscillator (MTF)", wk_txt)
        aligned = (wk["weekly_osc"] is not None and
                   np.sign(wk["weekly_osc"]) == np.sign(bx_last["long_osc"]))
        bm[3].metric("Timeframes aligned", "YES ✅" if aligned else "NO ⚠️")
        div_txt = ("🐻 Bearish" if div["bearish"] else
                   "🐂 Bullish" if div["bullish"] else "None")
        bm[4].metric("Divergence", div_txt)
        if div["detail"]:
            st.caption(f"Divergence detail: {div['detail']}")

        # Two-pane oscillator chart, TradingView style
        from plotly.subplots import make_subplots
        w = df.index[-252:]
        bxw = bx.loc[w]
        so, t3l = bxw["short_osc"], bxw["t3"]
        rising_now = so > so.shift(1)
        colors_s = np.where(so > 0, np.where(rising_now, "#22ff44", "#228B22"),
                            np.where(rising_now, "#ff5555", "#8B0000"))
        lo = bxw["long_osc"]
        colors_l = np.where(lo > 0, np.where(lo > lo.shift(1), "#22ff44", "#228B22"),
                            np.where(lo > lo.shift(1), "#ff5555", "#8B0000"))

        figbx = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.06,
                              subplot_titles=("Short-term oscillator + T3 signal",
                                              "Long-term oscillator (trend)"))
        figbx.add_trace(go.Bar(x=w, y=so, marker_color=colors_s,
                               name="Short osc"), row=1, col=1)
        figbx.add_trace(go.Scatter(x=w, y=t3l, name="T3 signal",
                                   line=dict(color="#e6edf3", width=2.5)),
                        row=1, col=1)
        buys_bx = w[bxw["buy_turn"].values]
        sells_bx = w[bxw["sell_turn"].values]
        figbx.add_trace(go.Scatter(x=buys_bx, y=t3l.loc[buys_bx],
                                   mode="markers", name="Buy turn",
                                   marker=dict(color="#22ff44", size=9,
                                               symbol="circle")), row=1, col=1)
        figbx.add_trace(go.Scatter(x=sells_bx, y=t3l.loc[sells_bx],
                                   mode="markers", name="Sell turn",
                                   marker=dict(color="#ff5555", size=9,
                                               symbol="circle")), row=1, col=1)
        figbx.add_trace(go.Bar(x=w, y=lo, marker_color=colors_l,
                               name="Long osc", showlegend=False), row=2, col=1)
        figbx.add_hline(y=0, line_color="#2a3644", row=1, col=1)
        figbx.add_hline(y=0, line_color="#2a3644", row=2, col=1)
        figbx.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                            bargap=0.15, **PLOTLY_LAYOUT)
        st.plotly_chart(figbx, use_container_width=True)

        st.markdown("**📊 Event study — what actually happened after each "
                    f"signal on {tkr}:**")
        st.dataframe(event_study(df), use_container_width=True, hide_index=True)

        with st.expander("❓ What is B-Xtrender & the upgrades here?"):
            st.markdown("""
**B-Xtrender** (Bharat Jhunjhunwala, IFTA Journal) is a double-smoothed momentum oscillator: an RSI applied to the *spread between two EMAs*, filtered by a Tillson T3 line. It reacts faster than MACD with less noise than raw RSI.

How to read it:
- **Short oscillator (top)** — bright green/red = accelerating, dark = decelerating. The white **T3 line** is the smoothed heartbeat; the 🟢/🔴 dots are **turn signals** where it flips.
- **Long oscillator (bottom)** — the trend filter. Institutional rule: only take buy turns when the long oscillator is **above zero** (and vice versa).

The institutional upgrades:
- **MTF alignment** — the same oscillator computed on weekly bars. Daily signals *against* the weekly trend are how retail gets chopped up.
- **Divergence** — price making new highs while the oscillator makes lower highs = momentum is quietly leaving the move. The classic institutional exit tell.
- **Event study** — the table above measures every historical turn on this exact ticker: average forward return and win rate 5/10/20 days later. If buy turns only won 45% of the time on this name, now you know — trust data, not paint.
""")

        st.markdown("---")

        # ---- 🧠 Model breakdown
        st.markdown("### 🧠 Model breakdown")
        last = comp.iloc[-1]
        sub = last[["trend", "momentum", "bxtrender", "macd", "rsi", "meanrev", "volume"]]
        sub_df = pd.DataFrame(
            {"model": [str(k) for k in sub.index],
             "score": [float(xv) for xv in sub.values]}).set_index("model")
        st.bar_chart(sub_df, height=220)
        with st.expander("❓ What does each model measure?"):
            st.markdown("""
- **trend** — price vs its 50 & 200-day averages + golden/death cross. The big picture direction.
- **momentum** — the classic academic "12-1" factor: how the stock did over the past year excluding the last month. Winners tend to keep winning.
- **macd** — momentum of momentum; catches acceleration and deceleration earlier than trend.
- **rsi** — is buying or selling pressure dominant right now (above/below 50), with a small fade at extreme readings.
- **meanrev** — Bollinger z-score: how stretched the price is from its 20-day average. Stretched rubber bands snap back.
- **volume** — do volume surges confirm the price direction? Moves without volume are suspect.
- **bxtrender** — double-smoothed momentum (RSI of an EMA spread, T3-filtered): direction from the long oscillator, timing from the short one. Less lag than MACD.

Bars pointing the **same way** = high-quality signal. Bars fighting each other = chop — usually a NO TRADE.
""")

# ===========================================================================
# 2. SCREENER
# ===========================================================================
with tab_screener:
    st.subheader("Scan the market")
    col1, col2 = st.columns([3, 1])
    with col1:
        custom = st.text_input(
            "Tickers (comma-separated) — empty = default 50-name universe",
            placeholder="AAPL, NVDA, SPY ...")
    with col2:
        min_score = st.slider("Min |score| filter", 0.0, 0.8, 0.0, 0.05)

    universe = tuple(t.strip().upper() for t in custom.split(",")
                     if t.strip()) or tuple(DEFAULT_UNIVERSE)

    if st.button("Run scan", type="primary"):
        with st.spinner(f"Downloading & scoring {len(universe)} tickers…"):
            data = fetch_many(universe, period="2y")
            rows = []
            for sym, dfr in data.items():
                try:
                    snap = latest_snapshot(dfr)
                    snap["ticker"] = sym
                    rows.append(snap)
                except Exception:
                    continue
        if not rows:
            st.error("No data returned — check tickers or try again shortly.")
        else:
            table = pd.DataFrame(rows).set_index("ticker")
            table = table[abs(table["score"]) >= min_score]
            st.session_state["scan"] = table.sort_values("score",
                                                         ascending=False)

    if "scan" in st.session_state:
        table = st.session_state["scan"]
        n_buy = int((table["signal"] == "BUY").sum())
        n_sell = int((table["signal"] == "SELL").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("BUY signals", n_buy)
        c2.metric("SELL signals", n_sell)
        c3.metric("HOLD", int(len(table)) - n_buy - n_sell)

        def _style(vv):
            return f"color: {SIGNAL_COLORS.get(vv, '')}; font-weight: 700"

        st.dataframe(
            table.style.map(_style, subset=["signal"]).background_gradient(
                subset=["score"], cmap="RdYlGn", vmin=-0.6, vmax=0.6),
            use_container_width=True, height=560)
        with st.expander("❓ How to read this table"):
            st.markdown("""
- **score** — the fused output of all six models, −1 (max bearish) to +1 (max bullish), dampened in volatility storms. ≥ +0.25 = BUY zone, ≤ −0.25 = SELL zone.
- **trend / momentum / macd / rsi_score / meanrev** — each model's individual vote, so you can see *who* is driving the score.
- **atr** — the average daily range in $; bigger = wilder stock = smaller position for the same risk.
- **ret_1m / ret_3m** — recent performance for context.

Workflow: scan → pick the extremes → take them to the **🎯 Trade desk** for the full verdict with Monte Carlo odds. Never trade off the scan alone.
""")

# ===========================================================================
# 3. BACKTEST
# ===========================================================================
with tab_backtest:
    st.subheader("Backtest the signal on any ticker")
    c1, c2, c3, c4 = st.columns(4)
    bt_tkr = c1.text_input("Ticker", value="AAPL", key="bt").upper().strip()
    bt_period = c2.selectbox("History", ["2y", "5y", "10y"], index=1)
    bt_cash = c3.number_input("Starting cash $", 500, 1_000_000, 5000,
                              step=500)
    bt_risk = c4.slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)

    if st.button("Run backtest", type="primary", key="btrun"):
        df = fetch_history(bt_tkr, period=bt_period)
        if len(df) < 260:
            st.error("Not enough history — need at least ~1 year.")
        else:
            cfg = BTConfig(starting_cash=float(bt_cash),
                           risk_per_trade=bt_risk / 100)
            res = run_backtest(df, cfg)
            cols = st.columns(5)
            for col, (k, val) in zip(cols * 2, res.metrics.items()):
                col.metric(k, val)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     name="Strategy",
                                     line=dict(width=2, color="#10b981")))
            fig.add_trace(go.Scatter(x=res.bh_equity.index, y=res.bh_equity,
                                     name="Buy & Hold",
                                     line=dict(width=1.5, dash="dot",
                                               color="#8b98a5")))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Equity $", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("❓ What do these metrics mean?"):
                st.markdown("""
- **CAGR %** — compound annual growth rate; the "interest rate" your strategy earned.
- **Sharpe** — return per unit of risk. Below 0.5 = weak, ~1 = decent, 2+ = suspicious (check for bugs 😉).
- **Sortino** — like Sharpe but only punishes *downside* wiggle. More honest for asymmetric strategies.
- **Max Drawdown %** — worst peak-to-valley loss. The number that decides whether you'd actually *survive* running this.
- **Win Rate** — % of winning trades. Low win rate is fine with big winners; the pair to watch is win rate × risk/reward.
- **vs Buy & Hold** — the brutal benchmark. A signal earns its life by better *risk-adjusted* numbers (Sharpe, drawdown), not necessarily more raw profit.
""")

            st.markdown("#### Walk-forward check (4 sequential folds)")
            st.dataframe(walk_forward(df, cfg), use_container_width=True)
            with st.expander("❓ Why walk-forward matters"):
                st.markdown("""
The same rules re-run on 4 **separate, sequential periods**. A real edge shows up in most folds; a curve-fit illusion shines in one period and dies in the rest. This is the single best overfitting detector available to a retail quant — institutional desks live by it.
""")

            if len(res.trades):
                st.markdown("#### Trade log")
                st.dataframe(res.trades, use_container_width=True, height=300)

# ===========================================================================
# 4. OPTIONS / IV SURFACE
# ===========================================================================
with tab_options:
    st.subheader("Implied volatility surface")
    c1, c2 = st.columns([2, 1])
    opt_tkr = c1.text_input("Ticker (must have listed options)", value="SPY",
                            key="opt").upper().strip()
    n_exp = c2.slider("Expiries to load", 3, 12, 8)

    if st.button("Build surface", type="primary", key="optrun"):
        with st.spinner(f"Downloading {n_exp} option chains for {opt_tkr}…"):
            spot, chain = fetch_chains(opt_tkr, max_expiries=n_exp)

        if chain.empty:
            st.error(f"No usable options data for {opt_tkr}. Try SPY, QQQ, "
                     "AAPL, NVDA or another liquid name.")
        else:
            ts = atm_term_structure(chain)
            atm_iv = float(ts["iv"].iloc[0]) if len(ts) else None
            skew = skew_25(chain)
            # Black-Scholes expected move: ~0.8 * S * IV * sqrt(T)
            exp_move = None
            if atm_iv:
                t_yrs = float(ts["dte"].iloc[0]) / 365.0
                exp_move = 0.8 * spot * (atm_iv / 100) * np.sqrt(t_yrs)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Spot", f"${spot:,.2f}")
            m2.metric("ATM IV (near expiry)",
                      f"{atm_iv:.1f}%" if atm_iv else "—")
            m3.metric("Expected move by expiry",
                      f"±${exp_move:,.2f}" if exp_move else "—")
            m4.metric("Skew (95P − 105C)",
                      f"{skew:+.1f} pts" if skew is not None else "—")
            m5.metric("Quotes used", f"{len(chain):,}")

            with st.expander("❓ What do these numbers mean?"):
                st.markdown(f"""
- **ATM IV** — the market's own forecast of {opt_tkr}'s volatility, backed by real money. High IV = the market expects fireworks.
- **Expected move** — the Black-Scholes straddle approximation (≈ 0.8 × price × IV × √time): how far the market prices {opt_tkr} to move by the nearest expiry, in dollars. Your target inside this range = realistic; far outside = you're betting against the options market.
- **Skew** — how much more expensive downside puts are vs upside calls. Positive & large = crash insurance in demand (fear). Near zero or negative = complacency or upside chase.
""")

            strikes, dtes, grid = build_surface(chain)
            if grid.size:
                fig = go.Figure(data=[go.Surface(
                    x=strikes, y=dtes, z=grid, colorscale="Jet",
                    colorbar=dict(title="IV %"), connectgaps=True)])
                fig.update_layout(
                    height=620,
                    scene=dict(xaxis_title="Strike",
                               yaxis_title="Days to expiry",
                               zaxis_title="Implied vol (%)",
                               camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.7))),
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"{opt_tkr} IV surface — spot ${spot:,.2f}",
                    **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("❓ How to read the 3D surface"):
                    st.markdown("""
Every point = the implied volatility of one option (strike × days to expiry). What pros look for:
- **The smile/smirk** (curve across strikes): deep OTM options cost more vol — the market pays up for tail protection. A steep left wing = crash fear.
- **The term slope** (across expiries): upward = calm now, uncertainty later (normal). **Inverted** = the market is bracing for a near-term event (earnings, Fed, war).
- **Bumps** at specific expiries = event risk priced exactly there. Find the bump, find the catalyst date.
""")

            colA, colB = st.columns(2)
            with colA:
                exps = sorted(chain["expiry"].unique())
                pick = st.selectbox("Smile for expiry", exps, index=0)
                sm = chain[chain["expiry"] == pick]
                figs = go.Figure()
                for side, color in (("P", "#ef4444"), ("C", "#10b981")):
                    s = sm[sm["type"] == side].groupby("strike")["iv"].median()
                    figs.add_trace(go.Scatter(
                        x=s.index, y=s.values, mode="lines+markers",
                        name="Puts" if side == "P" else "Calls",
                        line=dict(color=color)))
                figs.add_vline(x=spot, line_dash="dot",
                               annotation_text="spot")
                figs.update_layout(height=360, xaxis_title="Strike",
                                   yaxis_title="IV %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figs, use_container_width=True)
            with colB:
                st.markdown("**ATM term structure**")
                figt = go.Figure(go.Scatter(x=ts["dte"], y=ts["iv"],
                                            mode="lines+markers",
                                            line=dict(color="#6ee7b7")))
                figt.update_layout(height=360, xaxis_title="Days to expiry",
                                   yaxis_title="ATM IV %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figt, use_container_width=True)
                st.caption("Upward slope = calm; inverted = near-term event fear.")

            st.markdown("#### Option chain + Greeks (selected expiry)")
            sel = chain[chain["expiry"] == pick].copy()
            g = bs_greeks(spot, sel["strike"].values,
                          sel["dte"].values / 365.0,
                          sel["iv"].values / 100.0,
                          (sel["type"] == "C").values)
            sel = pd.concat([sel.reset_index(drop=True), g], axis=1)
            show_cols = ["type", "strike", "last", "bid", "ask", "iv",
                         "volume", "oi", "delta", "gamma", "vega", "theta"]
            st.dataframe(sel[show_cols].sort_values(["type", "strike"]),
                         use_container_width=True, height=380)
            with st.expander("❓ Greeks cheat-sheet"):
                st.markdown("""
Black-Scholes sensitivities — what moves your option's price:
- **Delta** — $ change per $1 move in the stock. Also ≈ the market's probability the option expires in the money.
- **Gamma** — how fast delta itself changes. High gamma = the option's behaviour flips quickly near the strike.
- **Vega** — $ change per 1-point change in IV. Buying high-IV options = paying up; if IV collapses (e.g. after earnings), vega is how much you bleed.
- **Theta** — $ lost per day from time decay. The rent you pay for holding the option.

Data ~15 min delayed (Yahoo). Educational, not advice.
""")

# ===========================================================================
# 5. POSITION SIZING
# ===========================================================================
with tab_sizing:
    st.subheader("How many shares for your account?")
    c1, c2, c3 = st.columns(3)
    acct = c1.number_input("Account $", 100, 1_000_000, 5000, step=100,
                           key="ps_acct")
    risk_p = c2.slider("Max loss per trade %", 0.5, 3.0, 1.0, 0.25,
                       key="ps_risk")
    ps_tkr = c3.text_input("Ticker", value="MSFT", key="ps").upper().strip()

    if st.button("Calculate", type="primary", key="pscalc"):
        df = fetch_history(ps_tkr, period="6mo")
        if df.empty:
            st.error(f"No data for {ps_tkr}")
        else:
            price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            stop_dist = 2.5 * a
            risk_dollars = acct * risk_p / 100
            shares = int(min(risk_dollars / stop_dist, acct / price))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${price:,.2f}")
            c2.metric("Stop distance (2.5×ATR)", f"${stop_dist:,.2f}")
            c3.metric("Shares to buy", shares)
            c4.metric("Position value", f"${shares * price:,.0f}")
            st.info(f"Stop-loss ≈ **${price - stop_dist:,.2f}**. If hit, you "
                    f"lose ≈ **${shares * stop_dist:,.0f}** ({risk_p}% of the "
                    f"account).")
            with st.expander("❓ Why size by risk instead of by dollars?"):
                st.markdown("""
Buying "$1,000 of each stock" means a calm stock and a wild stock give you totally different risk. Sizing by **risk** (account % ÷ stop distance) equalises it: every trade can hurt you the same, small, survivable amount. With a $5,000 account, ruin-avoidance *is* the strategy — a 50% drawdown needs a 100% gain just to get back to even. Professional desks size everything this way.
""")
