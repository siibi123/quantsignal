"""QuantSignal — quant screener, trade desk & backtester for US stocks."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.backtest import BTConfig, run_backtest, walk_forward
from quant.data import DEFAULT_UNIVERSE, fetch_history, fetch_many
from quant.options import (atm_term_structure, bs_greeks, build_surface,
                           fetch_chains, skew_25)
from quant.signals import BUY_TH, SELL_TH, atr, composite, latest_snapshot
from quant.verdict import analyze

st.set_page_config(page_title="QuantSignal", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1 { font-weight: 800; letter-spacing: -0.5px; }
[data-testid="stMetric"] {
  background: #131a22; border: 1px solid #1f2a36;
  border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricValue"] { font-weight: 700; }
.verdict {
  border-radius: 16px; padding: 26px 30px; margin: 6px 0 14px 0;
  display: flex; align-items: center; justify-content: space-between;
}
.verdict h2 { margin: 0; font-size: 2.2rem; font-weight: 800; }
.verdict .sub { opacity: .85; font-size: .95rem; margin-top: 4px; }
.v-long  { background: linear-gradient(135deg,#064e3b,#065f46); border:1px solid #10b981; }
.v-short { background: linear-gradient(135deg,#7f1d1d,#991b1b); border:1px solid #ef4444; }
.v-none  { background: linear-gradient(135deg,#1f2937,#374151); border:1px solid #6b7280; }
.reason-pro, .reason-con {
  border-radius: 10px; padding: 8px 14px; margin: 5px 0; font-size: .92rem;
}
.reason-pro { background:#0b2e22; border-left: 3px solid #10b981; }
.reason-con { background:#2e0f0f; border-left: 3px solid #ef4444; }
.conv-wrap { background:#1f2a36; border-radius: 8px; height: 14px; width: 100%; }
.conv-bar  { height: 14px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 QuantSignal")
st.caption(
    "Multi-model quant signals for US stocks — trend, momentum, mean reversion, "
    "volatility regime, volume & options skew — fused into one decision with "
    "honest walk-forward backtesting. **Educational tool, not financial advice.**"
)

tab_desk, tab_screener, tab_backtest, tab_options, tab_sizing = st.tabs(
    ["🎯 Trade desk", "🔍 Screener", "🧪 Backtest",
     "🌋 Options / IV surface", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#8a8a8a"}
VERDICT_CLASS = {"LONG": "v-long", "SHORT": "v-short", "NO TRADE": "v-none"}
VERDICT_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}

# ---------------------------------------------------------------------------
# 1. TRADE DESK — the flagship
# ---------------------------------------------------------------------------
with tab_desk:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4])
    tkr = c1.text_input("Ticker", value="NVDA", key="desk").upper().strip()
    account = c2.number_input("Account $", 500, 1_000_000, 5000, step=500)
    risk_pct = c3.slider("Risk/trade %", 0.5, 3.0, 1.0, 0.25)
    use_opts = c4.toggle("Include options skew (slower)", value=False)

    if st.button("Run desk analysis", type="primary", key="deskrun"):
        with st.spinner("Crunching signals, edge history & levels…"):
            df = fetch_history(tkr, period="2y")
            if df.empty or len(df) < 260:
                st.error(f"Not enough data for {tkr} (need ~1y of history).")
                st.stop()
            skew = None
            if use_opts:
                try:
                    _, chain = fetch_chains(tkr, max_expiries=4)
                    skew = skew_25(chain)
                except Exception:
                    skew = None
            v = analyze(df, account=account, risk_pct=risk_pct, skew=skew)

        # ---- Verdict banner --------------------------------------------
        cls = VERDICT_CLASS[v["verdict"]]
        conv_color = ("#10b981" if v["verdict"] == "LONG"
                      else "#ef4444" if v["verdict"] == "SHORT" else "#6b7280")
        st.markdown(f"""
        <div class="verdict {cls}">
          <div>
            <h2>{VERDICT_EMOJI[v['verdict']]} {v['verdict']} — {tkr}</h2>
            <div class="sub">Composite {v['score']:+.2f} · {v['agree']}/6 models aligned ·
              signal Sharpe on {tkr}: {v['sharpe']} ({v['n_trades']} trades)</div>
          </div>
          <div style="min-width:220px">
            <div style="font-size:.85rem;opacity:.8;margin-bottom:4px">
              Conviction {v['conviction']}/100</div>
            <div class="conv-wrap"><div class="conv-bar"
              style="width:{v['conviction']}%;background:{conv_color}"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Levels ------------------------------------------------------
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
        else:
            st.info("**Standing aside is a position.** The desk found no edge "
                    "worth risking money on right now — that's the system "
                    "working, not failing.")

        # ---- Reasons ------------------------------------------------------
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

        # ---- Chart with levels --------------------------------------------
        comp = composite(df)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index[-252:], open=df["Open"][-252:], high=df["High"][-252:],
            low=df["Low"][-252:], close=df["Close"][-252:], name=tkr))
        if v["verdict"] != "NO TRADE":
            for lvl, name, color in ((v["entry"], "Entry", "#e6edf3"),
                                     (v["stop"], "Stop", "#ef4444"),
                                     (v["target"], "Target", "#10b981")):
                fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                              annotation_text=f"{name} ${lvl:,.2f}",
                              annotation_font_color=color)
        buys = comp[comp["signal"] == "BUY"].index[-252:]
        sells = comp[comp["signal"] == "SELL"].index[-252:]
        buys = [d for d in buys if d in df.index[-252:]]
        sells = [d for d in sells if d in df.index[-252:]]
        fig.add_trace(go.Scatter(x=buys, y=df.loc[buys, "Low"] * 0.985,
                                 mode="markers", name="BUY zone",
                                 marker=dict(symbol="triangle-up", size=7,
                                             color="#10b981")))
        fig.add_trace(go.Scatter(x=sells, y=df.loc[sells, "High"] * 1.015,
                                 mode="markers", name="SELL zone",
                                 marker=dict(symbol="triangle-down", size=7,
                                             color="#ef4444")))
        fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # ---- Model breakdown ----------------------------------------------
        last = comp.iloc[-1]
        sub = last[["trend", "momentum", "macd", "rsi", "meanrev", "volume"]]
        sub_df = pd.DataFrame(
            {"model": [str(k) for k in sub.index],
             "score": [float(x) for x in sub.values]}).set_index("model")
        st.bar_chart(sub_df, height=220)
        st.caption("Sub-scores — which models drive (or fight) the verdict")

# ---------------------------------------------------------------------------
# 2. Screener
# ---------------------------------------------------------------------------
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

        def _style(v):
            return f"color: {SIGNAL_COLORS.get(v, '')}; font-weight: 700"

        st.dataframe(
            table.style.map(_style, subset=["signal"]).background_gradient(
                subset=["score"], cmap="RdYlGn", vmin=-0.6, vmax=0.6),
            use_container_width=True, height=560)
        st.caption("Found a candidate? Take it to the 🎯 Trade desk for the "
                   "full verdict before doing anything.")

# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------
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
                                     name="Strategy", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=res.bh_equity.index, y=res.bh_equity,
                                     name="Buy & Hold",
                                     line=dict(width=1.5, dash="dot")))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Equity $",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Walk-forward check (4 sequential folds)")
            st.caption("Same rules, 4 separate periods. Winning in only one "
                       "fold = fragile edge.")
            st.dataframe(walk_forward(df, cfg), use_container_width=True)

            if len(res.trades):
                st.markdown("#### Trade log")
                st.dataframe(res.trades, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# 4. Options / IV surface
# ---------------------------------------------------------------------------
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

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Spot", f"${spot:,.2f}")
            m2.metric("ATM IV (near expiry)",
                      f"{atm_iv:.1f}%" if atm_iv else "—")
            m3.metric("Skew (95P − 105C)",
                      f"{skew:+.1f} pts" if skew is not None else "—",
                      help="Positive = puts richer than calls (crash fear)")
            m4.metric("Quotes used", f"{len(chain):,}")

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
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=f"{opt_tkr} IV surface — spot ${spot:,.2f}")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("OTM quotes only, median IV per strike, "
                           "interpolated. Drag to rotate, scroll to zoom.")

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
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figs, use_container_width=True)
            with colB:
                st.markdown("**ATM term structure**")
                figt = go.Figure(go.Scatter(x=ts["dte"], y=ts["iv"],
                                            mode="lines+markers"))
                figt.update_layout(height=360, xaxis_title="Days to expiry",
                                   yaxis_title="ATM IV %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figt, use_container_width=True)
                st.caption("Upward slope = calm; inverted = near-term event "
                           "fear.")

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
            st.caption("Greeks are Black-Scholes estimates from quoted IV; "
                       "data is ~15 min delayed (Yahoo). Educational, not "
                       "advice.")

# ---------------------------------------------------------------------------
# 5. Position sizing
# ---------------------------------------------------------------------------
with tab_sizing:
    st.subheader("How many shares for your account?")
    st.caption("Risk-based sizing: if the stop is hit you lose only the % "
               "you chose.")
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
                    f"account). Position limits keep you in the game long "
                    f"enough for any edge to matter.")
