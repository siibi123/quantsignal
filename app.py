"""QuantSignal — free open quant screener & backtester for US stocks.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub → share.streamlit.io  (see README.md)
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.backtest import BTConfig, run_backtest, walk_forward
from quant.data import DEFAULT_UNIVERSE, fetch_history, fetch_many
from quant.signals import BUY_TH, SELL_TH, atr, composite, latest_snapshot

st.set_page_config(page_title="QuantSignal", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Header + disclaimer
# ---------------------------------------------------------------------------
st.title("📈 QuantSignal")
st.caption(
    "Multi-model quant signals for US stocks — trend, momentum, mean reversion, "
    "volatility regime & volume, combined into one score, with honest walk-forward "
    "backtesting. **Educational tool, not financial advice. Past performance does "
    "not predict future results.**"
)

tab_screener, tab_ticker, tab_backtest, tab_sizing = st.tabs(
    ["🔍 Screener", "📊 Ticker analysis", "🧪 Backtest", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#0a8f5b", "SELL": "#c0392b", "HOLD": "#8a8a8a"}

# ---------------------------------------------------------------------------
# 1. Screener
# ---------------------------------------------------------------------------
with tab_screener:
    st.subheader("Scan the market")
    col1, col2 = st.columns([3, 1])
    with col1:
        custom = st.text_input(
            "Tickers (comma-separated) — leave empty for the default 50-name universe",
            placeholder="AAPL, NVDA, SPY ...",
        )
    with col2:
        min_score = st.slider("Min |score| filter", 0.0, 0.8, 0.0, 0.05)

    universe = tuple(
        t.strip().upper() for t in custom.split(",") if t.strip()
    ) or tuple(DEFAULT_UNIVERSE)

    if st.button("Run scan", type="primary"):
        with st.spinner(f"Downloading & scoring {len(universe)} tickers…"):
            data = fetch_many(universe, period="2y")
            rows = []
            for tkr, df in data.items():
                try:
                    snap = latest_snapshot(df)
                    snap["ticker"] = tkr
                    rows.append(snap)
                except Exception:
                    continue
        if not rows:
            st.error("No data returned — check tickers or try again in a minute.")
        else:
            table = pd.DataFrame(rows).set_index("ticker")
            table = table[abs(table["score"]) >= min_score]
            table = table.sort_values("score", ascending=False)
            st.session_state["scan"] = table

    if "scan" in st.session_state:
        table = st.session_state["scan"]
        n_buy = (table["signal"] == "BUY").sum()
        n_sell = (table["signal"] == "SELL").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("BUY signals", int(n_buy))
        c2.metric("SELL signals", int(n_sell))
        c3.metric("HOLD", int(len(table) - n_buy - n_sell))

        def _style(v):
            return f"color: {SIGNAL_COLORS.get(v, '')}; font-weight: 700"

        st.dataframe(
            table.style.map(_style, subset=["signal"]).background_gradient(
                subset=["score"], cmap="RdYlGn", vmin=-0.6, vmax=0.6
            ),
            use_container_width=True,
            height=560,
        )
        st.caption(
            "Score ∈ [-1, +1]. BUY ≥ +0.25, SELL ≤ −0.25. Sub-scores show which "
            "model drives the signal. Always backtest a ticker before acting."
        )

# ---------------------------------------------------------------------------
# 2. Ticker analysis
# ---------------------------------------------------------------------------
with tab_ticker:
    st.subheader("Deep-dive one ticker")
    tkr = st.text_input("Ticker", value="NVDA", key="single").upper().strip()
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)

    if st.button("Analyze", type="primary", key="an"):
        df = fetch_history(tkr, period=period)
        if df.empty:
            st.error(f"No data for {tkr}")
        else:
            comp = composite(df)
            last = comp.iloc[-1]
            sig = last["signal"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"${df['Close'].iloc[-1]:,.2f}")
            m2.metric("Composite score", f"{last['score']:+.3f}")
            m3.markdown(
                f"### <span style='color:{SIGNAL_COLORS[sig]}'>{sig}</span>",
                unsafe_allow_html=True,
            )
            m4.metric("ATR (14d)", f"${atr(df).iloc[-1]:,.2f}")

            # Price chart with signal markers
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=tkr,
            ))
            buys = comp[comp["signal"] == "BUY"].index
            sells = comp[comp["signal"] == "SELL"].index
            fig.add_trace(go.Scatter(
                x=buys, y=df.loc[buys, "Low"] * 0.985, mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="#0a8f5b"),
                name="BUY zone",
            ))
            fig.add_trace(go.Scatter(
                x=sells, y=df.loc[sells, "High"] * 1.015, mode="markers",
                marker=dict(symbol="triangle-down", size=8, color="#c0392b"),
                name="SELL zone",
            ))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Score history + sub-score breakdown
            st.line_chart(comp["score"], height=180)
            st.caption(f"Composite score history (BUY ≥ {BUY_TH}, SELL ≤ {SELL_TH})")
            sub = last[["trend", "momentum", "macd", "rsi", "meanrev", "volume"]]
            sub_df = pd.DataFrame(
                {"model": [str(k) for k in sub.index],
                 "score": [float(v) for v in sub.values]}
            ).set_index("model")
            st.bar_chart(sub_df, height=220)
            st.caption("Current sub-scores — which model is driving the signal")

# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Backtest the signal on any ticker")
    c1, c2, c3, c4 = st.columns(4)
    bt_tkr = c1.text_input("Ticker", value="AAPL", key="bt").upper().strip()
    bt_period = c2.selectbox("History", ["2y", "5y", "10y"], index=1)
    bt_cash = c3.number_input("Starting cash $", 500, 1_000_000, 5000, step=500)
    bt_risk = c4.slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)

    if st.button("Run backtest", type="primary", key="btrun"):
        df = fetch_history(bt_tkr, period=bt_period)
        if len(df) < 260:
            st.error("Not enough history — need at least ~1 year of daily bars.")
        else:
            cfg = BTConfig(starting_cash=float(bt_cash),
                           risk_per_trade=bt_risk / 100)
            res = run_backtest(df, cfg)

            cols = st.columns(5)
            for col, (k, v) in zip(cols * 2, res.metrics.items()):
                col.metric(k, v)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity,
                                     name="Strategy", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=res.bh_equity.index, y=res.bh_equity,
                                     name="Buy & Hold",
                                     line=dict(width=1.5, dash="dot")))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Equity $")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Walk-forward check (4 sequential folds)")
            st.caption(
                "The same rules applied to 4 separate periods. If it only wins in "
                "one fold, the edge is probably not robust."
            )
            wf = walk_forward(df, cfg)
            st.dataframe(wf, use_container_width=True)

            if len(res.trades):
                st.markdown("#### Trade log")
                st.dataframe(res.trades, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# 4. Position sizing
# ---------------------------------------------------------------------------
with tab_sizing:
    st.subheader("How many shares for your account?")
    st.caption("Risk-based sizing: if the stop is hit you lose only the % you chose.")
    c1, c2, c3 = st.columns(3)
    acct = c1.number_input("Account $", 100, 1_000_000, 5000, step=100)
    risk_pct = c2.slider("Max loss per trade %", 0.5, 3.0, 1.0, 0.25)
    ps_tkr = c3.text_input("Ticker", value="MSFT", key="ps").upper().strip()

    if st.button("Calculate", type="primary", key="pscalc"):
        df = fetch_history(ps_tkr, period="6mo")
        if df.empty:
            st.error(f"No data for {ps_tkr}")
        else:
            price = float(df["Close"].iloc[-1])
            a = float(atr(df).iloc[-1])
            stop_dist = 2.5 * a
            risk_dollars = acct * risk_pct / 100
            shares_risk = risk_dollars / stop_dist
            shares_cash = acct / price
            shares = int(min(shares_risk, shares_cash))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${price:,.2f}")
            c2.metric("Stop distance (2.5×ATR)", f"${stop_dist:,.2f}")
            c3.metric("Shares to buy", shares)
            c4.metric("Position value", f"${shares * price:,.0f}")
            st.info(
                f"Stop-loss level ≈ **${price - stop_dist:,.2f}**. If hit, you lose "
                f"≈ **${shares * stop_dist:,.0f}** ({risk_pct}% of the account). "
                f"With a ${acct:,} account, position limits are what keep you in "
                f"the game long enough for any edge to matter."
            )
