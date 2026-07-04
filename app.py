"""QuantSignal — neon-terminal quant desk for US stocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quant.advanced import ewma_vol, kelly, regime_quadrant, support_resistance
from quant.anomalies import ANOMALY_INFO
from quant.master import run_master
from quant.backtest import BTConfig, run_backtest, walk_forward
from quant.bxtrender import (bxtrender, detect_divergence, event_study,
                             weekly_alignment)
from quant.data import DEFAULT_UNIVERSE, fetch_history, fetch_many
from quant.flow import gex_profile, gex_summary, unusual_flow
from quant.seasonality import fundamental_snapshot, monthly_seasonality
from quant.levels import fib_levels, hurst
from quant.montecarlo import cone, simulate, trade_odds
from quant.options import (atm_term_structure, bs_greeks, build_surface,
                           fetch_chains, max_pain, put_call_ratio, skew_25)
from quant.signals import BUY_TH, SELL_TH, atr, composite, latest_snapshot
from quant.verdict import analyze

st.set_page_config(page_title="QuantSignal", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# NEON TERMINAL styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(16,185,129,.13), transparent),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(59,130,246,.07), transparent),
    linear-gradient(180deg, #070b10 0%, #0b0f14 100%);
}
.stApp::before {
  content:""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(16,185,129,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(16,185,129,.03) 1px, transparent 1px);
  background-size: 42px 42px;
}

.hero-title {
  font-size: 3.2rem; font-weight: 900; letter-spacing: -2px; margin: 0;
  background: linear-gradient(90deg, #10b981 0%, #34d399 35%, #22d3ee 75%, #60a5fa 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 24px rgba(16,185,129,.35));
}
.hero-sub { color: #8b98a5; font-size: .95rem; margin-top: 4px; }
.chips { margin-top: 10px; }
.chip {
  display:inline-block; padding: 4px 12px; margin: 0 6px 6px 0;
  border-radius: 999px; font-size: .75rem; font-weight: 600;
  color: #6ee7b7; background: rgba(16,185,129,.08);
  border: 1px solid rgba(16,185,129,.35);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 6px; background: rgba(19,26,34,.6); padding: 6px;
  border-radius: 14px; border: 1px solid #1f2a36;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px; padding: 8px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(16,185,129,.25), rgba(34,211,238,.12)) !important;
  border: 1px solid rgba(16,185,129,.5);
  box-shadow: 0 0 18px rgba(16,185,129,.25);
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(23,32,42,.85) 0%, rgba(14,20,27,.9) 100%);
  border: 1px solid #1f2a36; border-radius: 16px; padding: 14px 16px;
  backdrop-filter: blur(8px); transition: all .25s ease;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(16,185,129,.6);
  box-shadow: 0 0 22px rgba(16,185,129,.18); transform: translateY(-2px);
}
[data-testid="stMetricValue"] { font-weight: 800; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { color: #8b98a5; }

div[data-testid="stExpander"] {
  border: 1px dashed rgba(16,185,129,.35); border-radius: 14px;
  background: rgba(15,21,29,.7);
}

.verdict {
  border-radius: 20px; padding: 30px 34px; margin: 10px 0 18px 0;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 10px 44px rgba(0,0,0,.45); backdrop-filter: blur(6px);
}
.verdict h2 { margin: 0; font-size: 2.5rem; font-weight: 900; letter-spacing: -1px; }
.verdict .sub { opacity: .85; font-size: .95rem; margin-top: 6px; }
.v-long  { background: linear-gradient(135deg, rgba(5,59,45,.95), rgba(6,95,70,.9) 60%, rgba(4,120,87,.85));
           border: 1px solid #10b981; box-shadow: 0 0 46px rgba(16,185,129,.30); }
.v-short { background: linear-gradient(135deg, rgba(95,20,20,.95), rgba(153,27,27,.9) 60%, rgba(185,28,28,.85));
           border: 1px solid #ef4444; box-shadow: 0 0 46px rgba(239,68,68,.30); }
.v-none  { background: linear-gradient(135deg, rgba(26,35,50,.95), rgba(43,54,72,.9) 60%, rgba(55,65,81,.85));
           border: 1px solid #6b7280; }

.reason-pro, .reason-con {
  border-radius: 10px; padding: 9px 14px; margin: 6px 0; font-size: .92rem;
}
.reason-pro { background: rgba(11,46,34,.8); border-left: 3px solid #10b981; }
.reason-con { background: rgba(46,15,15,.8); border-left: 3px solid #ef4444; }

.conv-wrap { background:#1f2a36; border-radius: 8px; height: 16px; width: 100%; overflow:hidden; }
.conv-bar  { height: 16px; border-radius: 8px; position: relative; }
.conv-bar::after {
  content:""; position:absolute; inset:0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent);
  animation: shimmer 2.2s infinite; transform: translateX(-100%);
}
@keyframes shimmer { 100% { transform: translateX(100%); } }

.regime-badge {
  display:inline-block; padding: 8px 18px; border-radius: 12px;
  font-weight: 800; font-size: 1.05rem; letter-spacing: .3px;
  background: rgba(19,26,34,.9); border: 1px solid #2a3644;
}
hr { border-color: #1f2a36; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #1f2a36; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(16,185,129,.5); }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<p class="hero-title">QuantSignal</p>
<p class="hero-sub">Institutional-grade quant desk — every model fused into one decision.
Educational tool, not financial advice.</p>
<div class="chips">
<span class="chip">7-model composite</span><span class="chip">B-Xtrender</span>
<span class="chip">Monte Carlo</span><span class="chip">Kelly sizing</span>
<span class="chip">EWMA vol</span><span class="chip">Fibonacci</span>
<span class="chip">Black-Scholes</span><span class="chip">IV surface</span>
<span class="chip">Max pain</span><span class="chip">Walk-forward</span>
</div>
''', unsafe_allow_html=True)
st.write("")

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Inter", color="#e6edf3"))

tab_master, tab_desk, tab_screener, tab_backtest, tab_options, tab_sizing = st.tabs(
    ["🧬 Alpha engine", "🎯 Trade desk", "🔍 Screener", "🧪 Backtest",
     "🌋 Options / IV surface", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#8a8a8a"}
VERDICT_CLASS = {"LONG": "v-long", "SHORT": "v-short", "NO TRADE": "v-none"}
VERDICT_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}
FIB_COLORS = {"0": "#6ee7b7", "0.236": "#34d399", "0.382": "#fbbf24",
              "0.5": "#f59e0b", "0.618": "#f97316", "0.786": "#ef4444",
              "1": "#dc2626"}


# ===========================================================================
# 0. ALPHA ENGINE — the master algorithm
# ===========================================================================
with tab_master:
    st.subheader("🧬 The master algorithm — one answer: what to do now")
    st.caption("Research-backed cross-sectional anomalies × the 7-model verdict "
               "engine × market regime gate × portfolio risk caps. "
               "Built on published, replicated papers (citations below).")
    c1, c2, c3, c4 = st.columns(4)
    ma_acct = c1.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="ma_acct")
    ma_risk = c2.slider("Risk per position %", 0.5, 2.0, 1.0, 0.25,
                        key="ma_risk")
    ma_maxpos = c3.slider("Max positions", 2, 6, 4, key="ma_pos")
    ma_custom = c4.text_input("Universe (empty = default 50)",
                              placeholder="AAPL, NVDA ...", key="ma_uni")

    if st.button("🚀 Run the machine", type="primary", key="ma_run"):
        uni = tuple(t.strip().upper() for t in ma_custom.split(",")
                    if t.strip()) or tuple(DEFAULT_UNIVERSE)
        prog = st.progress(0, text="Downloading universe…")
        data = fetch_many(uni, period="2y")
        prog.progress(30, text="Downloading SPY (market gate)…")
        spy = fetch_history("SPY", period="2y")
        prog.progress(45, text="Ranking cross-sectional anomalies…")
        res = run_master(data, spy, account=float(ma_acct),
                         risk_pct=ma_risk, max_positions=ma_maxpos)
        prog.progress(100, text="Done")
        prog.empty()

        if "error" in res:
            st.error(res["error"])
            st.stop()

        # --- Market gate ---------------------------------------------------
        reg = res["regime"]
        g1, g2, g3, g4 = st.columns([1.7, 1, 1, 1])
        g1.markdown(f"<div class='regime-badge'>{reg['regime']} (SPY)</div>"
                    f"<div style='color:#8b98a5;font-size:.85rem;margin-top:6px'>"
                    f"{reg['playbook']}</div>", unsafe_allow_html=True)
        g2.metric("Capital allowed to deploy", f"{res['exposure_pct']}%")
        g3.metric("Portfolio risk if all stops hit",
                  f"${res['total_risk']:,.0f} ({res['total_risk_pct']}%)")
        g4.metric("Cash left", f"${res['cash']:,.0f} ({res['cash_pct']}%)")

        # --- THE PLAN --------------------------------------------------------
        st.markdown("## 📋 The plan")
        if len(res["plan"]):
            st.dataframe(res["plan"], use_container_width=True,
                         hide_index=True)
            st.success(f"**Do this:** open the {len(res['plan'])} position(s) "
                       f"above with the exact share counts, place the stops "
                       f"immediately, keep ${res['cash']:,.0f} in cash. "
                       f"If a stop hits — you're out, no negotiating with it.")
        else:
            st.info("**The machine says: do nothing.** No candidate passed all "
                    "four gates (anomaly rank → verdict → regime → risk). "
                    "Cash is a position; the next setup will come to you.")

        st.warning(f"**Honesty layer (McLean & Pontiff 2016):** published "
                   f"anomalies earn ~{res['haircut_pct']}% LESS after "
                   f"publication as arbitrageurs crowd in. Whatever edge this "
                   f"engine finds, assume roughly half survives in live "
                   f"trading. That is why risk caps matter more than signal "
                   f"strength.")

        # --- Anomaly ranking table ---------------------------------------------
        st.markdown("### 🏆 Cross-sectional anomaly ranking")
        st.dataframe(
            res["ranks"].style.background_gradient(subset=["alpha"],
                                                   cmap="RdYlGn"),
            use_container_width=True, height=480)
        st.caption(f"Bottom of the table = research says avoid: "
                   f"{', '.join(res['avoid'])}")

        with st.expander("📚 The research behind each column (SSRN / journals)"):
            for key, (name, paper, desc) in ANOMALY_INFO.items():
                st.markdown(f"- **{name}** (`{key}`) — *{paper}*: {desc}")
            st.markdown("""
---
**Meta-research this engine is built on:**
- *Jensen, Kelly & Pedersen (2023, Journal of Finance)* — most published factors replicate, cluster into 13 themes, and work out-of-sample across 93 countries. Anomaly investing is real.
- *McLean & Pontiff (2016, Journal of Finance)* — but returns are ~26% lower out-of-sample and ~58% lower post-publication. Hence the haircut above.
- *Bali, Brown, Murray & Tang (2017)* — lottery demand (MAX) largely subsumes the beta anomaly, which is why MAX gets a heavy weight here.

**How the fusion works:** each anomaly is z-scored ACROSS the universe (cross-sectional, exactly as defined in the papers), weighted, and summed into `alpha`. The top decile then has to survive the 7-model time-series verdict, the SPY regime gate, and the portfolio heat cap. Four independent filters — most stocks fail at least one, and that's the point.
""")

        # --- Considered but rejected ------------------------------------------
        rej = [v for v in res["considered"] if v["verdict"] != "LONG"]
        if rej:
            st.markdown("### 🚫 High-alpha names the verdict engine rejected")
            rej_df = pd.DataFrame([{
                "ticker": v["ticker"], "alpha rank %": v["pct_rank"],
                "verdict": v["verdict"], "conviction": v["conviction"],
                "why (top reason against)": (v["reasons_con"][0]
                                             if v["reasons_con"] else "—"),
            } for v in rej])
            st.dataframe(rej_df, use_container_width=True, hide_index=True)
            st.caption("Good anomaly scores, bad timing — the whole point of "
                       "layering time-series checks on top of cross-sectional "
                       "ranks.")

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
        with st.spinner("Crunching 7 models, Monte Carlo, vol forecast & levels…"):
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
            reg = regime_quadrant(df)
            vol = ewma_vol(df)
            sr = support_resistance(df)
            direction = 1 if v["verdict"] != "SHORT" else -1
            paths = simulate(df, days=30, n_paths=2000)
            odds = trade_odds(paths, v["entry"], v["stop"], v["target"],
                              direction)
            bands = cone(paths)
            p_win = (odds["p_target_first"] /
                     max(odds["p_target_first"] + odds["p_stop_first"], 1e-9))
            kel = kelly(p_win, v["rr"])

        # ---- Regime + vol forecast row --------------------------------------
        rg1, rg2, rg3, rg4 = st.columns([1.6, 1, 1, 1])
        rg1.markdown(f"<div class='regime-badge'>{reg['regime']}</div>"
                     f"<div style='color:#8b98a5;font-size:.85rem;margin-top:6px'>"
                     f"{reg['playbook']}</div>", unsafe_allow_html=True)
        rg2.metric("EWMA vol (annual)", f"{vol['sigma_annual_pct']}%")
        rg3.metric("Expected move (1 day)", f"±${vol['expected_move_1d']:,.2f}")
        rg4.metric("Hurst exponent", h,
                   delta="trending" if h > 0.55 else
                   "mean-reverting" if h < 0.45 else "random walk",
                   delta_color="off")
        with st.expander("❓ Regime, EWMA volatility & Hurst — why they matter"):
            st.markdown("""
- **Regime quadrant** — price vs its 200-day average (bull/bear) × current volatility vs its own history (calm/storm). Each quadrant has a playbook; most losing streaks come from running a bull-calm playbook in a bear-storm.
- **EWMA vol (RiskMetrics λ=0.94)** — the industry-standard forecast of tomorrow's volatility, weighting recent days most. The ± number is the *expected* one-day move: intraday wiggles inside it are noise, not signal.
- **Hurst exponent** — the ticker's memory. >0.5 moves tend to continue (trust trend models), <0.5 they reverse (trust mean-reversion), ≈0.5 random walk.
""")

        # ---- Verdict banner ---------------------------------------------------
        cls = VERDICT_CLASS[v["verdict"]]
        conv_color = ("#10b981" if v["verdict"] == "LONG"
                      else "#ef4444" if v["verdict"] == "SHORT" else "#6b7280")
        st.markdown(f"""
        <div class="verdict {cls}">
          <div>
            <h2>{VERDICT_EMOJI[v['verdict']]} {v['verdict']} — {tkr}</h2>
            <div class="sub">Composite {v['score']:+.2f} · {v['agree']}/7 models aligned ·
              signal Sharpe on {tkr}: {v['sharpe']} ({v['n_trades']} trades)</div>
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
            st.markdown("""
The verdict fuses seven models (trend, momentum, B-Xtrender, MACD, RSI, mean-reversion, volume) into one composite score, then demands: models agreeing, a calm volatility regime, a *proven* historical edge on this exact ticker, and risk/reward ≥ 1.3. Fail any → **NO TRADE**. Standing aside is a position.

**Conviction (0–100)**: 40% signal strength + 25% model agreement + 15% volatility regime + 20% historical edge (± options skew). Above 55 = tradeable.
""")

        # ---- Levels -------------------------------------------------------------
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
- **Stop** = entry ± 2.5×ATR — wide enough to survive noise, tight enough to cap damage.
- **Target** = the 63-day swing level, or a 2R measured move on breakouts to new highs.
- **Shares** = sized so a stop-out loses exactly your chosen % of the account.
""")
        else:
            st.info("**Standing aside is a position.** The desk found no edge "
                    "worth risking money on right now — that's the system "
                    "working, not failing.")

        # ---- Reasons -------------------------------------------------------------
        colp, colc = st.columns(2)
        with colp:
            st.markdown("**✅ For**")
            for r in v["reasons_pro"]:
                st.markdown(f"<div class='reason-pro'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_pro"]:
                st.markdown("<div class='reason-con'>Nothing working in favour "
                            "right now</div>", unsafe_allow_html=True)
        with colc:
            st.markdown("**⚠️ Against**")
            for r in v["reasons_con"]:
                st.markdown(f"<div class='reason-con'>{r}</div>",
                            unsafe_allow_html=True)
            if not v["reasons_con"]:
                st.markdown("<div class='reason-pro'>No red flags detected</div>",
                            unsafe_allow_html=True)

        st.markdown("---")

        # ---- 🎲 Monte Carlo + Kelly ------------------------------------------------
        st.markdown("### 🎲 Monte Carlo — 2,000 simulated futures (30 days)")
        mc = st.columns(6)
        mc[0].metric("P(hit target first)", f"{odds['p_target_first']}%")
        mc[1].metric("P(hit stop first)", f"{odds['p_stop_first']}%")
        mc[2].metric("P(profitable at day 30)", f"{odds['p_profit_end']}%")
        mc[3].metric("95% CVaR / share", f"${odds['cvar95_share']:,.2f}")
        mc[4].metric("Kelly optimal size", f"{kel['kelly_pct']}%" if
                     kel["edge_positive"] else "No edge")
        mc[5].metric("Half-Kelly (use this)", f"{kel['half_kelly_pct']}%" if
                     kel["edge_positive"] else "—")

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

        with st.expander("❓ Monte Carlo & Kelly — how to read this"):
            st.markdown(f"""
2,000 alternative futures simulated with Geometric Brownian Motion calibrated to {tkr}'s own recent drift & volatility.

- **The cone** — dark band = middle 50% of futures, light = 90%. Reality outside the cone = the market changed character.
- **P(target first) vs P(stop first)** — your *simulated win rate* for this exact trade setup.
- **CVaR 95%** — the average loss in the worst 5% of futures. The "how bad is bad" number desks size by.
- **Kelly criterion** — the mathematically optimal fraction of capital: f* = p − (1−p)/RR. Full Kelly is a wild ride; **half-Kelly** keeps ~75% of the growth at half the pain, which is why pros use it. "No edge" = the simulated odds don't justify the trade at all.
""")

        st.markdown("---")

        # ---- ⚡ B-Xtrender -----------------------------------------------------
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
                                   marker=dict(color="#22ff44", size=9)),
                        row=1, col=1)
        figbx.add_trace(go.Scatter(x=sells_bx, y=t3l.loc[sells_bx],
                                   mode="markers", name="Sell turn",
                                   marker=dict(color="#ff5555", size=9)),
                        row=1, col=1)
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
**B-Xtrender** (Bharat Jhunjhunwala, IFTA Journal) — an RSI applied to the *spread between two EMAs*, filtered by a Tillson T3 line. Faster than MACD, cleaner than raw RSI.

- **Short oscillator (top)** — bright = accelerating, dark = decelerating. Dots on the white T3 line = turn signals.
- **Long oscillator (bottom)** — the trend filter. Institutional rule: only take buy turns when it's above zero.
- **MTF alignment** — the same oscillator on weekly bars. Daily signals against the weekly trend are how retail gets chopped.
- **Divergence** — price at new highs while the oscillator makes lower highs = momentum quietly leaving.
- **Event study** — every historical turn on this exact ticker, measured: average forward return and win rate 5/10/20 days later. Trust data, not paint.
""")

        st.markdown("---")

        # ---- 📐 Price structure ---------------------------------------------------
        st.markdown("### 📐 Price structure — Fibonacci, support & resistance")
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
        for lv in sr:
            col = "#22d3ee" if lv["kind"] == "resistance" else "#a78bfa"
            figp.add_hline(y=lv["price"], line_width=2, line_color=col,
                           opacity=.7,
                           annotation_text=f"{lv['kind'].upper()} "
                                           f"${lv['price']:,.2f} "
                                           f"({lv['touches']} touches)",
                           annotation_font_color=col,
                           annotation_font_size=10)
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
        figp.update_layout(height=580, xaxis_rangeslider_visible=False,
                           margin=dict(l=10, r=10, t=30, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figp, use_container_width=True)

        swing_dir = "upswing" if fib["up_swing"] else "downswing"
        with st.expander("❓ Fibonacci + support/resistance — how to use"):
            st.markdown(f"""
Dominant swing detected: **{swing_dir}** from ${fib['swing_low']:,.2f} to ${fib['swing_high']:,.2f}.

- **Fibonacci retracements** — how much of a move typically gets "given back": 0.382/0.5 = shallow (strong trend), **0.618 = the golden pocket** where healthy pullbacks end, 0.786 = last defence.
- **Support/resistance** (cyan/purple lines) — price zones where swing pivots *clustered*; the touch count shows how battle-tested each level is. Fib level + S/R level + your stop all in one zone = a level that actually matters.
""")

        st.markdown("---")

        # ---- 📅 Seasonality + 🏦 Fundamentals (from the feed) ------------------
        colS, colF = st.columns([1.3, 1])
        with colS:
            st.markdown("### 📅 Seasonality — Detrick-style stats")
            seas = monthly_seasonality(fetch_history(tkr, period="10y"))
            if len(seas):
                import datetime as _dt
                cur_m = _dt.date.today().strftime("%b")
                def _hl(row):
                    return ["background-color: rgba(16,185,129,.18)"
                            if row.name == cur_m else "" for _ in row]
                st.dataframe(seas.style.apply(_hl, axis=1)
                             .background_gradient(subset=["avg return %"],
                                                  cmap="RdYlGn"),
                             use_container_width=True, height=460)
                if cur_m in seas.index:
                    r = seas.loc[cur_m]
                    st.caption(f"**{cur_m} historically:** up "
                               f"{r['win rate %']:.0f}% of years, average "
                               f"{r['avg return %']:+.2f}% "
                               f"({int(r['years'])} years of data).")
            else:
                st.info("Not enough history for seasonality stats.")
        with colF:
            st.markdown("### 🏦 Fundamentals check")
            st.caption("\"Price follows growth & margin & free cash flow\"")
            fund = fundamental_snapshot(tkr)
            if fund and fund.get("quality_score"):
                f1, f2 = st.columns(2)
                rg = fund.get("revenue_growth")
                gm = fund.get("gross_margin")
                om = fund.get("op_margin")
                fy = fund.get("fcf_yield")
                pe = fund.get("fwd_pe")
                f1.metric("Revenue growth (yoy)",
                          f"{rg*100:.1f}%" if rg is not None else "—")
                f2.metric("Gross margin",
                          f"{gm*100:.1f}%" if gm is not None else "—")
                f1.metric("Operating margin",
                          f"{om*100:.1f}%" if om is not None else "—")
                f2.metric("FCF yield",
                          f"{fy*100:.1f}%" if fy is not None else "—")
                f1.metric("Forward P/E",
                          f"{pe:.1f}" if pe else "—")
                f2.metric("Quality score", fund["quality_score"])
            else:
                st.info("Fundamentals unavailable for this ticker (ETFs "
                        "have none; some names return partial data).")
            with st.expander("❓ What are these & the thresholds?"):
                st.markdown("""
The 4-point quality check (1 point each): revenue growth ≥ 10%, gross margin ≥ 40%, operating margin ≥ 15%, FCF yield ≥ 3%. Quality growth compounders tend to score 3-4; melting ice cubes and story-stocks score 0-1. A great technical setup on a 0/4 business deserves a smaller size and a shorter leash. Seasonality: the highlighted row is the current month — a headwind or tailwind stat, never a signal by itself.
""")

        st.markdown("---")

        # ---- 🧠 Model breakdown
        st.markdown("### 🧠 Model breakdown")
        last = comp.iloc[-1]
        sub = last[["trend", "momentum", "bxtrender", "macd", "rsi",
                    "meanrev", "volume"]]
        sub_df = pd.DataFrame(
            {"model": [str(k) for k in sub.index],
             "score": [float(xv) for xv in sub.values]}).set_index("model")
        st.bar_chart(sub_df, height=220)
        with st.expander("❓ What does each model measure?"):
            st.markdown("""
- **trend** — price vs 50 & 200-day averages + golden/death cross. The big picture.
- **momentum** — the academic "12-1" factor: past-year performance excluding the last month. Winners keep winning.
- **bxtrender** — double-smoothed momentum (RSI of an EMA spread, T3-filtered). Less lag than MACD.
- **macd** — momentum of momentum; catches acceleration early.
- **rsi** — which side is dominant right now (above/below 50), fading extremes.
- **meanrev** — Bollinger z-score; stretched rubber bands snap back.
- **volume** — do volume surges confirm the move? Moves without volume are suspect.

Bars pointing the same way = quality signal. Bars fighting = chop = usually NO TRADE.
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
            spy_ret3m = None
            try:
                spy = fetch_history("SPY", period="1y")
                spy_ret3m = float(spy["Close"].pct_change(63).iloc[-1] * 100)
            except Exception:
                pass
            rows = []
            for sym, dfr in data.items():
                try:
                    snap = latest_snapshot(dfr)
                    snap["ticker"] = sym
                    if spy_ret3m is not None and snap.get("ret_3m") is not None:
                        snap["rs_vs_spy"] = round(snap["ret_3m"] - spy_ret3m, 1)
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
- **score** — the fused 7-model output, −1 to +1, dampened in vol storms. ≥ +0.25 BUY zone, ≤ −0.25 SELL zone.
- **rs_vs_spy** — 3-month return minus SPY's: is this name actually *beating the market*, or just floating with it? Institutions buy leaders, not laggards.
- **off_52w_high** — distance from the 52-week high. Research says strength near highs (0 to −10%) keeps working; −40% "bargains" usually aren't.
- **atr** — daily range in $; wilder stock = smaller position for the same risk.

Workflow: scan → pick extremes → **🎯 Trade desk** for the full verdict. Never trade off the scan alone.
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
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Equity $", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown + rolling Sharpe
            dd = res.equity / res.equity.cummax() - 1
            roll = res.equity.pct_change().rolling(63)
            rsharpe = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
            cA, cB = st.columns(2)
            with cA:
                figd = go.Figure(go.Scatter(x=dd.index, y=dd * 100,
                                            fill="tozeroy",
                                            line=dict(color="#ef4444"),
                                            name="Drawdown %"))
                figd.update_layout(height=280, title="Drawdown %",
                                   margin=dict(l=10, r=10, t=40, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figd, use_container_width=True)
            with cB:
                figs2 = go.Figure(go.Scatter(x=rsharpe.index, y=rsharpe,
                                             line=dict(color="#22d3ee"),
                                             name="Rolling Sharpe"))
                figs2.add_hline(y=0, line_color="#2a3644")
                figs2.update_layout(height=280, title="Rolling Sharpe (3m)",
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(figs2, use_container_width=True)

            # Monthly returns heatmap
            mrets = res.equity.resample("ME").last().pct_change().dropna()
            if len(mrets) >= 6:
                hm = pd.DataFrame({
                    "year": mrets.index.year,
                    "month": mrets.index.strftime("%b"),
                    "ret": mrets.values * 100})
                order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot = hm.pivot_table(index="year", columns="month",
                                       values="ret").reindex(columns=order)
                fighm = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns,
                    y=[str(y) for y in pivot.index],
                    colorscale="RdYlGn", zmid=0,
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text}", showscale=False))
                fighm.update_layout(height=90 + 45 * len(pivot),
                                    title="Monthly returns % (strategy)",
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(fighm, use_container_width=True)

            with st.expander("❓ What do these metrics & charts mean?"):
                st.markdown("""
- **CAGR** — the "interest rate" your strategy earned. **Sharpe** — return per unit of risk (~1 decent, 2+ = check for bugs 😉). **Sortino** — punishes only downside wiggle. **Max Drawdown** — worst peak-to-valley; the number that decides if you'd *survive* running this.
- **Drawdown chart** — how deep and how *long* the underwater periods were. Long flat red = the psychological killer.
- **Rolling Sharpe** — is the edge steady or was it one lucky quarter?
- **Monthly heatmap** — seasonality and consistency at a glance. A good system is boringly green, not one giant month.
""")

            st.markdown("#### Walk-forward check (4 sequential folds)")
            st.dataframe(walk_forward(df, cfg), use_container_width=True)
            with st.expander("❓ Why walk-forward matters"):
                st.markdown("""
The same rules re-run on 4 separate sequential periods. A real edge shows in most folds; a curve-fit illusion shines in one and dies in the rest. The single best overfitting detector available to a retail quant.
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
            pcr = put_call_ratio(chain)
            near_exp = sorted(chain["expiry"].unique())[0]
            mp = max_pain(chain, near_exp)
            exp_move = None
            if atm_iv:
                t_yrs = float(ts["dte"].iloc[0]) / 365.0
                exp_move = 0.8 * spot * (atm_iv / 100) * np.sqrt(t_yrs)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Spot", f"${spot:,.2f}")
            m2.metric("ATM IV", f"{atm_iv:.1f}%" if atm_iv else "—")
            m3.metric("Expected move", f"±${exp_move:,.2f}" if exp_move else "—")
            m4.metric("Skew (95P−105C)",
                      f"{skew:+.1f}" if skew is not None else "—")
            m5.metric("Put/Call OI ratio", pcr if pcr is not None else "—")
            m6.metric(f"Max pain ({near_exp[5:]})",
                      f"${mp:,.0f}" if mp else "—")

            with st.expander("❓ What do these numbers mean?"):
                st.markdown(f"""
- **ATM IV** — the market's own volatility forecast, backed by real money.
- **Expected move** — Black-Scholes straddle approximation (≈ 0.8 × price × IV × √time): how far the market prices {opt_tkr} to move by the nearest expiry. Targets outside it = betting against the options market.
- **Skew** — downside puts vs upside calls. Large positive = crash insurance in demand.
- **Put/Call OI ratio** — total put open interest ÷ calls. Extremes are contrarian: >1.2 = fear (often near bottoms), <0.6 = greed.
- **Max pain** — the strike where option *holders* lose the most at expiry. Price often gravitates toward it into expiration week (dealers hedging), a real but modest effect.
""")


            # ---- 🐋 Dealer gamma & whale flow ---------------------------------
            st.markdown("### 🐋 Dealer gamma exposure (GEX) & whale flow")
            prof = gex_profile(chain, spot)
            summ = gex_summary(prof, spot)
            if summ:
                gm1, gm2, gm3, gm4, gm5 = st.columns(5)
                gm1.metric("Net GEX", f"${summ['net_gex_m']:,.0f}M")
                gm2.metric("Regime", summ["regime"].split(" ")[0] + " " +
                           summ["regime"].split(" ")[1])
                gm3.metric("Call wall", f"${summ['call_wall']:,.0f}")
                gm4.metric("Put wall", f"${summ['put_wall']:,.0f}")
                gm5.metric("Gamma flip",
                           f"${summ['flip']:,.0f}" if summ["flip"] else "—",
                           delta=summ["spot_vs_flip"], delta_color="off")

                colors_gex = np.where(prof["gex_m"] >= 0, "#10b981", "#ef4444")
                figg = go.Figure(go.Bar(x=prof["strike"], y=prof["gex_m"],
                                        marker_color=colors_gex,
                                        name="Net GEX $M"))
                figg.add_vline(x=spot, line_dash="dot", line_color="#e6edf3",
                               annotation_text=f"spot ${spot:,.0f}")
                if summ["flip"]:
                    figg.add_vline(x=summ["flip"], line_dash="dash",
                                   line_color="#f59e0b",
                                   annotation_text="flip")
                figg.update_layout(height=380, xaxis_title="Strike",
                                   yaxis_title="Net GEX ($M per 1% move)",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figg, use_container_width=True)

                with st.expander("❓ GEX, walls & the flip — the flow-trader playbook"):
                    st.markdown("""
Dealers who sell options must hedge them by trading the stock — mechanically, without opinion. **GEX** estimates how much:

- **Positive net GEX (🧲 pinning)** — dealers buy dips and sell rips. Price gets *magnetized* between the walls; ranges, boring days, failed breakouts.
- **Negative net GEX (⛽ vol fuel)** — dealers must sell INTO drops and buy INTO rallies, amplifying every move. Crashes and face-rippers live here.
- **Call wall** — the strike with peak positive gamma; acts as resistance/pin magnet into expiry.
- **Put wall** — peak negative gamma; the air-pocket level where support turns to acceleration.
- **Gamma flip** — the spot level where the regime changes. Above it = stable zone, below = unstable. Watch what happens when price approaches it.

This is the same math behind the paid flow dashboards — computed from public open interest.
""")

            st.markdown("### 🔥 Unusual activity (fresh positioning)")
            uf = unusual_flow(chain)
            if len(uf):
                st.dataframe(uf, use_container_width=True, hide_index=True,
                             height=380)
                with st.expander("❓ How to read the flow table"):
                    st.markdown("""
- **vol/oi > 1** 🔥 — more contracts traded today than existed before: someone is OPENING a fresh position, not closing an old one. That's the signature flow-traders hunt.
- **premium $** — the actual money behind it. A million in premium is conviction; $5K is noise.
- Caveats the paid services rarely mention: you can't see if it's a buy or a sell from delayed data, and big prints are often hedges or spread legs. Treat as *context*, never as a signal alone.
""")
            else:
                st.info("No contracts with meaningful volume right now "
                        "(quiet session or delayed data).")

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
Every point = one option's implied vol (strike × expiry). Pros look for: the **smile/smirk** across strikes (steep left wing = crash fear), the **term slope** across expiries (inverted = near-term event fear), and **bumps** at specific expiries = event risk priced exactly there.
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
                figs.add_vline(x=spot, line_dash="dot", annotation_text="spot")
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
- **Delta** — $ change per $1 stock move; also ≈ probability of expiring in the money.
- **Gamma** — how fast delta changes; high gamma = behaviour flips fast near the strike.
- **Vega** — $ change per IV point; post-earnings IV crush is vega bleeding.
- **Theta** — $ lost per day to time decay; the rent you pay to hold.

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
Buying "$1,000 of each stock" gives a calm stock and a wild stock totally different risk. Sizing by **risk** (account % ÷ stop distance) equalises it. With a small account, ruin-avoidance *is* the strategy — a 50% drawdown needs +100% just to break even. For the mathematically optimal size per trade, see the **Kelly** number in the Trade desk's Monte Carlo section — and then use half of it, like the pros.
""")
