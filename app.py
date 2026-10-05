"""QuantSignal — neon-terminal quant desk for US stocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quant.advanced import ewma_vol, kelly, regime_quadrant, support_resistance
from quant.analyst import morning_briefing, ticker_news
from quant.autotrader import (bot_equity, bot_tick, default_state, load_bot,
                              save_bot)
from quant.anomalies import ANOMALY_INFO
from quant.master import run_master
from quant.backtest import BTConfig, run_backtest, walk_forward
from quant.bxlab import parameter_sweep, state_probabilities
from quant.bxtrender import (bxtrender, detect_divergence, event_study,
                             weekly_alignment)
from quant.data import DEFAULT_UNIVERSE, fetch_history, fetch_many
from quant.events import equity_risk_gauge, fetch_macro_markets
from quant.flow import gex_profile, gex_summary, unusual_flow
from quant.scanner import RISK_PROFILES, scan_setups
from quant.seasonality import fundamental_snapshot, monthly_seasonality
from quant.levels import fib_levels, hurst
from quant.lse import fetch_candles_lse, probe
from quant.live import live_quote, market_status, patch_live_bar
from quant.journal import (journal_from_csv, journal_to_csv, load_journal,
                           mark_to_market, record_plan, save_journal)
from quant.garch_pairs import garch_forecast, pairs_analysis
from quant.montecarlo import cone, simulate, trade_odds
from quant.playbook import build_playbook
from quant.portfolio import build_prices, optimize
from quant.runner import run_machine
from quant.risk import (correlation_heat, kelly_ladder, portfolio_var,
                        position_risk, risk_of_ruin)
from quant.rl_lab import (MDM_COLORS, MDM_STYLES, market_dynamics,
                          prudex_scores, train_agent)
from quant.opt_edge import (iv_richness, move_vs_model, realized_vol,
                            suggest_structure, vrp)
from quant.options import (atm_term_structure, bs_greeks, build_surface,
                           fetch_chains, max_pain, put_call_ratio, skew_25)
from quant.signals import BUY_TH, SELL_TH, atr, composite, latest_snapshot
from quant.timeframes import TF_LABELS, fetch_tf, tf_meta
from quant.validation import (bootstrap_cagr, deflated_sharpe,
                              haircut_pvalue, permutation_test)
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

/* ---- terminal touches ---- */
.tape {
  overflow: hidden; white-space: nowrap; border-top: 1px solid #1f2a36;
  border-bottom: 1px solid #1f2a36; background: #0a0e13;
  font-family: 'JetBrains Mono', monospace; font-size: .85rem;
  padding: 6px 0; margin: 4px 0 10px 0;
}
.tape-inner { display: inline-block; animation: scroll 40s linear infinite; }
@keyframes scroll { 0% {transform: translateX(0)} 100% {transform: translateX(-50%)} }
.tape .up { color: #10b981; } .tape .dn { color: #ef4444; }
.tape .amber { color: #ffb000; font-weight: 700; }
div[data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; }
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

# ---------------------------------------------------------------------------
# LIVE mode — one press arms a panel; it then self-refreshes on this pulse
# ---------------------------------------------------------------------------
ms = market_status()
lc1, lc2, lc3 = st.columns([1.1, 1, 3])
live_on = lc1.toggle("🔴 LIVE mode", value=False, key="live_on",
                     help="Armed panels (watchlist, Trade desk, Runner, "
                          "Track record) auto-refresh with live prices.")
live_every = lc2.selectbox("Pulse", [30, 60, 120], index=1,
                           key="live_every",
                           format_func=lambda s: f"every {s}s")
wl_input = lc2.text_input("Watchlist", value="SPY, QQQ, ^VIX",
                          key="wl_input", label_visibility="collapsed",
                          placeholder="Watchlist: SPY, QQQ, ^VIX ...")
st.session_state["watchlist"] = [t.strip().upper() for t in
                                 wl_input.split(",") if t.strip()][:6]
LIVE_EVERY = live_every if live_on else None
lc3.markdown(f"<div class='regime-badge'>{ms['emoji']} {ms['label']} · "
             f"{ms['detail']} · {ms['et_time']}</div>",
             unsafe_allow_html=True)
if live_on:
    st.caption("🔴 Live uses the free Yahoo feed on a shared IP — if data "
               "briefly shows '—' or a rate-limit note, that's throttling, "
               "not a crash. It recovers on the next pulse; a slower pulse "
               "helps.")


@st.fragment(run_every=LIVE_EVERY)
def _watchlist_strip():
    from datetime import datetime as _dt
    syms = st.session_state.get("watchlist", ["SPY", "QQQ", "^VIX"])[:6]
    armed = st.session_state.get("desk_params", {}).get("tkr")
    if armed and armed not in syms:
        syms.append(armed)
    cols = st.columns(len(syms) + 1)
    for col, s in zip(cols, syms):
        q = live_quote(s)
        if q:
            col.metric(s.replace("^", ""), f"{q['price']:,.2f}",
                       delta=f"{q['chg_pct']:+.2f}%")
        else:
            col.metric(s.replace("^", ""), "—")
    cols[-1].caption(f"{'🔴 LIVE' if LIVE_EVERY else '⏸ static'} · "
                     f"updated {_dt.now().strftime('%H:%M:%S')}")
    tape_syms = list(dict.fromkeys(
        st.session_state.get("watchlist", []) +
        ["SPY", "QQQ", "^VIX", "AAPL", "NVDA", "MSFT", "TSLA", "GLD"]))[:10]
    parts = []
    for ts_ in tape_syms:
        tq = live_quote(ts_)
        if tq:
            cls = "up" if tq["chg_pct"] >= 0 else "dn"
            arrow = "▲" if tq["chg_pct"] >= 0 else "▼"
            parts.append(f"<span class='amber'>{ts_.replace('^','')}</span> "
                         f"<span class='{cls}'>{tq['price']:,.2f} {arrow}"
                         f"{abs(tq['chg_pct']):.2f}%</span>")
    if parts:
        line = " &nbsp;·&nbsp; ".join(parts)
        st.markdown(f"<div class='tape'><div class='tape-inner'>{line}"
                    f" &nbsp;·&nbsp; {line}</div></div>",
                    unsafe_allow_html=True)


_watchlist_strip()

st.session_state.setdefault("memory", {})


def _remember(section: str, payload: dict):
    st.session_state["memory"][section] = payload


def _memory_chips():
    mem = st.session_state.get("memory", {})
    if not mem:
        return
    bits = []
    d = mem.get("desk")
    if d:
        bits.append(f"🎯 {d['ticker']}: {d['verdict']} ({d['conviction']})")
    o = mem.get("options")
    if o:
        bits.append(f"🌋 {o['ticker']}: {o.get('vol_state','')[:12]}…")
    e = mem.get("events")
    if e:
        bits.append(f"🌐 macro: {e['label']}")
    bt = mem.get("backtest")
    if bt:
        bits.append(f"🧪 {bt['ticker']}: {bt['mode']} Sharpe {bt['sharpe']}")
    if bits:
        st.caption("🧠 **Session memory (tabs share this):** " +
                   "  ·  ".join(bits))


_memory_chips()

# ---------------------------------------------------------------------------
# ⌨️ Terminal command line — type like a Bloomberg jockey
# ---------------------------------------------------------------------------
cmd = st.text_input("⌨️", placeholder="Command line — try: NVDA GO · AAPL PB · "
                    "TSLA VOL · SCAN", key="cmdline",
                    label_visibility="collapsed")
if cmd:
    parts = cmd.strip().upper().split()
    try:
        if parts[-1] == "GO" and len(parts) == 2:
            _t = parts[0]
            _q = live_quote(_t)
            _d = fetch_history(_t, period="1y")
            if _q and len(_d) > 220:
                _v = analyze(_d)
                cg = st.columns(5)
                cg[0].metric(_t, f"${_q['price']:,.2f}",
                             f"{_q['chg_pct']:+.2f}%")
                cg[1].metric("Verdict", _v["verdict"])
                cg[2].metric("Conviction", _v["conviction"])
                cg[3].metric("Score", f"{_v['score']:+.2f}")
                cg[4].metric("Signal Sharpe", _v["sharpe"])
            else:
                st.warning(f"{_t}: no data (throttled or bad ticker)")
        elif parts[-1] == "PB" and len(parts) == 2:
            _d = fetch_history(parts[0], period="1y")
            if len(_d) > 220:
                _pb = build_playbook(_d)
                st.info(f"**{parts[0]} · {_pb['urgency']}** — "
                        f"{_pb['instruction']}")
        elif parts[-1] == "VOL" and len(parts) == 2:
            _d = fetch_history(parts[0], period="2y")
            _e = ewma_vol(_d); _g = garch_forecast(_d)
            st.info(f"**{parts[0]} vol** — EWMA {_e['sigma_annual_pct']}% ann "
                    f"(±${_e['expected_move_1d']}/day)"
                    + (f" · GARCH {_g['sigma_annual_pct']}%" if _g else ""))
        elif parts[0] == "SCAN":
            st.info("→ open **🧬 Alpha engine** and hit **☀️ Scan today's "
                    "setups** — the full ranked list lives there.")
        else:
            st.caption("Commands: `TICKER GO` quote+verdict · `TICKER PB` "
                       "playbook · `TICKER VOL` volatility · `SCAN`")
    except Exception:
        st.warning("Command failed (data throttled?) — try again.")

with st.expander("🔌 Data sources — free LSE API (experimental, kills the "
                 "Yahoo rate limit)"):
    st.markdown("**London Strategic Edge** (londonstrategicedge.com) offers a "
                "free-key API: candles, options chains **with greeks**, "
                "options flow, macro. A working key here = no more shared-IP "
                "throttling. Get a free key on their site, then:")
    l1, l2 = st.columns([2, 1])
    st.session_state["lse_key"] = l1.text_input(
        "LSE API key", value=st.session_state.get("lse_key", ""),
        type="password", key="lse_key_in")
    lse_base = l2.text_input("Base URL (optional — from their docs)",
                             key="lse_base",
                             placeholder="https://api.londonstrategicedge.com")
    cA, cB = st.columns(2)
    if cA.button("🧪 Test connection (probes endpoint patterns)",
                 key="lse_probe"):
        if not st.session_state["lse_key"]:
            st.warning("Paste your free key first.")
        else:
            with st.spinner("Probing candidate endpoints…"):
                res = probe(st.session_state["lse_key"],
                            extra_base=lse_base or None)
            hits = [r for r in res if r["ok"]]
            if hits:
                st.success(f"✅ CONNECTED — working pattern found "
                           f"({hits[0]['auth']} auth). LSE fetch is now "
                           f"available this session.")
            else:
                st.warning("No candidate pattern responded 200. Open their "
                           "API docs in your browser, copy ONE example "
                           "request URL (the curl line for candles), paste "
                           "it into the Base URL box exactly, and re-test — "
                           "or send it to me/Claude Code and we lock it in "
                           "permanently.")
                st.dataframe(pd.DataFrame(res).head(10),
                             use_container_width=True, hide_index=True)
    if cB.button("📈 Demo: fetch AAPL via LSE", key="lse_demo"):
        d_ = fetch_candles_lse("AAPL", "1d")
        if len(d_):
            st.success(f"LSE returned {len(d_)} AAPL bars "
                       f"({d_.index[0].date()} → {d_.index[-1].date()}) — "
                       f"pipeline works.")
            st.line_chart(d_["Close"].iloc[-252:])
        else:
            st.info("No data — run the connection test first (or the "
                    "pattern needs their docs' exact URL).")
st.write("")

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)",
                     plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Inter", color="#e6edf3"))

(tab_analyst, tab_bot, tab_master, tab_journal, tab_runner, tab_desk,
 tab_screener, tab_backtest, tab_options, tab_pp, tab_events, tab_rl,
 tab_sizing) = st.tabs(
    ["🤵 Analyst", "🦾 AutoTrader", "🧬 Alpha engine", "📒 Track record",
     "⚙️ Runner", "🎯 Trade desk", "🔍 Screener", "🧪 Backtest",
     "🌋 Options / IV surface", "⚖️ Portfolio & Pairs", "🌐 Event radar",
     "🤖 RL lab", "💰 Position size"]
)

SIGNAL_COLORS = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#8a8a8a"}
VERDICT_CLASS = {"LONG": "v-long", "SHORT": "v-short", "NO TRADE": "v-none"}
VERDICT_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "NO TRADE": "⚪"}
FIB_COLORS = {"0": "#6ee7b7", "0.236": "#34d399", "0.382": "#fbbf24",
              "0.5": "#f59e0b", "0.618": "#f97316", "0.786": "#ef4444",
              "1": "#dc2626"}



# ===========================================================================
# -1. THE ANALYST — runs the whole desk, writes you the note
# ===========================================================================
with tab_analyst:
    st.subheader("🤵 The Analyst — your desk, run for you")
    st.caption("One button executes the entire operation: market regime, "
               "macro odds, all 50 tickers through the setup gates, every "
               "open position through the playbook, book-level risk, the "
               "news touching your names, and a statistical self-audit of "
               "the live track record. Every sentence is computed, never "
               "imagined.")
    a1, a2, a3 = st.columns(3)
    an_acct = a1.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="an_acct")
    an_prof = a2.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                           index=1, key="an_prof")
    an_news = a3.toggle("Include news", value=True, key="an_news")

    if st.button("☕ Run my morning", type="primary", key="an_run"):
        _ap = RISK_PROFILES[an_prof]
        prog = st.progress(0, text="Market regime…")
        spy = fetch_history("SPY", period="2y")
        prog.progress(15, text="Macro event odds…")
        gauge = None
        try:
            ev = fetch_macro_markets()
            gauge = equity_risk_gauge(ev) if len(ev) else None
        except Exception:
            pass
        prog.progress(30, text="Scanning the universe…")
        data = fetch_many(tuple(DEFAULT_UNIVERSE), period="2y")
        prog.progress(60, text="Marking your book…")
        jj = load_journal()
        blotter = None
        if jj["positions"]:
            mtm = mark_to_market(
                jj, lambda t: patch_live_bar(fetch_history(t, period="1y"), t))
            save_journal(jj)
            blotter = mtm.get("blotter")
        prog.progress(80, text="Writing the note…")
        brief = morning_briefing(spy, data, blotter, float(an_acct),
                                 _ap["risk_pct"], gauge)
        prog.progress(100); prog.empty()
        st.session_state["briefing"] = brief

    if "briefing" in st.session_state:
        brief = st.session_state["briefing"]
        st.markdown(f"""<div style="border:1px solid #ffb000;border-radius:14px;
            padding:14px 20px;background:rgba(255,176,0,.05);
            font-family:'JetBrains Mono',monospace">
            <span style="color:#ffb000;font-weight:800">MORNING NOTE</span>
            · {brief['stamp']} · profile: {st.session_state.get('an_prof','')}
            </div>""", unsafe_allow_html=True)
        st.write("")

        sec_defs = [("🌍 Market", "market"), ("💼 Your book", "book"),
                    ("☀️ Today's trades", "setups"),
                    ("👀 Watch list", "watch"),
                    ("🔬 System self-audit", "audit")]
        for title, key in sec_defs:
            lines = brief.get(key) or []
            if not lines:
                continue
            st.markdown(f"#### {title}")
            for ln in lines:
                cls = "reason-con" if ("⚠️" in ln or "🔴" in ln) else "reason-pro"
                st.markdown(f"<div class='{cls}'>{ln}</div>",
                            unsafe_allow_html=True)
            st.write("")

        if len(brief["setups_table"]):
            with st.expander("📋 Full setups table (sized to your profile)"):
                st.dataframe(brief["setups_table"],
                             use_container_width=True, hide_index=True)

        if st.session_state.get("an_news") and brief["news_tickers"]:
            st.markdown("#### 📰 News on your names")
            nws = ticker_news(tuple(brief["news_tickers"]))
            if len(nws):
                for _, r in nws.iterrows():
                    st.markdown(f"<div class='reason-pro'>"
                                f"<b style='color:#ffb000'>{r['ticker']}</b> — "
                                f"<a href='{r['url']}' target='_blank' "
                                f"style='color:#e6edf3'>{r['headline']}</a> "
                                f"<span style='color:#8b98a5;font-size:.8rem'>"
                                f"({r['source']})</span></div>",
                                unsafe_allow_html=True)
            else:
                st.caption("News feed empty or throttled right now.")

        with st.expander("❓ What the Analyst is (and deliberately isn't)"):
            st.markdown("""
The Analyst is an **orchestrator**: it runs every engine on the site in sequence and converts the numbers into sentences by fixed rules. If it says your position is +1.4R and the stop should move — that came from the playbook math, checkable in the 🎯 desk. If it says the live edge isn't statistically proven yet — that's the bootstrap CI on your actual recorded trades.

What it deliberately **isn't**: a language model improvising opinions. In trading, confident-sounding text without verified computation behind it is how accounts die. Everything here is auditable — click into any tab and find the number behind the sentence. If we ever add a conversational layer, it will only be allowed to narrate what these engines computed.

**The routine**: ☕ every morning before the open (16:30 your time). Read the note top to bottom — market, your book's instructions, today's trades, the watch list. Execute through the Trade desk, record to the Track record. The self-audit keeps score of whether the whole thing is actually working — and it will tell you honestly if it isn't.
""")


# ===========================================================================
# -0.5 AUTOTRADER — the bot that lives here
# ===========================================================================
with tab_bot:
    st.subheader("🦾 AutoTrader — the in-website trading bot (paper)")
    st.caption("Give it a mandate once, flip it ON. It scans, stages orders "
               "for the open, fills them, banks profits at +1R/+2R, trails, "
               "stops out — and narrates every decision. While the site is "
               "closed it sleeps; on your next visit it CATCHES UP bar-by-"
               "bar, executing exactly what the rules dictated. Real "
               "prices, paper money — building the record that decides if "
               "this logic ever touches a real broker.")

    bot = load_bot()

    bc1, bc2, bc3, bc4 = st.columns([1, 1, 1, 1.6])
    with bc1:
        enabled = st.toggle("🦾 BOT ACTIVE", value=bot["enabled"],
                            key="bot_on")
    bot_prof = bc2.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                             index=1, key="bot_prof")
    bot_acct = bc3.number_input("Paper capital $", 1000, 1_000_000,
                                int(bot.get("start_equity", 5000)),
                                step=1000, key="bot_acct")
    _bp = RISK_PROFILES[bot_prof]
    bc4.caption(f"Mandate: {_bp['risk_pct']}%/trade · max "
                f"{_bp['max_pos']} positions · default 50-name universe · "
                f"entries only on 🟢 5/5-gate setups · scale ⅓ at +1R and "
                f"+2R · 2.5×ATR trail.")

    colA, colB = st.columns([1, 4])
    if colA.button("🔄 Reset bot (wipe paper history)", key="bot_reset"):
        bot = default_state(float(bot_acct))
        save_bot(bot)
        st.success("Bot reset — fresh paper account.")

    if enabled != bot["enabled"]:
        bot["enabled"] = enabled
        if enabled and not bot["created"]:
            bot = default_state(float(bot_acct))
            bot["enabled"] = True
            from datetime import datetime as _dtnow, timezone as _tz
            bot["created"] = _dtnow.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
        bot["mandate"]["risk_pct"] = _bp["risk_pct"]
        bot["mandate"]["max_positions"] = _bp["max_pos"]
        save_bot(bot)

    if bot["enabled"]:
        @st.fragment(run_every=LIVE_EVERY)
        def _bot_live():
            b = load_bot()
            with st.spinner("Bot tick — catching up & deciding…"):
                data_b = fetch_many(tuple(DEFAULT_UNIVERSE), period="2y")
                spy_b = fetch_history("SPY", period="2y")
                if spy_b.empty or not data_b:
                    st.warning("Data throttled — bot will retry next pulse.")
                    return
                b = bot_tick(b, data_b, spy_b)
                save_bot(b)
                eq = bot_equity(b, data_b)

            m = st.columns(6)
            m[0].metric("Paper equity", f"${eq['equity']:,.0f}",
                        f"{eq['return_pct']:+.2f}%")
            m[1].metric("Cash", f"${eq['cash']:,.0f}")
            m[2].metric("Open positions", len(eq["open_table"]))
            m[3].metric("Closed deals", len(eq["closed_table"]))
            m[4].metric("Wins", eq["n_wins"])
            m[5].metric("Realized P&L", f"${eq['realized']:+,.0f}")
            st.caption(f"Bot since {b.get('created','—')} · last processed "
                       f"session: {b.get('last_tick','—')} · "
                       f"{'🔴 watching live' if LIVE_EVERY else '⏸ tick on load only (turn LIVE on to watch it work)'}")

            if len(eq["open_table"]):
                st.markdown("**📈 Bot holdings (live marks):**")
                st.dataframe(eq["open_table"], use_container_width=True,
                             hide_index=True)
            if b["pending"]:
                st.markdown("**📋 Orders staged for next open:**")
                for od in b["pending"]:
                    st.markdown(f"<div class='reason-pro'>📋 BUY "
                                f"{od['shares']} {od['ticker']} at open — "
                                f"{od['why']}</div>", unsafe_allow_html=True)

            st.markdown("**🗣️ Bot decision feed (newest first):**")
            for L in list(reversed(b["log"]))[:12]:
                good = any(k in L["action"] for k in ("BUY", "SCALE", "ORDER"))
                st.markdown(f"<div class='{'reason-pro' if good else 'reason-con'}'"
                            f" style=\"font-family:'JetBrains Mono',monospace\">"
                            f"{L['action']} <b>{L['ticker']}</b> · "
                            f"{L['shares']} sh @ ${L['price']:,.2f} · "
                            f"{L['date']}<br>→ {L['why']}</div>",
                            unsafe_allow_html=True)

            if len(eq["closed_table"]):
                st.markdown("**🧾 Bot deal ledger:**")
                st.dataframe(eq["closed_table"].iloc[::-1],
                             use_container_width=True, hide_index=True,
                             height=280)
        _bot_live()
    else:
        st.info("Bot is OFF. Flip the toggle to give it the mandate — it "
                "starts scanning immediately and stages its first orders "
                "for the next open.")

    with st.expander("❓ How the bot works & why it's trustworthy"):
        st.markdown("""
**The loop**: every tick it (1) fills yesterday's staged orders at today's real open, (2) walks every open position through the bar — scale ⅓ at +1R (stop→breakeven), scale ⅓ at +2R (stop→+1R), trail the rest at 2.5×ATR, stop out where price actually touched — then (3) scans the universe and stages tomorrow's entries from 5/5-gate setups only.

**The catch-up trick**: the rules are bar-mechanical with next-open execution, so replaying missed days is *identical* to having watched them live. The bot being "asleep" while the site is closed costs nothing.

**Why paper first is non-negotiable**: this ledger becomes the statistical evidence (the 🔬 Validation Lab can audit it) that decides whether the logic deserves a real broker API. Bots don't get promoted on enthusiasm — they get promoted on a verified record. ⚠️ The free server wipes files on redeploy; like the journal, the bot's memory resets then — one more reason its early life is paper-only.
""")

# ===========================================================================
# 0. ALPHA ENGINE — the master algorithm
# ===========================================================================
with tab_master:
    st.subheader("🧬 The master algorithm — one answer: what to do now")
    with st.expander("❓ How this machine works (read once — 60 seconds)"):
        st.markdown("""
Four filters in a row; a stock must survive ALL of them to reach your plan:

**1️⃣ MARKET GATE** — is the overall market (SPY) healthy? Decides how much of your account may deploy at all (Bull·Calm = 100% → Bear·Storm = 15%). *Don't fight the tape.*

**2️⃣ CROSS-SECTIONAL RANK** — all 50 stocks scored on 6 published anomalies (momentum, 52-week high, anti-lottery, low-vol, low-beta, reversal) and ranked against each other. Top decile advances; bottom 5 become the avoid list.

**3️⃣ TIME-SERIES VERDICT** — survivors face the full 7-model engine (trend, B-X, MACD…) + a quick backtest of the signal on that exact ticker. Only LONG with real conviction passes. **This is the strictest gate — most days most names fail here. An empty plan means the machine is protecting you, not malfunctioning.**

**4️⃣ RISK SIZING** — equal risk per position, total portfolio heat capped. Then the plan prints: ticker, shares, entry, stop, target.

**☀️ Today's setups** below is the FAST lane: the same entry gates, no backtests — what's tradeable *right now*, in ~10 seconds.
""")

    prof_c1, prof_c2 = st.columns([1.2, 3])
    ma_profile = prof_c1.selectbox("Risk profile", list(RISK_PROFILES.keys()),
                                   index=1, key="ma_prof")
    _pp = RISK_PROFILES[ma_profile]
    prof_c2.caption(f"**{ma_profile}**: {_pp['risk_pct']}% risk/position · "
                    f"up to {_pp['max_pos']} positions · "
                    f"{_pp['heat_cap']}% total heat · conviction ≥ "
                    f"{_pp['conviction_min']}. Aggressive ≈ 2-3× the P&L "
                    f"swing of Conservative — in BOTH directions. The 🛡️ "
                    f"risk-of-ruin stats in Backtest show what your profile "
                    f"survives.")

    c1, c2, c3 = st.columns(3)
    ma_acct = c1.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="ma_acct")
    ma_custom = c2.text_input("Universe (empty = default 50)",
                              placeholder="AAPL, NVDA ...", key="ma_uni")
    ma_aggfill = c3.toggle("Aggressive fill (never return empty)",
                           value=("Aggressive" in ma_profile),
                           help="If the strict gates pass <2 names, take the "
                                "top alpha names at HALF size, tagged as "
                                "lower-confidence.")
    ma_risk = _pp["risk_pct"]
    ma_maxpos = _pp["max_pos"]

    # ---- ☀️ TODAY'S SETUPS — the daily trades scanner --------------------
    if st.button("☀️ Scan today's setups (fast — all data, right now)",
                 key="scan_today"):
        uni_s = tuple(t.strip().upper() for t in ma_custom.split(",")
                      if t.strip()) or tuple(DEFAULT_UNIVERSE)
        with st.spinner(f"Playbook gates on {len(uni_s)} tickers…"):
            data_s = fetch_many(uni_s, period="2y")
            setups = scan_setups(data_s, account=float(ma_acct),
                                 risk_pct=ma_risk)
        if len(setups):
            n_enter = int((setups["urgency"] == "🟢 ENTER").sum())
            n_fast = int((setups["urgency"] == "🟡 FAST").sum())
            st.success(f"**Today: {n_enter} full entries · {n_fast} dip "
                       f"scalps · {len(setups)-n_enter-n_fast} stalking.** "
                       f"Sized at {ma_profile} ({ma_risk}%/trade).")
            st.dataframe(setups, use_container_width=True, hide_index=True)
            st.caption("🟢 ENTER = all 5 gates green, full playbook trade. "
                       "🟡 FAST = RSI2 panic in an uptrend — quick scalp, "
                       "exit on the snap-back. 👀 STALK = one gate away; "
                       "set an alert. Run any name through 🎯 Trade desk "
                       "for the full workup before pulling the trigger.")
        else:
            st.info("**Zero setups today.** The market isn't offering — "
                    "chasing anyway is how edges become donations. "
                    "Tomorrow is another scan.")

    st.markdown("---")

    if st.button("🚀 Run the machine", type="primary", key="ma_run"):
        uni = tuple(t.strip().upper() for t in ma_custom.split(",")
                    if t.strip()) or tuple(DEFAULT_UNIVERSE)
        prog = st.progress(0, text="Downloading universe…")
        data = fetch_many(uni, period="2y")
        prog.progress(30, text="Downloading SPY (market gate)…")
        spy = fetch_history("SPY", period="2y")
        prog.progress(45, text="Ranking cross-sectional anomalies…")
        res = run_master(data, spy, account=float(ma_acct),
                         risk_pct=ma_risk, max_positions=ma_maxpos,
                         heat_cap_pct=_pp["heat_cap"],
                         conviction_min=_pp["conviction_min"],
                         aggressive_fill=ma_aggfill)
        prog.progress(100, text="Done")
        prog.empty()
        st.session_state["master_res"] = res
        st.session_state["master_acct"] = float(ma_acct)

    if "master_res" in st.session_state:
        res = st.session_state["master_res"]
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
            if res["plan"]["action"].str.contains("½").any():
                st.caption("*½size = aggressive fill: strong alpha rank but "
                           "the verdict engine wasn't fully convinced — "
                           "taken at HALF risk so a good rank can't hurt "
                           "you at full weight.")
            st.success(f"**Do this:** open the {len(res['plan'])} position(s) "
                       f"above with the exact share counts, place the stops "
                       f"immediately, keep ${res['cash']:,.0f} in cash. "
                       f"If a stop hits — you're out, no negotiating with it.")
        else:
            st.info("**The machine says: do nothing.** No candidate passed all "
                    "four gates (anomaly rank → verdict → regime → risk). "
                    "Cash is a position; the next setup will come to you.")

        if len(res["plan"]):
            if st.button("📒 Record this plan to the track record",
                         key="rec_plan"):
                jj = load_journal()
                jj, n_added = record_plan(jj, res["plan"],
                                          res["regime"]["regime"],
                                          st.session_state.get("master_acct",
                                                               5000.0))
                save_journal(jj)
                st.success(f"Recorded {n_added} position(s) with UTC "
                           f"timestamp, model version and regime stamp. "
                           f"See the 📒 Track record tab.")

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
# 0b. TRACK RECORD — the fund fact sheet
# ===========================================================================
with tab_journal:
    st.subheader("📒 Track record — verified paper-trading journal")
    st.caption("Append-only journal: every plan stamped with UTC time, model "
               "version and market regime at entry. Stops & targets enforced "
               "mechanically on real daily bars (first touch; stop wins ties). "
               "This is how a strategy earns trust before real money.")

    @st.fragment(run_every=LIVE_EVERY)
    def _journal_live():
        jj = load_journal()
        n_pos = len(jj["positions"])

        ctop = st.columns([1, 1, 2])
        if n_pos:
            ctop[0].download_button("⬇️ Export journal (CSV)",
                                    journal_to_csv(jj), "quantsignal_journal.csv",
                                    "text/csv")
        up = ctop[1].file_uploader("Restore from CSV", type="csv",
                                   label_visibility="collapsed")
        if up is not None:
            jj = journal_from_csv(up.getvalue().decode())
            save_journal(jj)
            st.success(f"Journal restored — {len(jj['positions'])} positions.")
            n_pos = len(jj["positions"])
        ctop[2].info("⚠️ Free hosting wipes local files on redeploy — export "
                     "after every session. The CSV is your custody.")

        if not n_pos:
            st.info("No positions recorded yet. Run the 🧬 Alpha engine and hit "
                    "**Record this plan** — the clock starts there.")
        else:
            with st.spinner("Marking positions to market…"):
                mtm = mark_to_market(
                jj, lambda t: patch_live_bar(fetch_history(t, period="1y"), t))
                save_journal(jj)   # persist any auto-closed stops/targets

            if mtm.get("data_issues"):
                st.warning("Data issues: " + ", ".join(mtm["data_issues"]))

            s = mtm["stats"]
            head = st.columns(6)
            head[0].metric("Paper equity", f"${s.get('Equity $', 0):,.0f}")
            head[1].metric("Total return", f"{s.get('Total return %', 0)}%")
            head[2].metric("vs SPY", f"{s.get('Alpha vs SPY %', '—')}%"
                           if "Alpha vs SPY %" in s else "—")
            head[3].metric("Live Sharpe", s.get("Sharpe (live)", "—"))
            head[4].metric("Hit rate", f"{s.get('Hit rate %', '—')}%"
                           if "Hit rate %" in s else "—")
            head[5].metric("Open heat", f"${s.get('Heat (risk if all stops hit) $', 0):,.0f}")

            meta1, meta2, meta3 = st.columns(3)
            meta1.caption(f"Inception: {jj['meta'].get('inception', '—')}")
            meta2.caption(f"Model: {jj['meta'].get('version', '—')}")
            meta3.caption(f"Positions: {s.get('Open / Closed', '—')} "
                          f"(open/closed) · Max DD {s.get('Max DD %', '—')}%")

            # ---- 🛡️ RISK DESK — live book risk, the quant way -------------
            open_b = mtm["blotter"][mtm["blotter"]["status"] == "OPEN"]
            if len(open_b):
                st.markdown("### 🛡️ Risk desk — the open book")
                poss = [{"ticker": r["ticker"], "shares": int(r["shares"]),
                         "entry": float(r["entry"]), "stop": float(r["stop"])}
                        for _, r in open_b.iterrows()]
                rets_map = {}
                for p_ in poss:
                    try:
                        h_ = fetch_history(p_["ticker"], period="1y")
                        rets_map[p_["ticker"]] = \
                            h_["Close"].pct_change().dropna()
                    except Exception:
                        continue
                acct_ = float(jj["meta"].get("account", 5000.0))
                pv = portfolio_var(poss, rets_map, acct_)
                ch = correlation_heat(poss, rets_map, acct_)
                rk1, rk2, rk3, rk4, rk5 = st.columns(5)
                if pv:
                    rk1.metric("1-day VaR 95%",
                               f"${pv['VaR_$']:,.0f} ({pv['VaR_%']}%)",
                               help="On a normal bad day (1 in 20), expect "
                                    "to lose up to this much.")
                    rk2.metric("1-day CVaR 95%",
                               f"${pv['CVaR_$']:,.0f} ({pv['CVaR_%']}%)",
                               help="When that bad day happens, this is the "
                                    "AVERAGE loss — the tail number desks "
                                    "size by.")
                    rk3.metric("Gross exposure",
                               f"{pv['gross_exposure_%']}%")
                if ch:
                    rk4.metric("Heat: naive → corr-adj",
                               f"${ch['naive_heat_$']:,.0f} → "
                               f"${ch['corr_adj_heat_$']:,.0f}")
                    rk5.metric("Avg pairwise corr", ch["avg_correlation"],
                               delta="⚠️ crowded book" if ch["warning"]
                               else "diversified",
                               delta_color="inverse" if ch["warning"]
                               else "normal")
                if ch and ch["warning"]:
                    st.warning("Your open positions are highly correlated — "
                               "effectively ONE big trade. A single market "
                               "move can hit every stop together. Consider "
                               "trimming or diversifying sectors.")
                with st.expander("❓ Reading the risk desk"):
                    st.markdown("""
- **VaR 95%** — parametric 1-day Value-at-Risk from the actual covariance of your holdings: "on a normal bad day, expect up to this."
- **CVaR** — the average loss *given* that bad day happened. Desks size by CVaR, not VaR, because tails are where accounts die.
- **Correlation-adjusted heat** — summing per-position risk pretends positions are independent. When names move together (avg corr > 0.6), your true worst case approaches the naive sum — the diversification you think you have is an illusion. The gap between the numbers = your real diversification benefit.
""")

            if len(mtm["equity"]) > 1:
                fige = go.Figure()
                fige.add_trace(go.Scatter(x=mtm["equity"].index, y=mtm["equity"],
                                          name="Portfolio",
                                          line=dict(color="#10b981", width=2)))
                if len(mtm["bench"]) > 1:
                    fige.add_trace(go.Scatter(x=mtm["bench"].index, y=mtm["bench"],
                                              name="SPY (same $)",
                                              line=dict(color="#8b98a5", width=1.5,
                                                        dash="dot")))
                fige.update_layout(height=380, yaxis_title="Equity $",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(fige, use_container_width=True)

            st.markdown("#### Blotter")
            st.dataframe(mtm["blotter"], use_container_width=True,
                         hide_index=True, height=360)

            if len(mtm["monthly"]):
                st.markdown("#### Monthly returns")
                st.dataframe(mtm["monthly"], use_container_width=True,
                             hide_index=True)

            with st.expander("❓ Why this is the feature that matters most"):
                st.markdown("""
    Backtests can be (accidentally) curve-fit. A **forward paper record** cannot: the timestamps prove every pick was made *before* the outcome. This is exactly how allocators evaluate new managers — months of verified process before a dollar moves. Rules of the game: record every plan (no cherry-picking), let stops do their job, export the CSV after each session, and judge nothing before ~20 closed trades. If after months the record shows an edge over SPY — it's real. If it doesn't — the site just saved you real money.
    """)


    _journal_live()

# ===========================================================================
# 0c. RUNNER — the trade lifecycle machine
# ===========================================================================
with tab_runner:
    st.subheader("⚙️ Runner — the machine trades it, you read the log")
    st.caption("Bar-by-bar lifecycle on real history: entries (trend or dip, "
               "auto-picked by Hurst), scale-outs at +1R and +2R, breakeven "
               "jumps, chandelier trail, time exits — every event logged with "
               "the model state and the reason. Ends with today's live "
               "status and tomorrow's order.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    rn_tkr = c1.text_input("Ticker", value="NVDA", key="rn").upper().strip()
    rn_tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="rnp")
    rn_acct = c3.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="rn_acct")
    rn_mode = c4.selectbox("Mode", ["auto", "trend", "dip"], key="rn_mode")

    if st.button("▶️ Run the machine", type="primary", key="rn_run"):
        st.session_state["rn_params"] = dict(
            rn_tkr=rn_tkr, rn_tf=rn_tf, rn_acct=float(rn_acct),
            rn_mode=rn_mode)

    @st.fragment(run_every=LIVE_EVERY)
    def _runner_live():
        if "rn_params" not in st.session_state:
            return
        _p = st.session_state["rn_params"]
        rn_tkr, rn_tf = _p["rn_tkr"], _p.get("rn_tf", "Daily")
        rn_acct, rn_mode = _p["rn_acct"], _p["rn_mode"]
        _rm = tf_meta(rn_tf)
        with st.spinner(f"Replaying every {rn_tf} bar through the models…"):
            df = fetch_tf(rn_tkr, rn_tf)
            if rn_tf == "Daily":
                df = patch_live_bar(df, rn_tkr)
            if len(df) < _rm["min_bars"]:
                st.error(f"Need at least ~{_rm['min_bars']} {rn_tf} bars.")
                return
            res = run_machine(df, account=float(rn_acct), mode=rn_mode)
            if "error" in res:
                st.error(res["error"])
                return

        # live state card
        stt = res["state"]
        if stt["in_position"]:
            st.markdown(f"""
            <div class="verdict v-long">
              <div>
                <h2>🟢 IN POSITION — {rn_tkr}</h2>
                <div class="sub">{stt['shares']} shares @ ${stt['entry']:,.2f}
                 · held {stt['bars_held']} bars · {stt['scaled']} ·
                 unrealized ${stt['unrealized']:,.0f}</div>
              </div>
              <div style="text-align:right">
                <div style="font-size:.85rem;opacity:.8">Active stop</div>
                <div style="font-size:1.6rem;font-weight:800">
                  ${stt['stop_now']:,.2f}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict v-none">
              <div><h2>⚪ FLAT — {rn_tkr}</h2>
              <div class="sub">machine mode: {res['mode'].upper()}</div></div>
            </div>""", unsafe_allow_html=True)
        st.info(f"**Tomorrow's order:** {stt['tomorrow']}")

        sc = st.columns(len(res["stats"]))
        for col, (k, v) in zip(sc, res["stats"].items()):
            col.metric(k, v)

        # price chart with event markers
        ev = res["events"]
        figrn = go.Figure()
        figrn.add_trace(go.Candlestick(
            x=df.index[-504:], open=df["Open"][-504:], high=df["High"][-504:],
            low=df["Low"][-504:], close=df["Close"][-504:], name=rn_tkr))
        if len(ev):
            ev_plot = ev[pd.to_datetime(ev["date"]).isin(df.index[-504:])]
            marker_map = [("ENTRY", "triangle-up", "#10b981"),
                          ("SCALE", "diamond", "#22d3ee"),
                          ("TARGET", "star", "#6ee7b7"),
                          ("STOP", "x", "#ef4444"),
                          ("EXIT", "circle", "#f59e0b")]
            for key, sym, col_ in marker_map:
                sub = ev_plot[ev_plot["event"].str.contains(key)]
                if len(sub):
                    figrn.add_trace(go.Scatter(
                        x=pd.to_datetime(sub["date"]), y=sub["price"],
                        mode="markers", name=key.title(),
                        marker=dict(symbol=sym, size=11, color=col_,
                                    line=dict(width=1, color="#0b0f14"))))
        figrn.update_layout(height=520, xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=10, t=30, b=10),
                            **PLOTLY_LAYOUT)
        st.plotly_chart(figrn, use_container_width=True)

        # ---- 🗣️ Decision feed — the machine narrates itself -----------------
        st.markdown("#### 🗣️ Decision feed — the machine explains every move")
        ev_feed = res["events"].iloc[::-1].head(10)
        ICONS = {"ENTRY": "🟢", "SCALE": "💰", "TARGET": "🎯",
                 "STOP": "🛑", "BREAKEVEN": "🟡", "TRAIL": "📉",
                 "SIGNAL": "🔵", "TIME": "⏰"}
        for _, e_ in ev_feed.iterrows():
            ic = next((v for k, v in ICONS.items() if k in e_["event"]), "•")
            is_entry = "ENTRY" in e_["event"]
            cls = "reason-pro" if is_entry or "SCALE" in e_["event"] \
                or "TARGET" in e_["event"] else "reason-con"
            st.markdown(
                f"<div class='{cls}' style=\"font-family:'JetBrains Mono',"
                f"monospace\">{ic} <b>{e_['date']}</b> · {e_['event']} · "
                f"{e_['shares']} sh @ ${e_['price']:,.2f}<br>"
                f"<span style='color:#8b98a5'>saw: score {e_['score']:+.2f} · "
                f"BX {e_['bx']} · RSI2 {e_.get('rsi2','—')}</span><br>"
                f"→ {e_['note']}</div>", unsafe_allow_html=True)

        # ---- 🧾 Deal ledger — full closure reports ---------------------------
        if len(res.get("closures", [])):
            st.markdown("#### 🧾 Deal ledger — every round-trip, fully "
                        "accounted")
            cl = res["closures"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Round-trips", len(cl))
            k2.metric("Won", f"{float((cl['P&L $']>0).mean())*100:.0f}%")
            k3.metric("Avg return on capital",
                      f"{float(cl['return on capital %'].mean()):+.2f}%")
            k4.metric("Total realized P&L",
                      f"${float(cl['P&L $'].sum()):+,.0f}")
            st.dataframe(cl.iloc[::-1], use_container_width=True,
                         hide_index=True, height=320)

        # ---- 🎓 What the machine learned -------------------------------------
        if res.get("lessons"):
            st.markdown("#### 🎓 What the machine learned on this ticker")
            for L in res["lessons"]:
                cls = "reason-con" if "⚠️" in L else "reason-pro"
                st.markdown(f"<div class='{cls}'>🎓 {L}</div>",
                            unsafe_allow_html=True)
            with st.expander("❓ How the self-learning works"):
                st.markdown("""
After every run the machine audits its own closed trades on this ticker and derives lessons **from data, not vibes**: which exit types made vs cost money, whether entries with strong B-X confirmation outperformed weak ones, the win rate and average return **on deployed capital** per round-trip, and the time-asymmetry check — if losers are held longer than winners, that's the single most common way traders bleed, and the machine will flag itself for it. Every lesson traces to rows you can see in the deal ledger above. This is what "self-learning" honestly means at this scale: measured feedback, not magic.
""")

        st.markdown("#### 📜 The log — every decision, with its reason")
        if len(ev):
            st.dataframe(ev.iloc[::-1], use_container_width=True,
                         hide_index=True, height=420)
        else:
            st.info("The machine never found an entry it liked on this "
                    "ticker — that is a valid (and cheap) outcome.")

        figeq = go.Figure(go.Scatter(x=res["equity"].index, y=res["equity"],
                                     line=dict(color="#10b981", width=2),
                                     name="Machine equity"))
        figeq.update_layout(height=260, yaxis_title="Equity $",
                            margin=dict(l=10, r=10, t=30, b=10),
                            **PLOTLY_LAYOUT)
        st.plotly_chart(figeq, use_container_width=True)

        with st.expander("❓ How the machine decides"):
            st.markdown("""
**Entry** — trend mode: composite BUY + B-Xtrender rising & positive + above the 200-SMA. Dip mode: RSI(2) < 10 panic *inside the Fibonacci 0.382–0.786 pocket* of an uptrend. Mode auto-picked by the ticker's Hurst exponent.

**The lifecycle** — at **+1R**: sell ⅓, stop jumps to breakeven (the trade can no longer lose). At **+2R**: sell another ⅓, stop jumps to entry+1R (profit is locked). The last third rides a 2.5×ATR chandelier trail as far as the trend goes. Stale unprofitable trades get time-stopped.

**Why scale-outs** — they resolve the eternal "take profit vs let it run" fight by doing both: the win rate rises (thirds get banked), the tail stays open (the runner catches the big moves). It costs a little expectancy vs all-or-nothing in pure trends — and buys a smoother equity curve and a calmer trader. That's usually the right trade.
""")

    _runner_live()

# ===========================================================================
# 1. TRADE DESK
# ===========================================================================
with tab_desk:
    c1, c2, c3, c4, c5 = st.columns([1.7, 1, 1, 1, 1.3])
    tkr = c1.text_input("Ticker", value="NVDA", key="desk").upper().strip()
    tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="desk_tf")
    account = c3.number_input("Account $", 500, 1_000_000, 5000, step=500)
    risk_pct = c4.slider("Risk/trade %", 0.5, 3.0, 1.0, 0.25)
    use_opts = c5.toggle("Include options skew (slower)", value=False)

    if st.button("Run desk analysis", type="primary", key="deskrun"):
        st.session_state["desk_params"] = dict(
            tkr=tkr, account=float(account), risk_pct=float(risk_pct),
            use_opts=bool(use_opts), tf=tf)

    @st.fragment(run_every=LIVE_EVERY)
    def _desk_live():
        if "desk_params" not in st.session_state:
            return
        _p = st.session_state["desk_params"]
        tkr, account = _p["tkr"], _p["account"]
        risk_pct, use_opts = _p["risk_pct"], _p["use_opts"]
        tf = _p.get("tf", "Daily")
        _m = tf_meta(tf)
        with st.spinner(f"Crunching 7 models on {tf} bars…"):
            df = fetch_tf(tkr, tf)
            if tf == "Daily":
                df = patch_live_bar(df, tkr)
            if df.empty or len(df) < _m["min_bars"]:
                st.error(f"Not enough {tf} bars for {tkr} "
                         f"(need ~{_m['min_bars']}).")
                return
            skew = None
            flow_share = None
            if use_opts:
                try:
                    _, chain_d = fetch_chains(tkr, max_expiries=4)
                    skew = skew_25(chain_d)
                    uf_d = unusual_flow(chain_d, top_n=30)
                    if len(uf_d):
                        callp = uf_d.loc[uf_d["type"] == "C", "premium_$"].sum()
                        totp = uf_d["premium_$"].sum()
                        flow_share = float(callp / totp) if totp > 0 else None
                except Exception:
                    skew = None
            v = analyze(df, account=account, risk_pct=risk_pct, skew=skew,
                        flow_call_share=flow_share)
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

        # ---- 📖 PLAYBOOK — the WHEN engine -----------------------------------
        pb = build_playbook(df, account=account, risk_pct=risk_pct)
        urg_color = {"🟢 ACTIONABLE": "#10b981", "🟡 FAST SETUP": "#f59e0b",
                     "🟡 WATCH": "#f59e0b", "⚪ NO TRADE": "#6b7280",
                     "🟢 CALM": "#10b981", "🟡 SOON": "#f59e0b",
                     "🟠 TODAY": "#f97316", "🔴 IMMEDIATE": "#ef4444"}.get(
            pb["urgency"], "#8b98a5")
        st.markdown(f"""
        <div style="border:1px solid {urg_color};border-radius:16px;
                    padding:18px 22px;margin-bottom:14px;
                    background:linear-gradient(135deg,rgba(19,26,34,.95),
                    rgba(11,15,20,.95));box-shadow:0 0 24px {urg_color}22">
          <div style="font-size:.8rem;color:{urg_color};font-weight:800;
                      letter-spacing:1px">📖 PLAYBOOK · {pb['urgency']} ·
                      {pb['greens']}/5 gates green</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:6px;
                      font-family:'JetBrains Mono',monospace">
                      {pb['instruction']}</div>
        </div>""", unsafe_allow_html=True)
        gc = st.columns(5)
        for col, (name, ok, detail) in zip(gc, pb["gates"]):
            col.markdown(f"{'✅' if ok else '⛔'} **{name.split('(')[0]}**")
            col.caption(detail)
        with st.expander("❓ The playbook — when to enter, manage, exit"):
            st.markdown("""
The playbook runs the exact five gates the backtest engine trades, live:
**ENTER** when all 5 are green (with shares/stop/scale levels printed). **DIP SETUP** when RSI(2) panics inside an intact uptrend — the fast scalp lane. **STALK** at 3–4 greens: it names what's blocking, you set an alert. **STAND DOWN** below that — no setup exists, and forcing one is how accounts bleed.

Once you're in a trade, re-run with your entry/stop (Runner tracks this automatically) and the playbook switches to management: **PROTECT** at +1R (stop → breakeven), **SCALE** at +2R (bank a third), **TIGHTEN** when B-X rolls over, **EXIT** on a composite flip or stop violation — each with an urgency color. It's the same lifecycle the ⚙️ Runner trades historically, pointed at *right now*.
""")

        # ---- Regime + vol forecast row --------------------------------------
        garch = garch_forecast(df) if tf == "Daily" else {}
        rg1, rg2, rg3, rg4, rg5 = st.columns([1.6, 1, 1, 1, 1])
        rg1.markdown(f"<div class='regime-badge'>{reg['regime']}</div>"
                     f"<div style='color:#8b98a5;font-size:.85rem;margin-top:6px'>"
                     f"{reg['playbook']}</div>", unsafe_allow_html=True)
        rg2.metric("EWMA vol (annual)", f"{vol['sigma_annual_pct']}%")
        rg3.metric("Expected move (1 day)", f"±${vol['expected_move_1d']:,.2f}")
        rg4.metric("GARCH(1,1) 1-day move",
                   f"±${garch['move_1d']:,.2f}" if garch else "—",
                   delta=f"persistence {garch['persistence']}" if garch else None,
                   delta_color="off")
        rg5.metric("Hurst exponent", h,
                   delta="trending" if h > 0.55 else
                   "mean-reverting" if h < 0.45 else "random walk",
                   delta_color="off")
        with st.expander("❓ Regime, EWMA volatility & Hurst — why they matter"):
            st.markdown("""
- **Regime quadrant** — price vs its 200-day average (bull/bear) × current volatility vs its own history (calm/storm). Each quadrant has a playbook; most losing streaks come from running a bull-calm playbook in a bear-storm.
- **EWMA vol (RiskMetrics λ=0.94)** — the industry-standard forecast of tomorrow's volatility, weighting recent days most. The ± number is the *expected* one-day move: intraday wiggles inside it are noise, not signal.
- **Hurst exponent** — the ticker's memory. >0.5 moves tend to continue (trust trend models), <0.5 they reverse (trust mean-reversion), ≈0.5 random walk.
""")

        _remember("desk", {"ticker": tkr, "verdict": v["verdict"],
                           "conviction": v["conviction"],
                           "garch": garch.get("sigma_annual_pct") if garch else None,
                           "tf": tf})
        _mem_o = st.session_state["memory"].get("options")
        if _mem_o and _mem_o.get("ticker") == tkr:
            st.caption(f"🧠 From your options run: {_mem_o['vol_state']} · "
                       f"skew {_mem_o.get('skew','—')} — factored into how "
                       f"you should express this view (see 🌋 Edge finder).")

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
            if v["verdict"] == "LONG":
                if st.button(f"📒 Track this trade ({tkr}: {v['shares']} sh, "
                             f"stop ${v['stop']:,.2f})", key="desk_rec"):
                    _plan1 = pd.DataFrame([{
                        "ticker": tkr, "shares": v["shares"],
                        "entry ~": v["entry"], "stop": v["stop"],
                        "target": v["target"],
                        "conviction": v["conviction"]}])
                    _jj = load_journal()
                    _jj, _n = record_plan(_jj, _plan1, reg["regime"],
                                          float(account))
                    save_journal(_jj)
                    st.success(f"Recorded {tkr} to the 📒 Track record "
                               f"(UTC-stamped, regime: {reg['regime']}). "
                               f"Export the CSV after your session!")
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

        if odds["p_target_first"] + odds["p_stop_first"] > 10:
            _wr = odds["p_target_first"] / max(
                odds["p_target_first"] + odds["p_stop_first"], 1e-9)
            _ror = risk_of_ruin(win_rate=_wr, avg_win=v["rr"], avg_loss=1.0,
                                risk_per_trade_pct=risk_pct)
            if _ror:
                st.caption(f"🛡️ **Risk of a 30% drawdown** trading this "
                           f"setup repeatedly at {risk_pct}% risk: "
                           f"**{_ror['prob_of_ruin_%']}%** "
                           f"({_ror['verdict']}) · expectancy "
                           f"{_ror['expectancy_R']:+.2f}R per trade "
                           f"(5,000 simulated careers).")

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
                            xaxis_title=f"{tf} bars ahead",
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

        st.markdown("**🔬 BX Lab — probability-calibrated states "
                    f"({tkr}, 5-bar horizon):**")
        sp_tab = state_probabilities(df)
        st.dataframe(sp_tab, use_container_width=True, hide_index=True)
        st.caption(f"Current state: **{sp_tab.attrs.get('current_state','—')}**"
                   " — find it in the table for its historical odds.")
        if st.button("🧪 Run BX parameter sweep (8 presets, OOS-validated)",
                     key="bx_sweep"):
            with st.spinner("Testing 8 parameter sets, out-of-sample…"):
                sw = parameter_sweep(df)
            st.dataframe(sw, use_container_width=True, hide_index=True)
            best = sw.iloc[0]
            st.caption(f"Best OOS: **{best['preset']}** (Sharpe "
                       f"{best['OOS Sharpe']}). Watch the **overfit gap** "
                       f"column — a preset that shines in-sample and dies "
                       f"OOS is curve-fitting, and this table catches it "
                       f"in the act.")

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

    _desk_live()

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
    c1, c2, c3, c4, c5 = st.columns(5)
    bt_tkr = c1.text_input("Ticker", value="AAPL", key="bt").upper().strip()
    bt_tf = c2.selectbox("Timeframe", TF_LABELS, index=1, key="bt_tf")
    bt_cash = c3.number_input("Starting cash $", 500, 1_000_000, 5000,
                              step=500)
    bt_risk = c4.slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)
    bt_mode = c5.selectbox("Engine", ["auto", "core", "trend", "dip",
                                      "blend"], index=0,
                           help="auto picks per ticker by Hurst exponent")
    bt_short = st.checkbox("Allow shorts (trend mode, below 200-SMA only)",
                           value=False, key="bt_short")

    if st.button("Run backtest", type="primary", key="btrun"):
        _bm = tf_meta(bt_tf)
        df = fetch_tf(bt_tkr, bt_tf)
        if len(df) < _bm["min_bars"]:
            st.error(f"Not enough {bt_tf} bars (need ~{_bm['min_bars']}).")
        else:
            cfg = BTConfig(starting_cash=float(bt_cash),
                           risk_per_trade=bt_risk / 100, mode=bt_mode,
                           allow_short=bt_short,
                           bars_per_year=_bm["bars_per_year"])
            res = run_backtest(df, cfg)
            st.caption(f"Engine: **{res.mode_used.upper()}** on **{bt_tf}** "
                       f"bars — Sharpe/CAGR annualized with "
                       f"{_bm['bars_per_year']} bars/yr"
                       f"{' · mode picked by Hurst' if bt_mode == 'auto' else ''}")
            _remember("backtest", {"ticker": bt_tkr,
                                   "mode": res.mode_used,
                                   "sharpe": res.metrics.get("Sharpe"),
                                   "tf": bt_tf})
            cols = st.columns(5)
            for col, (k, val) in zip(cols * 2, res.metrics.items()):
                col.metric(k, val if val is not None else "—")

            with st.expander("📉 Why is my Sharpe low? (read this once)"):
                st.markdown("""
Three usual suspects, in order of impact:
1. **Cash drag** — risking 1%/trade with a 2.5×ATR stop deploys only ~15-25% of capital; the rest earns nothing while buy & hold is 100% invested. **Fix: the CORE engine** — near-fully invested while the regime is healthy (above 200-SMA + B-Xtrender positive), cash when it breaks. B&H-like CAGR in bulls, a fraction of the drawdown in bears.
2. **Single-name concentration** — one stock's noise dominates. Sharpe rises with diversification faster than with better signals; that's what the 🧬 Alpha engine's multi-position plan is for.
3. **Trend systems are streaky** — 40% win rates with occasional big winners produce lumpy equity. The DIP engine smooths it (77%+ win rate, small wins). A **blend** of core + dip is how small systematic accounts actually maximize Sharpe.
""")

            with st.expander("⚙️ What's inside the v2 engine?"):
                st.markdown("""
Two strategies, auto-selected per ticker by its **Hurst exponent**:
- **TREND** — follows the 7-model composite signal. For tickers whose moves *continue* (H > 0.5).
- **DIP** — Connors-style RSI(2) pullback buyer: buys short-term panic **inside an uptrend**, exits on the snap-back or a strict time limit — now with a **Fibonacci pocket filter**: panic is only bought when price sits inside the 0.382–0.786 retracement zone of the dominant swing (or on a B-Xtrender buy-turn). Confluence, not just oversold.
- **CORE** — improved buy & hold: ~fully invested while price > 200-SMA **and** the B-Xtrender long oscillator is positive; steps to cash when either breaks. Fixes the cash-drag problem that makes signal strategies lose to B&H in bull markets.
- **Shorts (optional)** — trend mode can short below the 200-SMA when the composite says SELL and B-Xtrender confirms (falling & negative). Symmetric stops, breakeven and time exits.
- **B-Xtrender confirmation** — trend longs now require the long oscillator positive AND the T3 rising. Fewer, better entries.

Risk mechanics on every trade:
- **Regime gate (Faber 2007)** — longs only above the 200-day SMA. No knife-catching.
- **Breakeven stop after +1R** — once the trade is one risk-unit in profit, the stop jumps to entry. Converts would-be losers into scratches → directly raises win rate.
- **Time stop** — not working within 10–20 bars? Out. Dead capital is a cost.
- **Volatility-targeted sizing (Moreira & Muir 2017)** — risk shrinks automatically when the stock's ATR% is elevated vs its own history.
- Chandelier 2.5×ATR trail on trend trades, next-bar-open execution, commissions included.
""")

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

            # ---- 🔬 Statistical validation ------------------------------
            st.markdown("### 🔬 Is this edge REAL? (institutional validation)")
            st.caption("The tests that separate genuine edges from the "
                       "thousands you'd find by data-mining. This is what "
                       "makes a quant trust — or discard — a backtest.")
            N_TRIALS = 20   # we test ~20 models across the site; be honest about it
            rets_bt = res.equity.pct_change().dropna()
            sharpe_bt = res.metrics.get("Sharpe", 0) or 0
            tstat_bt = (sharpe_bt * np.sqrt(len(rets_bt) / 252)
                        if len(rets_bt) > 252 else sharpe_bt)

            vc1, vc2 = st.columns(2)
            with vc1:
                dsr = deflated_sharpe(sharpe_bt, N_TRIALS, len(rets_bt))
                if "error" not in dsr:
                    st.markdown(f"**Deflated Sharpe** — {dsr['verdict']}")
                    st.caption(f"Observed {dsr['observed_sharpe']} vs "
                               f"noise-benchmark {dsr['deflated_benchmark_ann']} "
                               f"· P(real) = {dsr['DSR_probability']}")
                perm = permutation_test(rets_bt)
                if "error" not in perm:
                    st.markdown(f"**Permutation test** — {perm['verdict']}")
                    st.caption(f"Real Sharpe {perm['actual_sharpe']} vs luck's "
                               f"95th pct {perm['perm_sharpe_95pct']} · "
                               f"p = {perm['perm_p_value']}")
            with vc2:
                hc = haircut_pvalue(tstat_bt, N_TRIALS)
                st.markdown(f"**Multiple-testing haircut** — {hc['verdict']}")
                st.caption(f"Raw p {hc['raw_p']} → after correcting for "
                           f"{N_TRIALS} models: {hc['bonferroni_p']}")
                if len(res.trades):
                    bs = bootstrap_cagr(res.trades["pnl"],
                                        starting=float(bt_cash))
                    if "error" not in bs:
                        st.markdown(f"**Bootstrap 90% CI** — {bs['verdict']}")
                        st.caption(f"Return CI: {bs['CI90_low_%']}% … "
                                   f"{bs['median_return_%']}% … "
                                   f"{bs['CI90_high_%']}%")

            with st.expander("❓ Why these four tests are the real grade"):
                st.markdown("""
Anyone can produce a pretty backtest — try 1,000 parameter combos and *one* will look brilliant by pure chance. These tests fight that:

- **Deflated Sharpe** (Bailey & López de Prado 2014) — lowers your Sharpe to account for how many strategies were tried. A Sharpe of 1.5 from testing 20 models is worth far less than 1.5 from testing one.
- **Multiple-testing haircut** (Harvey, Liu & Zhu 2016) — a t-stat of 2 (the classic "significant") is NOT significant when mined across 20 signals. This corrects it.
- **Permutation test** — randomly flips the sign of each day's return 500× to build the distribution of "luck." If your real Sharpe isn't clearly above that cloud, you have nothing.
- **Bootstrap CI** — resamples your trades 1,000× for a 90% confidence band on returns. If the band straddles zero, you genuinely don't know if the strategy works — and that knowledge is worth more than false confidence.

**If a strategy passes all four, it's in rarer air than 99% of what retail traders trade on.** If it fails — the app just saved you from a mirage. That honesty is the highest-value thing this whole site does.
""")
            with st.expander("❓ Why walk-forward matters"):
                st.markdown("""
The same rules re-run on 4 separate sequential periods. A real edge shows in most folds; a curve-fit illusion shines in one and dies in the rest. The single best overfitting detector available to a retail quant.
""")

            if len(res.trades):
                st.markdown("#### 🗺️ Every trade on the chart — "
                            "entry, exit, price, reason")
                tr = res.trades.copy()
                figtr = go.Figure()
                figtr.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name=bt_tkr,
                    increasing_line_color="#2a3644",
                    decreasing_line_color="#1a222c",
                    increasing_fillcolor="#2a3644",
                    decreasing_fillcolor="#1a222c"))
                figtr.add_trace(go.Scatter(
                    x=pd.to_datetime(tr["entry_date"]), y=tr["entry"],
                    mode="markers", name="Entry",
                    marker=dict(symbol="triangle-up", size=12,
                                color="#10b981",
                                line=dict(width=1, color="#0b0f14")),
                    customdata=tr[["entry"]].values,
                    hovertemplate="ENTRY @ $%{y:.2f}<br>%{x|%Y-%m-%d}"
                                  "<extra></extra>"))
                exit_colors = tr["reason"].map(
                    {"stop": "#ef4444", "breakeven": "#f59e0b",
                     "time": "#8b98a5", "signal": "#22d3ee",
                     "target(rsi)": "#6ee7b7"}).fillna("#e6edf3")
                figtr.add_trace(go.Scatter(
                    x=pd.to_datetime(tr["exit_date"]), y=tr["exit"],
                    mode="markers", name="Exit",
                    marker=dict(symbol="triangle-down", size=12,
                                color=exit_colors,
                                line=dict(width=1, color="#0b0f14")),
                    customdata=np.stack([tr["reason"], tr["pnl"]], axis=-1),
                    hovertemplate="EXIT @ $%{y:.2f}<br>reason: %{customdata[0]}"
                                  "<br>P&L: $%{customdata[1]}<br>%{x|%Y-%m-%d}"
                                  "<extra></extra>"))
                # connect entry->exit with a thin win/loss colored line
                for _, t_ in tr.iterrows():
                    figtr.add_trace(go.Scatter(
                        x=[pd.to_datetime(t_["entry_date"]),
                           pd.to_datetime(t_["exit_date"])],
                        y=[t_["entry"], t_["exit"]], mode="lines",
                        line=dict(width=1.2,
                                  color="rgba(16,185,129,.55)" if t_["pnl"] > 0
                                  else "rgba(239,68,68,.55)"),
                        showlegend=False, hoverinfo="skip"))
                figtr.update_layout(height=520,
                                    xaxis_rangeslider_visible=False,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    **PLOTLY_LAYOUT)
                st.plotly_chart(figtr, use_container_width=True)
                st.caption("▲ green = entry · ▼ exit colored by reason "
                           "(🔴 stop, 🟠 breakeven, 🔵 signal, 🟢 RSI target, "
                           "⚪ time) · connecting line green = winner, "
                           "red = loser. Hover any marker for exact price, "
                           "date, reason and P&L.")

                colR, colM = st.columns(2)
                with colR:
                    if "R" in res.trades:
                        figR = go.Figure(go.Histogram(
                            x=res.trades["R"], nbinsx=24,
                            marker_color=np.where(
                                np.histogram(res.trades["R"], bins=24)[1][:-1]
                                >= 0, "#10b981", "#ef4444")))
                        figR.add_vline(x=0, line_color="#8b98a5")
                        figR.update_layout(height=300,
                                           title="R-multiple distribution",
                                           xaxis_title="R", yaxis_title="trades",
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           **PLOTLY_LAYOUT)
                        st.plotly_chart(figR, use_container_width=True)
                with colM:
                    if "MAE_R" in res.trades:
                        figM = go.Figure(go.Scatter(
                            x=res.trades["MAE_R"], y=res.trades["MFE_R"],
                            mode="markers",
                            marker=dict(size=9,
                                        color=np.where(res.trades["pnl"] > 0,
                                                       "#10b981", "#ef4444"))))
                        figM.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                       line=dict(color="#8b98a5", dash="dot"))
                        figM.update_layout(height=300,
                                           title="MAE vs MFE (per trade, in R)",
                                           xaxis_title="Max pain (MAE, R)",
                                           yaxis_title="Max gain (MFE, R)",
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           **PLOTLY_LAYOUT)
                        st.plotly_chart(figM, use_container_width=True)
                with st.expander("❓ R-distribution & MAE/MFE — pro trade forensics"):
                    st.markdown("""
- **R-multiple distribution** — every trade's P&L in risk units. A healthy system: losses clustered at −1R (stops doing their job), a right tail of +2R/+3R winners. Losses beyond −1R = slippage/gap problem; no right tail = you're cutting winners.
- **MAE vs MFE** — each dot is one trade: how far it went AGAINST you (x) vs FOR you (y). Green dots high-left = ideal (little pain, much gain). Red dots that reached high MFE = winners you gave back → tighten trailing. Many reds with tiny MAE = stops too tight; they died without ever being wrong.
This is the same trade-forensics workflow a Bloomberg BTST user runs.
""")

                st.markdown("#### Trade log")
                st.dataframe(res.trades, use_container_width=True, height=300)

                # ---- 🛡️ Risk-of-ruin on THIS strategy's actual stats ------
                wins_ = res.trades[res.trades["pnl"] > 0]["pnl"]
                losses_ = -res.trades[res.trades["pnl"] < 0]["pnl"]
                if len(wins_) >= 3 and len(losses_) >= 3:
                    ror = risk_of_ruin(
                        win_rate=len(wins_) / len(res.trades),
                        avg_win=float(wins_.mean()),
                        avg_loss=float(losses_.mean()),
                        risk_per_trade_pct=bt_risk)
                    if ror:
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Risk of 30% drawdown",
                                  f"{ror['prob_of_ruin_%']}%",
                                  delta=ror["verdict"], delta_color="off")
                        r2.metric("Payoff ratio (avg W/L)",
                                  ror["payoff_ratio"])
                        r3.metric("Expectancy per trade",
                                  f"{ror['expectancy_R']:+.2f}R")
                        kl = kelly_ladder(len(wins_) / len(res.trades),
                                          ror["payoff_ratio"])
                        r4.metric("Kelly (full/half/¼)",
                                  f"{kl['full_kelly_%']}/{kl['half_kelly_%']}"
                                  f"/{kl['quarter_kelly_%']}%"
                                  if kl["edge"] else "No edge")
                        st.caption("Risk of ruin: 5,000 Monte Carlo careers "
                                   "of 200 trades each with THIS strategy's "
                                   "real win rate and payoff, at your chosen "
                                   "risk %. The single most important "
                                   "number on this page.")

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


            # ---- 💡 EDGE FINDER — our models vs the options market ------------
            st.markdown("### 💡 Edge finder — what WE forecast vs what "
                        "OPTIONS price")
            df_u = fetch_history(opt_tkr, period="2y")
            g_u = garch_forecast(df_u) if len(df_u) > 260 else {}
            ew_u = ewma_vol(df_u) if len(df_u) > 60 else {}
            vr = vrp(atm_iv, g_u.get("sigma_annual_pct"),
                     ew_u.get("sigma_annual_pct")) if atm_iv else {}
            rich = iv_richness(df_u, atm_iv) if atm_iv else {}

            mem_d = st.session_state.get("memory", {}).get("desk", {})
            if mem_d.get("ticker") == opt_tkr:
                direction = mem_d["verdict"]
                dir_src = f"🧠 from your Trade-desk run (conviction {mem_d['conviction']})"
            else:
                try:
                    v_q = analyze(df_u)
                    direction = v_q["verdict"]
                    dir_src = "computed fresh by the 7-model verdict engine"
                except Exception:
                    direction, dir_src = "NO TRADE", "unavailable"

            e1, e2, e3, e4 = st.columns(4)
            if vr:
                e1.metric("IV vs our vol forecast",
                          f"{vr['iv']}% vs {vr['forecast_vol']}%",
                          delta=f"VRP {vr['vrp_pts']:+.1f} pts",
                          delta_color="off")
            if rich:
                e2.metric("IV richness percentile", f"{rich['iv_pctile']}%",
                          help="Where today's IV sits vs this ticker's own "
                               "1-year realized-vol distribution.")
            e3.metric("Directional view", direction, delta=dir_src,
                      delta_color="off")
            mvm = {}
            try:
                paths_o = simulate(df_u, days=int(ts["dte"].iloc[0]),
                                   n_paths=2000)
                mvm = move_vs_model(exp_move, paths_o, spot,
                                    int(ts["dte"].iloc[0]))
            except Exception:
                pass
            if mvm:
                e4.metric("Move: market vs model",
                          f"±${mvm['market_move']} vs ±${mvm['model_move']}",
                          delta=mvm["read"], delta_color="off")

            if vr:
                st.markdown(f"**Vol verdict: {vr['state']}**")
                sug = suggest_structure(direction, vr["state"], chain,
                                        near_exp, spot, bs_greeks)
                st.success(f"**🎯 Suggested structure: {sug['name']}**  \n"
                           f"`{sug['legs']}`  \n{sug['logic']}")
                _remember("options", {"ticker": opt_tkr,
                                      "vol_state": vr["state"],
                                      "skew": skew,
                                      "atm_iv": atm_iv})
            with st.expander("❓ Where the options edge actually comes from"):
                st.markdown("""
The one durable, research-backed edge in listed options is the **variance risk premium** (Carr & Wu 2009): implied vol *persistently* overprices realized vol, because the world pays up for insurance. Everything in this panel is that comparison, done properly:

- **IV vs our forecast** — ATM implied vol against a GARCH(1,1) + EWMA blend forecast of what vol will actually be. Gap > +4 pts = the market is overpaying for options → *selling* structures have tailwind. Negative gap = options are statistically cheap → *own* them.
- **IV richness percentile** — level lies, rank doesn't. 90th percentile IV on a boring stock beats 40% IV on a meme stock.
- **Move: market vs model** — the straddle's expected move against our own 2,000-path Monte Carlo at the same horizon. Disagreement = someone is wrong; the panel tells you which side to take.
- **The structure suggester** fuses your **directional view** (from the Trade desk — the tabs share memory) with the **vol state** into one concrete trade with delta-picked strikes: bullish+rich IV → put credit spread (get *paid* to be long); bullish+cheap IV → call debit spread (own the move at a discount); no direction+rich IV → iron condor (harvest the premium). Direction, vol, and structure must all agree — that's the whole edge.

⚠️ Honesty: strikes are delta-suggestions from delayed data — always check live quotes, and spreads on Blink need options approval. Defined-risk structures only; never naked short options on a $5K account.
""")

            st.markdown("---")

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
# 7. PORTFOLIO & PAIRS (awesome-quant: PyPortfolioOpt, statsmodels)
# ===========================================================================
with tab_pp:
    st.subheader("⚖️ Portfolio optimizer")
    st.caption("PyPortfolioOpt: Max-Sharpe (Markowitz), Min-Vol, and HRP — "
               "Hierarchical Risk Parity (López de Prado 2016), the robust "
               "one that needs no return forecasts.")
    pp_in = st.text_input("Tickers (comma-separated, 3+)",
                          value="AAPL, MSFT, NVDA, JPM, XOM, GLD",
                          key="pp_in")
    pp_acct = st.number_input("Account $", 500, 1_000_000, 5000, step=500,
                              key="pp_acct")
    if st.button("Optimize", type="primary", key="pp_run"):
        tks = tuple(t.strip().upper() for t in pp_in.split(",") if t.strip())
        with st.spinner("Downloading & optimizing…"):
            data = fetch_many(tks, period="2y")
            px = build_prices(data)
            res = optimize(px, account=float(pp_acct))
        if "error" in res:
            st.error(res["error"])
        else:
            colw = st.columns(3)
            for col, key, title in zip(
                    colw, ("hrp", "max_sharpe", "min_vol"),
                    ("🌳 HRP (recommended)", "🎯 Max Sharpe", "🛡️ Min Vol")):
                r = res.get(key, {})
                with col:
                    st.markdown(f"**{title}**")
                    if "error" in r:
                        st.warning(r["error"])
                    else:
                        st.caption(f"exp. ret {r['ret']}% · vol {r['vol']}% "
                                   f"· Sharpe {r['sharpe']}")
                        wdf = pd.DataFrame(
                            {"weight %": {k: round(v * 100, 1)
                                          for k, v in r["weights"].items()
                                          if v > 0.001}})
                        st.dataframe(wdf, use_container_width=True)
            if res.get("frontier"):
                figf = go.Figure()
                figf.add_trace(go.Scatter(
                    x=[p[0] for p in res["frontier"]],
                    y=[p[1] for p in res["frontier"]],
                    mode="lines", name="Efficient frontier",
                    line=dict(color="#10b981", width=2)))
                for v_, r_, t_ in res.get("assets", []):
                    figf.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers+text",
                                              text=[t_], textposition="top center",
                                              showlegend=False,
                                              marker=dict(size=9,
                                                          color="#8b98a5")))
                figf.update_layout(height=380, xaxis_title="Volatility %",
                                   yaxis_title="Expected return %",
                                   margin=dict(l=10, r=10, t=30, b=10),
                                   **PLOTLY_LAYOUT)
                st.plotly_chart(figf, use_container_width=True)
            if res.get("allocation") and "shares" in res.get("allocation", {}):
                st.markdown("**🧾 Discrete allocation for your account "
                            "(HRP weights):**")
                st.json(res["allocation"])
            with st.expander("❓ Which one should I use?"):
                st.markdown("""
- **HRP** — clusters assets by how they move together and splits risk down the tree. No return forecasts, no unstable matrix math → the weights barely change when the data wiggles. What quants actually deploy.
- **Max Sharpe** — the textbook optimum, but it *inhales* estimation error: tiny changes in expected returns swing the weights wildly. We cap any single name at 35% to tame it.
- **Min Vol** — pure defense. Also the sneaky one: low-vol portfolios historically beat their risk-adjusted expectations (the low-volatility anomaly from the Alpha engine).
""")

    st.markdown("---")
    st.subheader("🔗 Pairs lab — cointegration (Engle-Granger)")
    st.caption("Two stocks whose spread is mean-reverting = a market-neutral "
               "trade: long the cheap one, short the rich one, profit on "
               "convergence — regardless of market direction.")
    p1, p2 = st.columns(2)
    pa = p1.text_input("Ticker A", value="KO", key="pa").upper().strip()
    pb = p2.text_input("Ticker B", value="PEP", key="pb").upper().strip()
    if st.button("Test the pair", type="primary", key="pair_run"):
        with st.spinner("Testing cointegration…"):
            da = fetch_history(pa, period="2y")
            db = fetch_history(pb, period="2y")
            pr = pairs_analysis(da, db)
        if "error" in pr:
            st.error(pr["error"])
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cointegration p-value", pr["pvalue"],
                      delta="cointegrated ✓" if pr["cointegrated"]
                      else "borderline" if pr["borderline"] else "not a pair",
                      delta_color="off")
            k2.metric("Hedge ratio", pr["hedge_ratio"],
                      help=f"1 share {pa} ≈ {pr['hedge_ratio']} shares {pb}")
            k3.metric("Spread z-score", pr["z"])
            k4.metric("Half-life (days)", pr["half_life_days"] or "—")
            st.info(f"**Signal:** {pr['signal']}")

            z = pr["z_series"]
            figz = go.Figure()
            figz.add_trace(go.Scatter(x=z.index, y=z, name="spread z",
                                      line=dict(color="#22d3ee")))
            for lvl, col_ in ((2, "#ef4444"), (-2, "#10b981"), (0, "#8b98a5")):
                figz.add_hline(y=lvl, line_dash="dot", line_color=col_)
            figz.update_layout(height=320, yaxis_title="z-score",
                               margin=dict(l=10, r=10, t=30, b=10),
                               **PLOTLY_LAYOUT)
            st.plotly_chart(figz, use_container_width=True)
            with st.expander("❓ How to read this"):
                st.markdown(f"""
- **p-value ≤ 0.05** — the spread between {pa} and {pb} is statistically mean-reverting (Engle-Granger test). Above 0.10: whatever the chart looks like, it's not a pair.
- **z-score** — how stretched the spread is right now. The classic playbook: enter at |z| ≥ 2 (long the cheap leg, short the rich leg, sized by the hedge ratio), exit near z = 0.
- **Half-life** — how fast the spread typically closes half its gap. 5–30 days = tradeable; 100+ days = your capital will die of boredom.
- Caveat: shorting requires margin; if unavailable, the pair still works as a *relative-value tell* for which of the two names to prefer long.
""")


# ===========================================================================
# 8. EVENT RADAR — Polymarket odds as information (never traded)
# ===========================================================================
with tab_events:
    st.subheader("🌐 Event radar — real-money macro odds")
    st.caption("Live Polymarket probabilities on the events that move US "
               "equities: Fed, recession, CPI, shutdowns, tariffs, elections. "
               "**Information source only — we read these markets, we never "
               "trade them.**")

    if st.button("Scan macro markets", type="primary", key="ev_run"):
        with st.spinner("Reading Polymarket odds…"):
            ev = fetch_macro_markets()
        if ev.empty:
            st.warning("Couldn't reach the Polymarket API right now (or no "
                       "macro markets matched). Try again in a minute.")
        else:
            g = equity_risk_gauge(ev)
            if g:
                _remember("events", {"label": g["label"],
                                     "score": g["score"]})
                c1, c2 = st.columns([1, 2.5])
                c1.metric("Equity event gauge", g["label"],
                          delta=f"score {g['score']:+.2f}",
                          delta_color="off")
                with c2:
                    st.markdown("**Top drivers (real-money odds):**")
                    for q, p, d in g["drivers"]:
                        arrow = "🟢" if d > 0 else "🔴"
                        st.markdown(f"<div class='reason-{'pro' if d>0 else 'con'}'>"
                                    f"{arrow} {q} — **{p:.0f}%**</div>",
                                    unsafe_allow_html=True)

            st.markdown("#### All macro/finance markets (by volume)")
            show = ev.copy()
            show["yes %"] = show["yes %"].astype(float)
            st.dataframe(
                show.style.background_gradient(subset=["yes %"],
                                               cmap="RdYlGn_r",
                                               vmin=0, vmax=100),
                use_container_width=True, height=480, hide_index=True)

            with st.expander("❓ How a stock trader uses prediction markets"):
                st.markdown("""
Prediction-market odds are **real-money consensus** — people betting actual dollars, updated in real time. For an equities desk they answer one question: *what event risk is already priced?*

- **Fed cut at 80%** — a cut that happens is a non-event (priced); a *hold* would be the shock. Trade the surprise, not the event.
- **Recession odds climbing week over week** — tighten stops, favor the defensive side of the screener, respect the regime gate.
- **Shutdown/tariff odds jumping** — expect vol regime shifts; the Trade Desk's EWMA/GARCH will confirm with a lag, this leads.
- The 🧲/⛽ GEX regime + this gauge together tell you *both* how the market is positioned and *what* it's positioned for.

Idea credit where due: the repo you sent reads **market skew as crowd positioning** before entering — that's exactly what this tab does, pointed at macro instead of 5-minute BTC. And per your rule: read-only. We inform the stock process; we don't touch the markets themselves.
""")

# ===========================================================================
# 6. RL LAB — TradeMaster-inspired
# ===========================================================================
with tab_rl:
    st.subheader("🤖 RL lab — a learning agent, evaluated honestly")
    st.caption("Inspired by TradeMaster (NTU, NeurIPS 2023): agent + market "
               "dynamics modeling + PRUDEX-style multi-axis evaluation. "
               "Trained on the first 70% of history, judged ONLY on the "
               "unseen last 30%.")
    c1, c2 = st.columns([2, 1])
    rl_tkr = c1.text_input("Ticker", value="AAPL", key="rl").upper().strip()
    rl_period = c2.selectbox("History", ["5y", "10y"], index=0, key="rlp")

    if st.button("Train & evaluate agent", type="primary", key="rlrun"):
        with st.spinner("Training agent on the first 70%, testing on the rest…"):
            df = fetch_history(rl_tkr, period=rl_period)
            if len(df) < 400:
                st.error("Need at least ~400 bars of history.")
                st.stop()
            res = train_agent(df)
            if "error" in res:
                st.error(res["error"])
                st.stop()
            md = market_dynamics(df)

        # --- current decision ------------------------------------------------
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Current market state", res["current_state"])
        act_color = "🟢" if res["current_action"] == "LONG" else "⚪"
        d2.metric("Agent says", f"{act_color} {res['current_action']}")
        d3.metric("Edge estimate (bps/day)", res["current_confidence"])
        d4.metric("OOS exposure", f"{res['oos_exposure_pct']}%")

        # --- OOS equity ------------------------------------------------------
        st.markdown(f"### Out-of-sample test (from {res['split_date']} — "
                    "data the agent never saw)")
        figr = go.Figure()
        figr.add_trace(go.Scatter(x=res["oos_equity"].index,
                                  y=res["oos_equity"], name="Agent",
                                  line=dict(color="#10b981", width=2)))
        figr.add_trace(go.Scatter(x=res["oos_bh"].index, y=res["oos_bh"],
                                  name="Buy & Hold",
                                  line=dict(color="#8b98a5", width=1.5,
                                            dash="dot")))
        figr.update_layout(height=380, yaxis_title="Growth of $1",
                           margin=dict(l=10, r=10, t=30, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figr, use_container_width=True)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Agent (out-of-sample)**")
            st.json(res["oos_stats"])
        with s2:
            st.markdown("**Buy & Hold (same period)**")
            st.json(res["bh_stats"])

        # --- PRUDEX radar -----------------------------------------------------
        st.markdown("### 🧭 PRUDEX-style evaluation compass")
        ax_a = prudex_scores(res["oos_equity"],
                             exposure_pct=res["oos_exposure_pct"])
        ax_b = prudex_scores(res["oos_bh"], exposure_pct=100)
        cats = list(ax_a.keys())
        figc = go.Figure()
        figc.add_trace(go.Scatterpolar(r=[ax_a[k] for k in cats] + [ax_a[cats[0]]],
                                       theta=cats + [cats[0]], fill="toself",
                                       name="Agent",
                                       line=dict(color="#10b981")))
        figc.add_trace(go.Scatterpolar(r=[ax_b[k] for k in cats] + [ax_b[cats[0]]],
                                       theta=cats + [cats[0]], fill="toself",
                                       name="Buy & Hold",
                                       line=dict(color="#8b98a5")))
        figc.update_layout(height=420, polar=dict(radialaxis=dict(range=[0, 100])),
                           margin=dict(l=40, r=40, t=30, b=30),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figc, use_container_width=True)

        # --- Policy table -------------------------------------------------------
        st.markdown("### 🧠 What the agent learned (full policy — nothing hidden)")
        st.dataframe(res["policy"], use_container_width=True, hide_index=True)

        # --- Market dynamics strip ----------------------------------------------
        st.markdown("### 🌍 Market dynamics modeling (TradeMaster MDM concept)")
        recent = md.iloc[-504:]
        figm = go.Figure()
        for s_i, (style, color) in enumerate(zip(MDM_STYLES, MDM_COLORS)):
            mask = recent["style"] == s_i
            if mask.any():
                figm.add_trace(go.Bar(x=recent.index[mask],
                                      y=np.ones(int(mask.sum())),
                                      marker_color=color, name=style,
                                      marker_line_width=0))
        figm.update_layout(height=140, barmode="stack", bargap=0,
                           yaxis=dict(visible=False),
                           margin=dict(l=10, r=10, t=10, b=10),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(figm, use_container_width=True)
        st.caption(f"Current market style: **{md['label'].iloc[-1]}**")

        with st.expander("❓ What is this & why it's honest"):
            st.markdown("""
**The agent** learns the expected next-day return for each of 12 market states (trend × B-Xtrender × RSI) from the first 70% of history, with statistical shrinkage — it only acts on states where evidence clears a hurdle. This is the *contextual-bandit* form of reinforcement learning: since our tiny orders don't move the market, estimating the conditional edge IS the optimal policy — and unlike deep RL, it can't hallucinate patterns the data can't support.

**Why the honesty obsession:** TradeMaster (NeurIPS 2023) and its PRUDEX-Compass benchmark exist because most published FinRL results don't survive out-of-sample testing. So this lab shows you ONLY out-of-sample performance, the full learned policy with sample counts and t-stats, and the compass comparison vs plain buy & hold. If the agent doesn't beat B&H on your ticker — that's the data talking, believe it.

**Market dynamics strip:** the last 2 years labeled into 5 styles. Agents (and humans) trained mostly on bull data will be over-optimistic in bears — check what diet your agent grew up on.
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


st.markdown("---")
st.caption("QuantSignal v25 · data: Yahoo Finance (delayed) · educational "
           "tool, not financial advice · every model documented in its ❓ "
           "expander · built for one very persistent trader 🇮🇱")
