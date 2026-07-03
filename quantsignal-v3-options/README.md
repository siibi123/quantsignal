# QuantSignal 📈

A free, open quant screener + signal engine + backtester for US stocks.
Built with Python + Streamlit. Data from Yahoo Finance (free, 1h cache).

## What it does

- **Screener** — scans 50 liquid US stocks/ETFs (or any list you type) and scores each one from −1 to +1 using six models: trend (SMA50/200), momentum (12-1), MACD, RSI regime, Bollinger mean-reversion, and volume confirmation — dampened by a volatility-regime filter. Score ≥ +0.25 → BUY, ≤ −0.25 → SELL.
- **Ticker analysis** — candlestick chart with historical BUY/SELL zones, score history, and a breakdown showing *which* model drives the current signal.
- **Backtest** — simulates the signal with next-bar-open execution (no look-ahead), commissions, ATR trailing stops, and risk-based position sizing. Reports CAGR, Sharpe, Sortino, max drawdown, win rate vs buy & hold, plus a **walk-forward table** (4 sequential periods) so you can spot overfitting.
- **Position size** — tells you how many shares to buy so a stop-out costs only 1% of your account.

## ⚠️ Honest warning

This is an educational tool, **not financial advice**. Backtests overestimate real results. Most signals lose to buy-and-hold after costs in strong bull markets. The value of this tool is *discipline*: defined entries, defined exits, defined risk.

## Run it on your computer (5 minutes)

1. Install Python 3.10+ from python.org (check "Add to PATH" on Windows).
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Your browser opens at http://localhost:8501 — done.

## Make it a public website for everyone (free, ~10 minutes)

1. Create a free account at **github.com**.
2. Click **New repository** → name it `quantsignal` → Public → Create.
3. Click **uploading an existing file** and drag in: `app.py`, `requirements.txt`, `README.md`, and the whole `quant/` folder (upload its 3 `.py` files into a folder named `quant`). Commit.
4. Go to **share.streamlit.io** → sign in with GitHub → **New app** → pick your `quantsignal` repo → main file: `app.py` → **Deploy**.
5. In ~2 minutes you get a public URL like `https://quantsignal.streamlit.app` — anyone in the world can use it. Free tier is enough for this app.

## Files

```
app.py              ← the website (Streamlit UI)
quant/data.py       ← Yahoo Finance downloads + caching
quant/signals.py    ← the 6 models + composite score
quant/backtest.py   ← backtester + walk-forward
requirements.txt    ← dependencies
```

## Ideas for v2 (ask Claude to add them)

- Fundamental filters (P/E, revenue growth) via yfinance `.info`
- HMM regime detection / GARCH vol forecast (you already explored these!)
- Email/Telegram alerts when a ticker flips to BUY
- Portfolio-level backtest (hold top-5 scores, rebalance weekly)
