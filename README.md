# 🧠 NEXT LEVEL TRADING SYSTEM

> **Automated grid & ICT/SMC trading bot for MetaTrader 5 — by Aleem Shahzad**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![MT5](https://img.shields.io/badge/Platform-MetaTrader%205-orange)](https://metatrader5.com)
[![Broker](https://img.shields.io/badge/Broker-Exness-green)](https://exness.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 What Does This Bot Do?

NEXT LEVEL TRADING SYSTEM is a professional automated trading bot that connects to MetaTrader 5 and executes a **dynamic grid strategy** on real market data. It does **not** use any mocked, simulated, or fake data — all prices, fills, and account information come directly from your live MT5 terminal.

| Feature | Details |
|---|---|
| **Strategy** | Smart Trailing Grid + ICT/SMC Trend Following |
| **Execution** | Real MT5 orders via `MetaTrader5` Python API |
| **Loop Speed** | 0.1-second cycle (10 checks per second) |
| **Supported Pairs** | 18 Exness pairs — Forex, Metals, Crypto, Indices, Oil |
| **Risk Control** | Per-trade trailing with profit-lock tiers |

---

## ⚙️ How It Works

### 1. System Initialization
- Connects to your MT5 terminal using credentials from `.env`
- Reads lot size and spacing settings from `config.yaml`
- Initializes a shared `MT5Broker` connection for stable, persistent access

### 2. Direction Modes
Choose how you want to trade at startup:

| Mode | Description |
|---|---|
| **BUY ONLY** | Places buy-limit orders below current price |
| **SELL ONLY** | Places sell-limit orders above current price |
| **BOTH (Hedging)** | Places grids on both sides simultaneously |
| **ICT SMC** | Uses Silver Bullet windows, FVG, OB, and liquidity sweeps |

### 3. Grid Placement
- Places **20 orders per batch** at defined spacing intervals
- When **15 orders** from a batch are filled, the next batch is prepared automatically
- Far-away orders are pruned; new orders roll with the market

### 4. 0.1-Second Trailing Loop
- Checks every filled position 10 times per second
- Applies **individual trailing** per trade — not basket-level
- When a trade hits its lock level, it closes and the level is **immediately recycled** with a fresh limit order

### 5. Smart Profit Locking

| Floating Profit | Lock Amount |
|---|---|
| $1.00 | $0.50 locked |
| $25.00+ | 80% of floating profit locked |

---

## 🪙 Supported Pairs (Exness)

```
Forex:   EURUSDm  GBPUSDm  USDJPYm  USDCHFm  AUDUSDm  NZDUSDm  USDCADm
         EURGBPm  EURJPYm  GBPJPYm
Metals:  XAUUSDm (Gold)    XAGUSDm (Silver)
Crypto:  BTCUSDm           ETHUSDm
Energy:  USOILm (WTI)      UKOILm (Brent)
Indices: NASDAQ            SP500m
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Edit `.env`:
```
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=Exness-MT5Trial15
```

### 3. Run the Bot
```bash
python live_trading.py
```

Or use the desktop GUI:
```bash
python brain_app.py
```

### 4. Run Backtesting (on real historical MT5 data)
```bash
python backtesting.py
```

---

## 📁 File Structure

```
NEXT-LEVEL-TRADING-SYSTEM/
├── live_trading.py        # Main live trading loop (CLI)
├── brain_app.py           # Desktop GUI (CustomTkinter)
├── backtesting.py         # Historical backtesting engine
├── grid_recycler.py       # Grid order management module
├── smart_trailing.py      # Trailing stop logic
├── mt5_broker.py          # MT5 connection wrapper
├── profit_controller.py   # Profit-lock and milestone tracker
├── dashboard.py           # Streamlit web dashboard
├── live_dashboard.py      # Lightweight live stats window
├── config.yaml            # Bot configuration
├── .env                   # MT5 credentials (never commit)
├── logs/                  # Runtime logs and session reports
├── models/                # Saved AI trade memories
├── backtest_results/      # Historical backtest output (JSON)
└── charts/                # Generated HTML charts
```

---

## ⚠️ Risk Disclaimer

> Trading involves a **significant risk of financial loss**. This software places real orders on a live trading account. Test on a **demo account** before using with real funds. Past grid performance does not guarantee future results.

---

**© 2026 Aleem Shahzad — NEXT LEVEL TRADING SYSTEM**
