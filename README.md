# 🧠 NEXT LEVEL TRADING SYSTEM - Professional AI Trading System

**Created by: Aleem Shahzad** | **AI Partner: Claude (Anthropic)**

A comprehensive, state-of-the-art AI trading ecosystem integrating **Smart Money Concepts (ICT/SMC)**, **High-Frequency Dynamic Grids**, and **Autonomous Market Intelligence**.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **MetaTrader 5 Terminal** (installed and logged in)
- **Windows OS** (required for MT5 Python integration)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and fill in your credentials:
```env
# MT5 Credentials
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server_name
MT5_TERMINAL_PATH=C:/Program Files/.../terminal64.exe

# AI Intelligence (Groq is recommended for speed)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxx
```

---

## 🎮 Desktop Command Center
Run the main GUI to manage all systems from a single dashboard:
```bash
python brain_app.py
```
*Features: Account management, Backtesting GUI, Live Trading launcher, Intelligence reports, and Real-time logs.*

---

## 📊 Trading Strategies

### 1. 🧠 ICT / Smart Money Concepts (SMC)
A precision-based institutional strategy that analyzes:
- **Liquidity Sweeps**: Detects stop-hunts below old lows or above old highs.
- **Fair Value Gaps (FVG)**: Identifies price displacement and imbalance.
- **Order Blocks**: Locates areas of institutional buying/selling interest.
- **Silver Bullet Windows**: Optimized for NY/London high-volatility hours (3-4 AM, 10-11 AM, 2-3 PM EST).
- **AI Brain**: A neural-inspired memory system that learns from past trade successes/failures.

### 2. 🛡️ Dynamic Grid & Smart Trailing
A robust volatility-harvesting engine:
- **Modes**: Choose between `BUY ONLY`, `SELL ONLY`, or `BOTH`.
- **Grid Recycler**: Automatically replaces levels as soon as they close, creating a "perpetual" profit machine.
- **Smart Trailing ($10/$20)**: Each grid level has its own trailing logic. Once a trade hits +$10, a trailing stop is activated to lock in profit, aiming for $20+ while protecting the baseline.
- **Dynamic Expansion**: The grid follows the market move, rolling levels forward to stay in the "Golden Zone."

### 3. 🌐 Autonomous Market Intelligence
Standalone sentiment engine that provides a macro "filter" for the technical strategies:
- **Crawl**: Scrapes Twitter, Reddit, and Macro News.
- **Analyze**: Uses Large Language Models (LLMs) to detect institutional bias.
- **Signal**: Outputs `ALLOW`, `REDUCE`, or `BLOCK` to the trading system.

---

## 📁 Core System Maps

```
NEXT LEVEL TRADING SYSTEM/
├── brain_app.py            # Desktop GUI Controller (Main Entry)
├── live_trading.py         # Advanced CLI Trading Engine
├── backtesting.py          # Pro Strategy Tester & AI Trainer
├── run_market_intelligence.py # Sentiment Analysis Launcher
│
├── smart_trailing.py       # Individual Trade Protection Logic
├── grid_recycler.py        # High-Frequency Level Management
├── mt5_broker.py           # Robust MT5 Bridge & Reconnect Watchdog
├── profit_controller.py    # Global P&L & Milestone Management
│
├── .env                    # Secure Credentials Store
├── config.yaml             # Strategy & Symbol Parameters
└── models/                 # AI Memories & Trained Weights
```

---

## 🛠️ Key Components

- **Watchdog Protection**: `mt5_broker.py` includes an auto-reconnect system that handles "IPC Failed" errors and terminal crashes.
- **Profit Milestone**: Automatically tracks equity growth and syncs baselines to lock in daily gains.
- **Visual Analytics**: `live_dashboard.py` provides a real-time web tracker for your balance, equity, and open positions.

---

## ⚠️ Disclaimer
This software is for **educational and research purposes only**. Trading financial markets involves significant risk. The developers are not responsible for any financial losses. Always test in **Demo Accounts** before committing real capital.

---
**© 2026 Aleem Shahzad - Next Level TRADING SYSTEM Ecosystem**
