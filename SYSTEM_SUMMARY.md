# NEXT LEVEL TRADING SYSTEM - SYSTEM SUMMARY

## 1. System Initialization
- Connects to the MetaTrader 5 (MT5) Terminal.
- Reads configuration from `config.yaml` including parameters such as symbol (e.g., Gold), lot size, and order spacing.
- Implements a shared MT5Broker connection to maintain a continuous data feed.

## 2. Direction Selection
The system provides three operational modes:
- **BUY ONLY**: Executes buy orders.
- **SELL ONLY**: Executes sell orders.
- **BOTH (Hedging)**: Executes a combination of buy and sell order grids simultaneously.

## 3. Grid Placement
- Generates a batch of 20 orders at predefined price levels (spacing).
- Prepares and deploys the next batch of orders when 15 trades from the current batch are activated.
- Modifies and expands grid levels as the market price fluctuates.

## 4. Execution Loop and Trailing
- Executes a main loop running at a 0.1-second interval.
- Applies a trailing stop mechanism that closes individual trades based on locked profit.
- Re-enters orders at the exact same entry price level immediately after a trailing close.

## 5. Smart Trailing and Display
- Executes a locking parameter: at $1 profit, it locks $0.5; at $25 profit or above, it locks 80% of the active profit.
- Terminal outputs are limited to essential operational log events, such as trade closure.
