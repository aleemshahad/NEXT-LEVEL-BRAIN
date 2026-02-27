import MetaTrader5 as mt5
import time
from loguru import logger
from smart_trailing import SmartTrailingHandler
from typing import Optional, List, Dict
import json
from pathlib import Path

class ProfitController:
    """
    Centralized Profit Controller:
    - Paste Close: Close positions when a fixed USD target is hit.
    - Trail Close: Use Smart Trailing to lock profits and exit.
    """
    
    def __init__(self, broker, strategy_name: str = "Controller"):
        self.broker = broker
        self.strategy_name = strategy_name
        self.trailing = SmartTrailingHandler()
        self.ticket_states = {}
        self.grand_basket_state = {'peak': 0.0, 'lock': 0.0}
        self.equity_milestone_state = {'baseline_equity': 0.0}
        self._last_log_time = 0
        
        # Persistence setup
        safe_name = "".join([c if c.isalnum() else "_" for c in strategy_name])
        self.state_file = Path(f"logs/profit_state_{safe_name}.json")
        self._load_state()

    def _load_state(self):
        """Load peaks and locks from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to int tickets
                    self.ticket_states = {int(k): v for k, v in data.get('ticket_states', {}).items()}
                    self.grand_basket_state = data.get('grand_basket_state', {'peak': 0.0, 'lock': 0.0})
                    self.equity_milestone_state = data.get('equity_milestone_state', {'baseline_equity': 0.0})
                    logger.info(f"💾 {self.strategy_name} recovered {len(self.ticket_states)} trade states + Milestone logic from disk.")
        except Exception as e:
            logger.error(f"Failed to load profit state: {e}")

    def _save_state(self, active_tickets: Optional[List[int]] = None):
        """Save peaks and locks to disk. If active_tickets provided, prune closed ones."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prune states for tickets that are no longer active
            if active_tickets is not None:
                self.ticket_states = {k: v for k, v in self.ticket_states.items() if k in active_tickets}

            data = {
                'ticket_states': self.ticket_states,
                'grand_basket_state': self.grand_basket_state,
                'equity_milestone_state': self.equity_milestone_state
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save profit state: {e}")

    async def check_basket_profit(self, symbol: str, buy_magic: int, sell_magic: int, target_usd: float, balance: float) -> bool:
        """
        Check combined profit of BUY and SELL sides and close if target_usd is reached.
        """
        if target_usd <= 0:
            return False

        positions = self.broker.get_positions()
        grid_positions = [p for p in positions if p['symbol'] == symbol and p.get('magic') in (buy_magic, sell_magic)]
        
        if not grid_positions:
            return False

        total_profit = sum(p['profit'] for p in grid_positions)
        
        # Periodic logging
        now = time.time()
        if now - self._last_log_time > 10:
            logger.info(f"📊 {self.strategy_name} | Combined Profit: ${total_profit:.2f} / Target: ${target_usd:.2f}")
            self._last_log_time = now

        if total_profit >= target_usd:
            logger.info(f"🚀 BASKET TARGET HIT: ${total_profit:.2f} >= ${target_usd:.2f}. Closing all {symbol} trades!")
            await self.broker.close_all_side(symbol, 'BUY', buy_magic)
            await self.broker.close_all_side(symbol, 'SELL', sell_magic)
            await self.broker.cancel_all_pendings(symbol, buy_magic)
            await self.broker.cancel_all_pendings(symbol, sell_magic)
            self.trailing.reset('BOTH')
            return True
            
        return False

    async def monitor_trailing(self, symbol: str, buy_magic: int, sell_magic: int) -> dict:
        """
        Monitor BUY and SELL sides independently for trailing profit.
        Returns a dict indicating if closures happened.
        """
        results = {'BUY': False, 'SELL': False}
        
        raw_positions = mt5.positions_get(symbol=symbol)
        if raw_positions is None:
            return results

        # Check BUY side
        buy_pos = [item for item in raw_positions if item.type == mt5.POSITION_TYPE_BUY and item.magic == buy_magic]
        if buy_pos:
            buy_profit = sum(p.profit + getattr(p, 'commission', 0.0) + p.swap for p in buy_pos)
            action = self.trailing.check_profit('BUY', buy_profit)
            if action == 'CLOSE':
                logger.info(f"🛡️ [{self.strategy_name}] BUY Trail Exit hit. Closing all BUY positions!")
                await self.broker.close_all_side(symbol, 'BUY', buy_magic)
                await self.broker.cancel_all_pendings(symbol, buy_magic)
                results['BUY'] = True

        # Check SELL side
        sell_pos = [item for item in raw_positions if item.type == mt5.POSITION_TYPE_SELL and item.magic == sell_magic]
        if sell_pos:
            sell_profit = sum(p.profit + getattr(p, 'commission', 0.0) + p.swap for p in sell_pos)
            action = self.trailing.check_profit('SELL', sell_profit)
            if action == 'CLOSE':
                logger.info(f"🛡️ [{self.strategy_name}] SELL Trail Exit hit. Closing all SELL positions!")
                await self.broker.close_all_side(symbol, 'SELL', sell_magic)
                await self.broker.cancel_all_pendings(symbol, sell_magic)
                results['SELL'] = True
                
        # Status update logs
        now = time.time()
        if now - self._last_log_time > 15:
            buy_lock = self.trailing.get_lock('BUY')
            sell_lock = self.trailing.get_lock('SELL')
            if buy_pos or sell_pos:
                buy_info = f"${buy_profit:.2f} (Lock: ${buy_lock:.2f})" if buy_pos else "Inactive"
                sell_info = f"${sell_profit:.2f} (Lock: ${sell_lock:.2f})" if sell_pos else "Inactive"
                logger.info(f"🔍 [{self.strategy_name}] Trail Trace | BUY: {buy_info} | SELL: {sell_info}")
                self._last_log_time = now

        return results

    async def monitor_individual_trailing(self, positions: list, per_trade_target: float) -> list:
        """
        Monitor individual tickets for trailing profit.
        Thresholds and locks scale proportionally to lot size (base 0.01).
        Returns a list of tickets that should be closed.
        """
        to_close = []
        
        for pos in positions:
            ticket = pos.ticket
            volume = pos.volume
            multiplier = max(1.0, volume / 0.01) # Scale factor based on base 0.01 lot
            
            pnl = pos.profit + getattr(pos, 'commission', 0.0) + pos.swap
            
            if ticket not in self.ticket_states:
                self.ticket_states[ticket] = {'peak': pnl, 'lock': 0.0}
            
            state = self.ticket_states[ticket]
            if pnl > state['peak']: state['peak'] = pnl

            # Scaled Target and Thresholds
            scaled_target = per_trade_target * multiplier
            
            # ── HYBRID LOGIC ─────────────────────────────────────────────
            # 1. Hard Take Profit: Scaled hit → INSTANT close
            if pnl >= scaled_target:
                logger.info(f"💰 Ticket {ticket} Hard TP Hit: PnL ${pnl:.2f} >= ${scaled_target:.2f} (Lot: {volume})")
                to_close.append(pos)
                if ticket in self.ticket_states:
                    del self.ticket_states[ticket]
                continue

            # 2. Micro-Trailing: scaled to lot size
            micro_levels = [
                (0.25 * multiplier, 0.10 * multiplier),
                (0.50 * multiplier, 0.25 * multiplier),
                (0.75 * multiplier, 0.50 * multiplier),
            ]

            new_lock = state['lock']
            if pnl > 0:
                for threshold, lock_val in micro_levels:
                    if state['peak'] >= threshold:
                        if lock_val > new_lock:
                            new_lock = lock_val
                    else:
                        break

            # Floating trail for large movers (Scaled trigger)
            scaled_floating_trigger = 25.0 * multiplier
            if state['peak'] >= scaled_floating_trigger and pnl > 0:
                floating_lock = state['peak'] * 0.8
                if floating_lock > new_lock:
                    new_lock = floating_lock

            state['lock'] = new_lock

            # Exit logic
            if new_lock > 0 and 0 <= pnl < new_lock:
                logger.info(f"💰 Ticket {ticket} Trail Exit: PnL ${pnl:.2f} < Lock ${new_lock:.2f} (Lot: {volume})")
                to_close.append(pos)
                if ticket in self.ticket_states:
                    del self.ticket_states[ticket]

        # Sync states to disk (cleanup closed, save updated)
        active_tickets = [p.ticket for p in positions]
        self._save_state(active_tickets)
        
        return to_close

    async def monitor_grand_basket(self, positions: list, trigger_usd: float = 5.0) -> bool:
        """
        Monitor total profit of ALL active positions across ALL symbols/strategies.
        Starts trailing once combined profit >= trigger_usd ($5.0).
        """
        if not positions:
            if hasattr(self, 'grand_basket_state'):
                self.grand_basket_state = {'peak': 0.0, 'lock': 0.0}
            return False

        if not hasattr(self, 'grand_basket_state'):
            self.grand_basket_state = {'peak': 0.0, 'lock': 0.0}

        total_profit = sum(p['profit'] for p in positions)
        state = self.grand_basket_state

        # Activate trailing when trigger hit
        if total_profit >= trigger_usd:
            if total_profit > state['peak']:
                state['peak'] = total_profit
            
            # Simple 80% lock of peak
            new_lock = state['peak'] * 0.8
            if new_lock > state['lock']:
                state['lock'] = new_lock
                logger.info(f"🛡️  GRAND BASKET LOCK: Total Profit ${total_profit:.2f}. New Lock: ${new_lock:.2f}")

        # Check for trailing exit
        if state['lock'] > 0 and total_profit < state['lock']:
            logger.info(f"🚀 GRAND BASKET EXIT: Combined Profit ${total_profit:.2f} < Lock ${state['lock']:.2f}. Closing Universe!")
            self.grand_basket_state = {'peak': 0.0, 'lock': 0.0}
            self._save_state() # Save reset state
            return True

        # Save state periodically to disk
        if time.time() - getattr(self, '_last_basket_save', 0) > 30:
            self._save_state([p['ticket'] for p in positions if 'ticket' in p])
            self._last_basket_save = time.time()

        return False

    async def monitor_equity_milestone(self, current_equity: float, target_increase: float = 100.0) -> bool:
        """
        Monitor total account equity. If equity increases by target_increase ($100) 
        from the baseline, return True to trigger a global close.
        """
        if current_equity <= 0: return False

        state = self.equity_milestone_state
        if state['baseline_equity'] <= 0:
            state['baseline_equity'] = current_equity
            self._save_state()
            logger.info(f"🎯 Milestone Baseline Set: ${current_equity:.2f}. Target: ${current_equity + target_increase:.2f}")
            return False

        target_equity = state['baseline_equity'] + target_increase
        
        # Periodic log (Every 60 seconds, independent of other monitors)
        now = time.time()
        if now - getattr(self, '_last_milestone_log', 0) > 60:
            diff = current_equity - state['baseline_equity']
            logger.info(f"📊 Milestone Track | Current: ${current_equity:.2f} | Baseline: ${state['baseline_equity']:.2f} | Progress: ${diff:.2f}/$100")
            self._last_milestone_log = now

        if current_equity >= target_equity:
            logger.info(f"🚀 EQUITY MILESTONE HIT! Equity ${current_equity:.2f} >= Target ${target_equity:.2f}. Closing Universe!")
            # Reset will be handled by the caller or specialized method
            return True

        return False

    def reset_equity_milestone(self, current_equity: float):
        """Set a new baseline for the next $100 milestone cycle"""
        if current_equity > 0:
            self.equity_milestone_state['baseline_equity'] = current_equity
            self._save_state()
            logger.info(f"🔄 Milestone Baseline RESET to ${current_equity:.2f}")

    def reset(self, side: str = 'BOTH'):
        self.trailing.reset(side)
        if side == 'BOTH': 
            self.ticket_states = {}
            self.grand_basket_state = {'peak': 0.0, 'lock': 0.0}
            self.equity_milestone_state = {'baseline_equity': 0.0}
            self._save_state()
