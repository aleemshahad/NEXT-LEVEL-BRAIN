import logging
import json
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

class SmartTrailingHandler:
    """Handles Multi-Stage Smart Trailing Profit with progressive locking."""
    
    def __init__(self, state_file: str = "logs/smart_trailing_state.json"):
        self.state_file = Path(state_file)
        
        # Define levels: (Threshold, LockValue)
        self.levels = [
            (1.0, 0.5),   # Level 1: $0.5 lock
            (1.5, 1.0),   # Level 2: $1 lock
            (2.0, 1.5),   # Level 3: $1.5 lock
            (5.0, 3.5),   # Level 4: $3.5 lock
            (10.0, 7.5),  # Level 5: $7.5 lock
            (20.0, 16.5)  # Level 6: $16.5 lock
        ]
        
        # Beyond $25: We trail at 80% of Peak Profit
        self.floating_trail_pct = 0.80
        self.floating_trigger = 25.0
        
        # State for each side
        self.states = {
            'BUY': {'current_lock': 0.0},
            'SELL': {'current_lock': 0.0}
        }
        self.load_state()

    def check_profit(self, side: str, profit: float) -> str:
        """
        Check profit against multi-stage trailing logic.
        Returns:
            'CLOSE': If target hit or trailing exit (profit < lock).
            'LOCK_UPDATED': If lock level increased.
            'NONE': Otherwise.
        """
        side = side.upper()
        if side not in self.states:
            return 'NONE'
            
        state = self.states[side]
        current_lock = state['current_lock']

        # 1. Update Absolute Peak (for floating trail)
        if 'peak' not in state: state['peak'] = profit
        if profit > state.get('peak', 0.0):
            state['peak'] = profit

        # 2. Check for Progressive Lock Updates

        # 2. Check for Lock Update (Aggressive Upgrade)
        new_lock = current_lock
        for threshold, lock_val in self.levels:
            if profit >= threshold:
                if lock_val > new_lock:
                    new_lock = lock_val
            else:
                break # Thresholds are ordered

        if new_lock > current_lock:
            state['current_lock'] = new_lock
            current_lock = new_lock # Update local for subsequent checks
            logger.info(f"🛡️ [{side}] LOCK UPDATED: Profit ${profit:.2f}. New Lock: ${new_lock:.2f}")
            self.save_state()

        # 3. Floating Trail (Unlimited Profit Capture)
        if profit >= self.floating_trigger:
            peak = state.get('peak', profit)
            floating_lock = peak * self.floating_trail_pct
            if floating_lock > state['current_lock']:
                state['current_lock'] = floating_lock
                self.save_state()

        # 4. Trailing Exit (Profit falls below current lock)
        # We only check exit if current_lock is active (> 0)
        if current_lock > 0 and profit < current_lock:
            logger.info(f"📉 [{side}] TRAILING EXIT: Profit dropped to ${profit:.2f} (Lock: ${current_lock:.2f}). Closing!")
            self.reset(side)
            return 'CLOSE'

        return 'NONE'

    def get_lock(self, side: str) -> float:
        return self.states.get(side.upper(), {}).get('current_lock', 0.0)

    def is_in_trail(self, side: str) -> bool:
        """Backward compatibility for existing logging in GridManager"""
        return self.get_lock(side) > 0

    def reset(self, side: str = 'BOTH'):
        """Reset trailing state for one or both sides."""
        if side == 'BOTH':
            self.states['BUY']['current_lock'] = 0.0
            self.states['SELL']['current_lock'] = 0.0
        elif side.upper() in self.states:
            self.states[side.upper()]['current_lock'] = 0.0
        self.save_state()

    def save_state(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.states, f)
        except Exception as e:
            logger.error(f"Error saving smart trailing state: {e}")

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for side in self.states:
                        if side in data:
                            # Handle migration from old 'in_trail' schema if exists
                            if 'in_trail' in data[side] and 'current_lock' not in data[side]:
                                # If it was trailing before, assign it the $10 level lock as a starting point
                                if data[side]['in_trail']:
                                    self.states[side]['current_lock'] = 7.0
                            else:
                                self.states[side]['current_lock'] = data[side].get('current_lock', 0.0)
            except Exception as e:
                logger.error(f"Error loading smart trailing state: {e}")
