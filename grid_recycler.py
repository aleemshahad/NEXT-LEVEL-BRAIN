import logging
import asyncio
import time
import MetaTrader5 as mt5
from pathlib import Path
from typing import Dict, List, Optional
from profit_controller import ProfitController

logger = logging.getLogger(__name__)


class GridRecycler:
    """
    Grid Recycler Strategy:
    - Places a grid of pending orders at fixed levels below (BUY) or above (SELL) market.
    - Monitors individual open positions.
    - When a position hits the per-trade profit target, it is closed.
    - A fresh pending order is instantly placed at the same price level.
    - Result: Infinite recycling of profits on small market oscillations.
    """

    def __init__(self, broker, config: Dict):
        self.broker = broker
        
        grid_cfg = config.get('grid', {})
        self.spacing = grid_cfg.get('spacing', 1.0)          # Distance between levels ($)
        self.lot_size = grid_cfg.get('lot_size', 0.01)
        self.batch_size = 20                                 # Orders per batch
        self.trigger_threshold = 5                           # Expand if <= 5 pendings left
        self.total_limit = 1000                              # Absolute max levels
        self.per_trade_profit = config.get('recycler_profit_usd', 1.0)  # $ profit per trade
        
        self.magic_buy = 20001
        self.magic_sell = 20002
        self.mode = "BUY_ONLY"   # BUY_ONLY / SELL_ONLY / BOTH
        
        self.profit_ctrl = ProfitController(broker, "Recycler")
        
        self._last_log_time = 0
        self.state_file = Path("logs/recycler_state.json")
        self.load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    async def place_batch(self, symbol: str, base_price: float, side: str, start_index: int, count: int):
        """Place a batch of pending orders."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found for recycler grid.")
            return 0

        placed = 0
        for i in range(1, count + 1):
            k = start_index + i
            level_price = base_price - (k * self.spacing) if side == 'BUY' else base_price + (k * self.spacing)

            # Normalize price
            tick_size = getattr(symbol_info, 'trade_tick_size', 0.0)
            if tick_size > 0:
                level_price = round(round(level_price / tick_size) * tick_size, symbol_info.digits)

            order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT
            magic = self.magic_buy if side == 'BUY' else self.magic_sell
            
            res = await self.broker.place_pending_order(symbol, order_type, self.lot_size, level_price, magic)
            if res.get('success'):
                placed += 1
            else:
                # If we fail (e.g. too close or limit reached), stop batching
                logger.warning(f"Batch placement stopped at index {k}: {res.get('error')}")
                break
        
        return placed

    async def initialize_grid(self, symbol: str, current_price: float):
        """Initialize the starting grid around the current price."""
        # Cleanup existing first
        await self.broker.cancel_all_pendings(symbol, self.magic_buy)
        await self.broker.cancel_all_pendings(symbol, self.magic_sell)
        
        self.active_grids = {
            symbol: {
                'BUY': {'base_price': current_price, 'first_index': 1, 'last_index': 0},
                'SELL': {'base_price': current_price, 'first_index': 1, 'last_index': 0}
            }
        }
        
        if self.mode in ("BUY_ONLY", "BOTH"):
            placed = await self.place_batch(symbol, current_price, 'BUY', 0, self.batch_size)
            self.active_grids[symbol]['BUY']['last_index'] = placed
            
        if self.mode in ("SELL_ONLY", "BOTH"):
            placed = await self.place_batch(symbol, current_price, 'SELL', 0, self.batch_size)
            self.active_grids[symbol]['SELL']['last_index'] = placed
            
        self.save_state()
        logger.info(f"🚀 Recycler Initialized for {symbol} | Mode: {self.mode} | Spacing: {self.spacing}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core Update Loop
    # ─────────────────────────────────────────────────────────────────────────

    async def update(self, symbol: str, current_price: float):
        """Monitor positions and manage the grid levels."""
        try:
            if not hasattr(self, 'active_grids') or symbol not in self.active_grids:
                await self.initialize_grid(symbol, current_price)
                return

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return

            # 1. Get current state from Broker
            all_positions = self.broker.get_positions()
            recycler_positions = [p for p in all_positions if p['symbol'] == symbol and p['magic'] in (self.magic_buy, self.magic_sell)]
            
            # Map items to MT5 objects for individual monitor_individual_trailing
            # Actually ProfitController uses raw positions from mt5.positions_get internally if we want, 
            # but monitor_individual_trailing takes a list of positions.
            # Convert dict positions back to objects or just use mt5.positions_get for the update logic.
            raw_positions = mt5.positions_get(symbol=symbol)
            if raw_positions is None: raw_positions = []
            recycler_objs = [p for p in raw_positions if p.magic in (self.magic_buy, self.magic_sell)]

            all_orders = mt5.orders_get(symbol=symbol)
            if all_orders is None: all_orders = []
            active_pendings = [o for o in all_orders if o.magic in (self.magic_buy, self.magic_sell)]

            # 2. Maintain Grid Depth (Expansion & Rolling)
            for side in ['BUY', 'SELL']:
                if self.mode != "BOTH" and side != self.mode.replace("_ONLY", ""):
                    continue
                    
                progress = self.active_grids[symbol][side]
                side_pendings = [o for o in active_pendings if (o.magic == self.magic_buy if side == 'BUY' else o.magic == self.magic_sell)]
                
                first_idx = progress.get('first_index', 1)
                last_idx = progress.get('last_index', 1)
                base_p = progress['base_price']
                
                # Check for "Rolling" towards Market (Add orders near current price)
                dist_to_front = abs(current_price - (base_p - (first_idx * self.spacing) if side == 'BUY' else base_p + (first_idx * self.spacing)))
                
                if dist_to_front > (1.1 * self.spacing):
                    slots = self.broker.get_available_slots()
                    if slots < 10:
                        await self._prune_furthest(symbol, side, active_pendings)
                    
                    new_idx = first_idx - 1
                    placed = await self.place_batch(symbol, base_p, side, new_idx - 1, 1)
                    if placed:
                        progress['first_index'] = new_idx
                        logger.info(f"🚜 Recycler Rolling: Added new {side} order near market. First Index is now {new_idx}")
                        self.save_state()

                # Normal Batch Expansion (Deep side)
                if len(side_pendings) <= self.trigger_threshold and (last_idx - first_idx) < self.total_limit:
                    slots = self.broker.get_available_slots()
                    num_to_place = min(self.batch_size, self.total_limit - (last_idx - first_idx), slots - 5)
                    if num_to_place > 0:
                        logger.info(f"🔄 Recycler Expansion: Adding {num_to_place} {side} pendings to depth...")
                        placed = await self.place_batch(symbol, base_p, side, last_idx, num_to_place)
                        progress['last_index'] += placed
                        self.save_state()

            # 3. Check each position for profit target & trailing via ProfitController
            to_close = await self.profit_ctrl.monitor_individual_trailing(recycler_objs, self.per_trade_profit)
            
            for pos in to_close:
                side = 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL'
                entry_price = pos.price_open
                
                logger.info(f"💰 Recycler: {side} @ {entry_price:.3f} trailing closed via ProfitController.")
                
                # Close the position
                closed = await self._close_position(symbol, pos)
                if closed:
                    await asyncio.sleep(0.2)
                    await self._recycle_level(symbol, entry_price, side, symbol_info)
                
            # Periodic logging for Recycler status
            now = time.time()
            if now - self._last_log_time > 10:
                open_count = len(recycler_objs)
                total_pnl = sum(p.profit + getattr(p, 'commission', 0.0) + p.swap for p in recycler_objs)
                if open_count > 0:
                    logger.info(f"🔄 Recycler Status | Symbols: {symbol} | Open: {open_count} | Net PnL: ${total_pnl:.2f}")
                self._last_log_time = now

        except Exception as e:
            logger.error(f"Recycler update error: {e}")

    async def _close_position(self, symbol: str, pos) -> bool:
        """Close an individual position using the broker's resilient method."""
        try:
            res = await self.broker.close_position(pos.ticket)
            return res.get('success', False)
        except Exception as e:
            logger.error(f"Error closing Recycler position: {e}")
            return False

    async def _recycle_level(self, symbol: str, price: float, side: str, symbol_info):
        """Immediately re-place a pending order at the level that just closed."""
        try:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT
            magic = self.magic_buy if side == 'BUY' else self.magic_sell
            
            # Normalize price
            tick_size = getattr(symbol_info, 'trade_tick_size', 0.0)
            if tick_size > 0:
                price = round(round(price / tick_size) * tick_size, symbol_info.digits)

            # Check if price is still valid (not too close to market)
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                # If market moved past this level, we can't place a limit. 
                # For recycler, we just try. If fail, it will be picked up by the rolling logic eventually.
                res = await self.broker.place_pending_order(symbol, order_type, self.lot_size, price, magic)
                if res.get('success'):
                    logger.info(f"♻️ Level Recycled: {side} Limit replaced at {price:.3f}")
                else:
                    logger.warning(f"Failed to recycle level at {price:.3f}: {res.get('error')}")
                    
        except Exception as e:
            logger.error(f"Error recycling level: {e}")

    async def _prune_furthest(self, symbol: str, side: str, all_pendings: list):
        """Cancel the pending order that is furthest from the current market price."""
        try:
            side_orders = [o for o in all_pendings if o.magic == (self.magic_buy if side == 'BUY' else self.magic_sell)]
            if not side_orders: return
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick: return
            mkt = tick.ask
            
            # Sort by distance from market (descending)
            furthest = sorted(side_orders, key=lambda x: abs(x.price - mkt), reverse=True)[0]
            
            if await self.broker.cancel_order(furthest.ticket):
                # Update our tracking if possible
                progress = self.active_grids[symbol][side]
                # If we pruned the 'last' one
                if furthest.price == (progress['base_price'] - (progress['last_index'] * self.spacing) if side == 'BUY' else progress['base_price'] + (progress['last_index'] * self.spacing)):
                     progress['last_index'] -= 1
                
                logger.info(f"✂️ Pruned furthest {side} order at {furthest.price:.3f} to save slots.")
        except Exception as e:
            logger.error(f"Error pruning: {e}")

    def save_state(self):
        """Save the current grid state."""
        import json
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'active_grids': self.active_grids
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving recycler state: {e}")

    def load_state(self):
        """Load grid state from file."""
        import json
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.active_grids = data.get('active_grids', {})
            except Exception as e:
                logger.error(f"Error loading recycler state: {e}")
