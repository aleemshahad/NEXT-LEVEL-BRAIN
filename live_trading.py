#!/usr/bin/env python3
"""
NEXT LEVEL BRAIN - Live Trading System (CLI only)
All-in-one live trading with AI enhancement
Created by: Aleem Shahzad | AI Partner: Claude (Anthropic)
"""

import asyncio
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys
import signal
import threading
import time
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("logs/live_trading_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")

class TradingBrain:
    """AI Trading Brain with Neural Network"""
    
    def __init__(self):
        self.memories = []
        self.model_trained = False
        self.confidence_threshold = 0.6
        
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """ICT/SMC AI market analysis"""
        try:
            if len(data) < 50:
                return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'Insufficient data'}
            
            # Add technical indicators
            data = self._add_indicators(data)
            index = len(data) - 1  # Current bar index
            
            # Get market structure analysis
            market_bias = self._determine_market_bias(data, index)
            if market_bias == 'NEUTRAL':
                return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'No clear market bias'}
            
            # Check for liquidity sweeps
            liquidity_sweep = self._detect_liquidity_sweep(data, index)
            
            # Check for displacement and FVG
            fvg = self._detect_fair_value_gap(data, index)
            
            # Check dealing range and discount/premium zones
            dealing_range = self._analyze_dealing_range(data, index)
            
            # Check for order blocks
            order_block = self._detect_order_block(data, index, market_bias)
            
            # Check OTE (Optimal Trade Entry) levels
            ote_level = self._check_ote_levels(data, index, fvg)
            
            current = data.iloc[index]
            
            # Calculate confluence score for available signals
            signals_present = []
            signal_strengths = []
            
            if liquidity_sweep['detected']:
                signals_present.append(f"Liquidity Sweep ({liquidity_sweep['type']})")
                signal_strengths.append(liquidity_sweep['strength'])
            
            if fvg['detected']:
                signals_present.append(f"FVG ({fvg['type']})")
                signal_strengths.append(fvg['strength'])
            
            if order_block['detected']:
                signals_present.append(f"Order Block ({order_block['type']})")
                signal_strengths.append(order_block['strength'])
            
            # BULLISH SETUP - Need at least 2 of 3 main signals
            if market_bias == 'BULLISH':
                bullish_conditions = 0
                
                if liquidity_sweep.get('type') == 'BELOW_LOW':
                    bullish_conditions += 1
                if fvg.get('type') == 'BULLISH':
                    bullish_conditions += 1
                if order_block.get('type') == 'BULLISH':
                    bullish_conditions += 1
                if dealing_range['zone'] == 'DISCOUNT':
                    bullish_conditions += 0.5
                if ote_level['valid']:
                    bullish_conditions += 0.5
                
                if bullish_conditions >= 2.0:
                    confidence = self._calculate_confluence_score(signal_strengths)
                    
                    if liquidity_sweep['detected']:
                        stop_loss = liquidity_sweep['swept_level'] - (current['close'] * 0.001)
                    else:
                        stop_loss = current['close'] * 0.98
                    
                    return {
                        'action': 'BUY',
                        'confidence': confidence,
                        'reasoning': f'ICT Bullish: {", ".join(signals_present)} (Score: {bullish_conditions:.1f})',
                        'entry_price': current['close'],
                        'stop_loss': stop_loss,
                        'take_profit': self._find_next_liquidity_pool(data, index, 'UP')
                    }
            
            # BEARISH SETUP - Need at least 2 of 3 main signals
            elif market_bias == 'BEARISH':
                bearish_conditions = 0
                
                if liquidity_sweep.get('type') == 'ABOVE_HIGH':
                    bearish_conditions += 1
                if fvg.get('type') == 'BEARISH':
                    bearish_conditions += 1
                if order_block.get('type') == 'BEARISH':
                    bearish_conditions += 1
                if dealing_range['zone'] == 'PREMIUM':
                    bearish_conditions += 0.5
                if ote_level['valid']:
                    bearish_conditions += 0.5
                
                if bearish_conditions >= 2.0:
                    confidence = self._calculate_confluence_score(signal_strengths)
                    
                    if liquidity_sweep['detected']:
                        stop_loss = liquidity_sweep['swept_level'] + (current['close'] * 0.001)
                    else:
                        stop_loss = current['close'] * 1.02
                    
                    return {
                        'action': 'SELL',
                        'confidence': confidence,
                        'reasoning': f'ICT Bearish: {", ".join(signals_present)} (Score: {bearish_conditions:.1f})',
                        'entry_price': current['close'],
                        'stop_loss': stop_loss,
                        'take_profit': self._find_next_liquidity_pool(data, index, 'DOWN')
                    }
            
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'ICT conditions not fully aligned'}
                
        except Exception as e:
            logger.error(f"ICT AI analysis error: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reasoning': 'Analysis failed'}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def _determine_market_bias(self, df: pd.DataFrame, index: int) -> str:
        """Determine market bias using MSS (Market Structure Shift)"""
        try:
            lookback = 50  # More lookback for M5 timeframe
            if index < lookback:
                return 'NEUTRAL'
            
            recent_data = df.iloc[index-lookback:index+1]
            highs = recent_data['high'].rolling(3, center=True).max()  # Smaller window for M5
            lows = recent_data['low'].rolling(3, center=True).min()
            current_price = df.iloc[index]['close']
            
            recent_high = highs.max()
            recent_low = lows.min()
            
            if current_price > recent_high * 0.9995:  # More sensitive for M5
                return 'BULLISH'
            elif current_price < recent_low * 1.0005:  # More sensitive for M5
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect liquidity sweeps below lows or above highs"""
        try:
            lookback = 20  # More lookback for M5 timeframe
            if index < lookback:
                return {'detected': False}
            
            current = df.iloc[index]
            recent_data = df.iloc[index-lookback:index]
            swing_low = recent_data['low'].min()
            swing_high = recent_data['high'].max()
            
            if current['low'] < swing_low and current['close'] > swing_low:
                return {'detected': True, 'type': 'BELOW_LOW', 'swept_level': swing_low, 'strength': 0.8}
            elif current['high'] > swing_high and current['close'] < swing_high:
                return {'detected': True, 'type': 'ABOVE_HIGH', 'swept_level': swing_high, 'strength': 0.8}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _detect_fair_value_gap(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect Fair Value Gaps (FVG)"""
        try:
            if index < 3:
                return {'detected': False}
            
            bar1 = df.iloc[index-2]
            bar2 = df.iloc[index-1]
            bar3 = df.iloc[index]
            
            if bar1['high'] < bar3['low']:
                gap_size = bar3['low'] - bar1['high']
                if gap_size > (bar2['close'] * 0.0005):
                    return {'detected': True, 'type': 'BULLISH', 'high': bar3['low'], 'low': bar1['high'], 'strength': min(gap_size / (bar2['close'] * 0.002), 1.0)}
            elif bar1['low'] > bar3['high']:
                gap_size = bar1['low'] - bar3['high']
                if gap_size > (bar2['close'] * 0.0005):
                    return {'detected': True, 'type': 'BEARISH', 'high': bar1['low'], 'low': bar3['high'], 'strength': min(gap_size / (bar2['close'] * 0.002), 1.0)}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _analyze_dealing_range(self, df: pd.DataFrame, index: int) -> Dict:
        """Analyze if price is in discount or premium zone"""
        try:
            lookback = 20
            if index < lookback:
                return {'zone': 'NEUTRAL'}
            
            recent_data = df.iloc[index-lookback:index+1]
            range_high = recent_data['high'].max()
            range_low = recent_data['low'].min()
            current_price = df.iloc[index]['close']
            range_50 = range_low + (range_high - range_low) * 0.5
            
            if current_price < range_50:
                return {'zone': 'DISCOUNT', 'level': range_50}
            else:
                return {'zone': 'PREMIUM', 'level': range_50}
        except Exception:
            return {'zone': 'NEUTRAL'}
    
    def _detect_order_block(self, df: pd.DataFrame, index: int, bias: str) -> Dict:
        """Detect institutional order blocks"""
        try:
            lookback = 15
            if index < lookback:
                return {'detected': False}
            
            for i in range(index-lookback, index-1):
                bar = df.iloc[i]
                next_bar = df.iloc[i+1]
                price_change = abs(next_bar['close'] - bar['close']) / bar['close']
                
                if bias == 'BULLISH' and price_change > 0.002 and next_bar['close'] > bar['close']:  # 0.2% for M5
                    return {'detected': True, 'type': 'BULLISH', 'high': bar['high'], 'low': bar['low'], 'strength': min(price_change * 10, 1.0)}
                elif bias == 'BEARISH' and price_change > 0.002 and next_bar['close'] < bar['close']:  # 0.2% for M5
                    return {'detected': True, 'type': 'BEARISH', 'high': bar['high'], 'low': bar['low'], 'strength': min(price_change * 10, 1.0)}
            
            return {'detected': False}
        except Exception:
            return {'detected': False}
    
    def _check_ote_levels(self, df: pd.DataFrame, index: int, fvg: Dict) -> Dict:
        """Check Optimal Trade Entry levels (62%-79% Fibonacci)"""
        try:
            if not fvg.get('detected'):
                return {'valid': False}
            
            current_price = df.iloc[index]['close']
            fvg_range = fvg['high'] - fvg['low']
            ote_62 = fvg['low'] + (fvg_range * 0.62)
            ote_79 = fvg['low'] + (fvg_range * 0.79)
            
            if ote_62 <= current_price <= ote_79:
                return {'valid': True, 'strength': 0.9, 'level_62': ote_62, 'level_79': ote_79}
            
            return {'valid': False}
        except Exception:
            return {'valid': False}
    
    def _find_next_liquidity_pool(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """Find next liquidity pool for take profit"""
        try:
            lookback = 30
            current_price = df.iloc[index]['close']
            
            if direction == 'UP':
                recent_highs = df.iloc[max(0, index-lookback):index]['high']
                resistance = recent_highs[recent_highs > current_price].min()
                return resistance if not pd.isna(resistance) else current_price * 1.02
            else:
                recent_lows = df.iloc[max(0, index-lookback):index]['low']
                support = recent_lows[recent_lows < current_price].max()
                return support if not pd.isna(support) else current_price * 0.98
        except Exception:
            return current_price * (1.02 if direction == 'UP' else 0.98)
    
    def _calculate_confluence_score(self, strengths: List[float]) -> float:
        """Calculate confluence score from multiple signal strengths"""
        try:
            if not strengths:
                return 0.0
            base_score = sum(strengths) / len(strengths)
            confluence_bonus = min(len(strengths) * 0.1, 0.3)
            return min(base_score + confluence_bonus, 1.0)
        except Exception:
            return 0.0

    def remember_trade(self, trade_data: Dict):
        """Store trade in memory for learning"""
        self.memories.append({
            'timestamp': datetime.now(),
            'symbol': trade_data.get('symbol'),
            'action': trade_data.get('action'),
            'success': trade_data.get('pnl', 0) > 0,
            'pnl': trade_data.get('pnl', 0)
        })
        
        # Keep only last 1000 memories
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]

class ICTAnalyzer:
    """ICT/SMC Strategy Implementation"""
    
    def __init__(self):
        self.lookback_periods = 20
        
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure for BOS/ChoCH"""
        try:
            highs = data['high'].rolling(self.lookback_periods).max()
            lows = data['low'].rolling(self.lookback_periods).min()
            
            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]
            prev_high = highs.iloc[-2] if len(highs) > 1 else current_high
            prev_low = lows.iloc[-2] if len(lows) > 1 else current_low
            
            # Simple structure analysis
            structure = "NEUTRAL"
            if current_high > prev_high:
                structure = "BULLISH_BOS"
            elif current_low < prev_low:
                structure = "BEARISH_BOS"
                
            return {
                'structure': structure,
                'strength': 0.6,
                'key_levels': {
                    'resistance': float(highs.iloc[-1]),
                    'support': float(lows.iloc[-1])
                }
            }
        except Exception as e:
            logger.error(f"Structure analysis error: {e}")
            return {'structure': 'NEUTRAL', 'strength': 0.0, 'key_levels': {}}
    
    def detect_order_blocks(self, data: pd.DataFrame) -> List[Dict]:
        """Detect institutional order blocks"""
        try:
            order_blocks = []
            
            # Simple order block detection
            for i in range(10, len(data) - 5):
                high = data['high'].iloc[i]
                low = data['low'].iloc[i]
                volume = data.get('tick_volume', pd.Series([1] * len(data))).iloc[i]
                
                # Check for significant price movement
                prev_close = data['close'].iloc[i-1]
                curr_close = data['close'].iloc[i]
                price_change = abs(curr_close - prev_close) / prev_close
                
                if price_change > 0.002 and volume > data.get('tick_volume', pd.Series([1] * len(data))).rolling(20).mean().iloc[i]:
                    order_blocks.append({
                        'type': 'BULLISH' if curr_close > prev_close else 'BEARISH',
                        'high': float(high),
                        'low': float(low),
                        'strength': min(price_change * 100, 1.0),
                        'timestamp': data.index[i] if hasattr(data.index[i], 'strftime') else datetime.now()
                    })
            
            return order_blocks[-5:]  # Return last 5 order blocks
        except Exception as e:
            logger.error(f"Order block detection error: {e}")
            return []

class MT5Broker:
    """MetaTrader 5 Broker Interface"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Login with credentials
            login = self.config.get('login') or int(os.getenv('MT5_LOGIN', 0))
            password = self.config.get('password') or os.getenv('MT5_PASSWORD')
            server = self.config.get('server') or os.getenv('MT5_SERVER')
            
            if login and password and server:
                if not mt5.login(login, password=password, server=server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                    
            self.connected = True
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to MT5 - Balance: ${account_info.balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str = "M5", count: int = 500) -> pd.DataFrame:
        """Get market data from MT5"""
        try:
            # Map timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def place_order(self, symbol: str, action: str, volume: float, price: float, 
                   stop_loss: float = None, take_profit: float = None) -> Dict:
        """Place trading order"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "NEXT_LEVEL_BRAIN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {'success': False, 'error': f'Order failed: {result.retcode}'}
            
            logger.info(f"Order placed: {action} {volume} {symbol} at {price}")
            return {
                'success': True,
                'ticket': result.order,
                'price': result.price,
                'volume': result.volume
            }
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
                
            return [
                {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time)
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

class RiskManager:
    """Risk Management System"""
    
    def __init__(self, config: Dict):
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.daily_pnl = 0.0
        
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss: float, symbol: str) -> float:
        """Calculate optimal position size"""
        try:
            # Risk amount
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Price difference
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                return 0.01  # Minimum position size
            
            # Calculate position size based on asset type
            if 'BTC' in symbol or 'ETH' in symbol:
                # For crypto, use smaller position sizes
                position_size = min(risk_amount / (price_diff * 10), 0.1)
            elif 'XAU' in symbol or 'XAG' in symbol:
                # For metals
                position_size = min(risk_amount / price_diff, 1.0)
            else:
                # For forex
                position_size = min(risk_amount / (price_diff * 100000), 1.0)
            
            # Round to valid lot size
            position_size = round(position_size, 2)
            
            # Ensure minimum and maximum limits
            return max(0.01, min(position_size, 1.0))
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.01
    
    def check_risk_limits(self, account_balance: float, current_drawdown: float) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Check daily loss limit
        if abs(self.daily_pnl) > account_balance * self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
            
        # Check maximum drawdown
        if current_drawdown > self.max_drawdown:
            logger.warning("Maximum drawdown limit reached")
            return False
            
        return True

class LiveTradingSystem:
    """Main Live Trading System"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.broker = MT5Broker(self.config.get('mt5', {}))
        self.ai_brain = TradingBrain()
        self.ict_analyzer = ICTAnalyzer()
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        
        self.running = False
        self.symbols = self.config.get('symbols', ['EURUSDm', 'GBPUSDm', 'XAUUSDm'])
        self.timeframe = self.config.get('timeframe', 'M5')
        
        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.start_balance = 0.0
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default configuration
                return {
                    'mt5': {
                        'login': None,
                        'password': None,
                        'server': None
                    },
                    'symbols': ['EURUSDm', 'GBPUSDm', 'XAUUSDm'],
                    'timeframe': 'M5',
                    'risk': {
                        'max_risk_per_trade': 0.02,
                        'max_daily_loss': 0.05,
                        'max_drawdown': 0.15
                    }
                }
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            logger.info("🧠 Initializing NEXT LEVEL BRAIN Live Trading System...")
            
            # Connect to broker
            if not await self.broker.connect():
                logger.error("Failed to connect to MT5")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                self.start_balance = account_info.balance
                logger.info(f"Account Balance: ${self.start_balance:.2f}")
            
            logger.info("✅ System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    async def analyze_and_trade(self, symbol: str):
        """Analyze market and execute trades for a symbol"""
        try:
            # Get market data (more bars for M5 timeframe)
            data = self.broker.get_market_data(symbol, self.timeframe, 500)
            if data.empty:
                return
            
            # AI Analysis
            ai_analysis = self.ai_brain.analyze_market(symbol, data)
            
            # ICT Analysis
            structure = self.ict_analyzer.analyze_market_structure(data)
            order_blocks = self.ict_analyzer.detect_order_blocks(data)
            
            # Check if we should trade
            if ai_analysis['confidence'] < self.ai_brain.confidence_threshold:
                return
            
            if ai_analysis['action'] in ['BUY', 'SELL']:
                await self._execute_trade(symbol, ai_analysis, structure)
                
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
    
    async def _execute_trade(self, symbol: str, ai_analysis: Dict, structure: Dict):
        """Execute a trade based on analysis"""
        try:
            # Get account balance
            account_info = mt5.account_info()
            if not account_info:
                return
            
            balance = account_info.balance
            
            # Check risk limits
            current_drawdown = max(0, (self.start_balance - balance) / self.start_balance) if self.start_balance else 0
            if not self.risk_manager.check_risk_limits(balance, current_drawdown):
                logger.warning(f"Risk limits prevent trading {symbol}")
                return
            
            # Calculate position size
            entry_price = ai_analysis['entry_price']
            stop_loss = ai_analysis['stop_loss']
            take_profit = ai_analysis['take_profit']
            
            position_size = self.risk_manager.calculate_position_size(
                balance, entry_price, stop_loss, symbol
            )
            
            # Place order
            result = self.broker.place_order(
                symbol=symbol,
                action=ai_analysis['action'],
                volume=position_size,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result['success']:
                self.trades_today += 1
                logger.info(f"✅ Trade executed: {ai_analysis['action']} {symbol}")
                logger.info(f"📊 Reasoning: {ai_analysis['reasoning']}")
                
                # Remember trade for AI learning
                self.ai_brain.remember_trade({
                    'symbol': symbol,
                    'action': ai_analysis['action'],
                    'entry_price': entry_price,
                    'confidence': ai_analysis['confidence']
                })
            else:
                logger.error(f"❌ Trade failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            positions = self.broker.get_positions()
            
            for pos in positions:
                # Simple trailing stop logic
                if pos['profit'] > 50:  # If profit > $50
                    logger.info(f"💰 Position {pos['symbol']} in profit: ${pos['profit']:.2f}")
                elif pos['profit'] < -100:  # If loss > $100
                    logger.warning(f"⚠️ Position {pos['symbol']} in loss: ${pos['profit']:.2f}")
                    
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    def display_status(self):
        """Display current system status"""
        try:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                daily_change = current_balance - self.start_balance
                
                print(f"\n{'='*50}")
                print(f"🧠 NEXT LEVEL BRAIN - LIVE TRADING STATUS")
                print(f"{'='*50}")
                print(f"💰 Balance: ${current_balance:.2f}")
                print(f"📈 Daily P&L: ${daily_change:.2f}")
                print(f"📊 Trades Today: {self.trades_today}")
                print(f"🎯 AI Memories: {len(self.ai_brain.memories)}")
                print(f"⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}")
                
                positions = self.broker.get_positions()
                if positions:
                    print(f"📋 Open Positions: {len(positions)}")
                    for pos in positions:
                        print(f"  {pos['symbol']}: {pos['type']} ${pos['profit']:.2f}")
                else:
                    print(f"📋 Open Positions: 0")
                print(f"{'='*50}")
                
        except Exception as e:
            logger.error(f"Status display error: {e}")
    
    async def run(self):
        """Main trading loop"""
        try:
            if not await self.initialize():
                return
            
            self.running = True
            logger.info("🚀 Starting live trading...")
            
            cycle_count = 0
            
            while self.running:
                try:
                    cycle_count += 1
                    
                    # Display status every 10 cycles
                    if cycle_count % 10 == 0:
                        self.display_status()
                    
                    # Analyze each symbol
                    for symbol in self.symbols:
                        await self.analyze_and_trade(symbol)
                        await asyncio.sleep(1)  # Small delay between symbols
                    
                    # Monitor positions
                    await self.monitor_positions()
                    
                    # Wait before next cycle
                    await asyncio.sleep(30)  # 30 second cycle
                    
                except KeyboardInterrupt:
                    logger.info("Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Fatal trading error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the trading system"""
        try:
            self.running = False
            logger.info("🛑 Shutting down trading system...")
            
            # Close MT5 connection
            mt5.shutdown()
            
            logger.info("✅ Shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

def select_trading_mode():
    """Select trading mode (CLI)"""
    print("\n" + "="*60)
    print("🧠 NEXT LEVEL BRAIN - LIVE TRADING SYSTEM (CLI)")
    print("Created by: Aleem Shahzad | AI Partner: Claude (Anthropic)")
    print("="*60)
    print("Select trading mode:")
    print("1. 🎯 Single Symbol Trading")
    print("2. 🌍 Multi-Symbol Trading")
    print("3. ❌ Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                symbols = {
                    1: "EURUSDm", 2: "GBPUSDm", 3: "USDJPYm",
                    4: "XAUUSDm", 5: "XAGUSDm", 6: "BTCUSDm", 7: "ETHUSDm"
                }
                
                print("\nSelect symbol:")
                for num, symbol in symbols.items():
                    print(f"  {num}. {symbol}")
                
                symbol_choice = int(input("Enter symbol choice: "))
                if symbol_choice in symbols:
                    return [symbols[symbol_choice]]
                    
            elif choice == "2":
                return ['EURUSDm', 'GBPUSDm', 'XAUUSDm', 'BTCUSDm']
                
            elif choice == "3":
                return None
                
            else:
                print("Invalid choice. Please try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            return None

def main():
    """Main function - CLI mode"""
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("charts").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        print("🚀 Launching NEXT LEVEL BRAIN Live Trading (CLI mode)...")
        symbols = select_trading_mode()
        if symbols is None:
            print("Exiting.")
            return
        
        trading_system = LiveTradingSystem()
        trading_system.symbols = symbols
        
        # Run trading loop
        asyncio.run(trading_system.run())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
