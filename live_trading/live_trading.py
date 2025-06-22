#!/usr/bin/env python3
"""
Suzlon Trading Bot - Automated Trading System
Converted from Jupyter notebook for production use in VM/tmux
"""

import sys
import os
import json
import numpy as np
import torch
import logging
import signal
import threading
import time
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytz
import argparse

# Setup logging
def setup_logging():
    """Setup comprehensive logging for the trading bot"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    loggers = {
        'trading': logging.getLogger('trading'),
        'websocket': logging.getLogger('websocket'),
        'orders': logging.getLogger('orders'),
        'model': logging.getLogger('model'),
        'system': logging.getLogger('system')
    }
    
    return loggers

# Initialize logging
loggers = setup_logging()
trading_logger = loggers['trading']
ws_logger = loggers['websocket']
orders_logger = loggers['orders']
model_logger = loggers['model']
system_logger = loggers['system']

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

# Import custom modules
try:
    from utils_data import normalize_new_row_with_mean_std, TechnicalIndicatorWindow, UpstoxTrader, TradeMonitoringBot
    from websocket_client import fetch_market_data, get_latest_ohlc_volume, market_data_store, get_market_summary
    from Models.DQN import DQN, DQNAgent
    from trading_environment import StockTradingEnv
    system_logger.info("Successfully imported all custom modules")
except ImportError as e:
    system_logger.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, symbol="SUZLON", model_path=None, norm_path=None):
        self.symbol = symbol
        self.model_path = model_path
        self.norm_path = norm_path
        self.running = True
        self.ws_thread = None
        self.india_tz = pytz.timezone('Asia/Kolkata')
        self.csv_path = None
        
        # Initialize components
        self._init_model()
        self._init_trading_components()
        self._load_normalization_params()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        system_logger.info(f"Trading bot initialized for symbol: {symbol}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        system_logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _init_model(self):
        """Initialize the DQN model"""
        try:
            self.policy_net = DQN(16, 3)
            self.target_net = DQN(16, 3)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.device = torch.device('cpu')
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            
            if self.model_path is None:
                raise ValueError("Model path not provided.")
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.policy_net.eval()
            
            model_logger.info("Model loaded successfully")
        except Exception as e:
            model_logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _init_trading_components(self):
        """Initialize trading components"""
        try:
            self.trader = UpstoxTrader()
            self.ti_window = TechnicalIndicatorWindow(window_size=28)
            self.bot = TradeMonitoringBot(self.trader)
            trading_logger.info("Trading components initialized successfully")
        except Exception as e:
            trading_logger.error(f"Failed to initialize trading components: {e}")
            raise
    
    def _load_normalization_params(self):
        """Load normalization parameters"""
        try:
            if self.norm_path is None:
                raise ValueError("Normalization params path not provided.")
            with open(self.norm_path, "r") as f:
                self.norm_params = json.load(f)
            system_logger.info("Normalization parameters loaded successfully")
        except Exception as e:
            system_logger.error(f"Failed to load normalization parameters: {e}")
            raise
    
    def get_state(self, row):
        """Convert row data to state array"""
        state = np.array([
            row['time'],
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume'],
            row['MA50'],
            row['RSI'],
            row['MACD'],
            row['BB_upper'],
            row['BB_lower'],
            row['ADX'],
            row['CCI'],
            row['ATR'],
            row['ROC'],
            row['OBV']
        ], dtype=np.float32)
        return state
    
    async def start_websocket_connection(self):
        """Start WebSocket connection with reconnection logic"""
        while self.running:
            try:
                ws_logger.info("Starting WebSocket connection...")
                await fetch_market_data(symbol=self.symbol)
            except Exception as e:
                ws_logger.error(f"WebSocket error: {e}, reconnecting in 5 seconds...")
                await asyncio.sleep(5)
    
    def start_ws_loop(self):
        """Start WebSocket event loop in separate thread"""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.start_websocket_connection())
            except Exception as e:
                ws_logger.error(f"WebSocket loop error: {e}")
            finally:
                loop.close()
        
        self.ws_thread = threading.Thread(target=run_loop, daemon=True)
        self.ws_thread.start()
        ws_logger.info("WebSocket thread started")
    
    def setup_csv_logging(self):
        """Setup CSV file for market data logging"""
        today = datetime.now(self.india_tz).date()
        norm_path = Path(self.norm_path) 
        folder = norm_path.parent 
        self.csv_path = folder / f"{self.symbol}_{today}.csv"
        trading_logger.info(f"CSV logging setup: {self.csv_path}")
    
    def log_market_data(self, new_bar):
        """Log market data to CSV file"""
        try:
            file_exists = self.csv_path.exists()
            with open(self.csv_path, 'a') as f:
                if not file_exists:
                    f.write("time,open,high,low,close,volume\n")
                f.write(f"{new_bar['timestamp']},{new_bar['open']},{new_bar['high']},{new_bar['low']},{new_bar['close']},{new_bar['volume']}\n")
        except Exception as e:
            trading_logger.error(f"Failed to log market data: {e}")
    
    def get_model_prediction(self, state):
        """Get model prediction for given state"""
        try:
            t_start = time.time()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            inference_time = time.time() - t_start
            model_logger.info(f"Model inference completed in {inference_time:.3f}s, action: {action}")
            return action
        except Exception as e:
            model_logger.error(f"Model prediction failed: {e}")
            return 0  # Default to HOLD
    
    def execute_trade(self, action, latest_price, available_cash):
        """Execute trade based on model prediction"""
        try:
            quantity = int(available_cash * 0.9 * 2.5 / latest_price)
            if quantity <= 10:
                trading_logger.warning("Insufficient funds to place an order")
                return None
            
            order_id = {}
            t_start = time.time()
            
            if action == 1:  # BUY
                orders_logger.info(f"Placing BUY order - Quantity: {quantity}, Price: {latest_price}")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_market = executor.submit(self.trader.IntraMarketOrder, self.symbol, quantity, "BUY")
                    future_target = executor.submit(self.trader.IntraLimitOrder, self.symbol, quantity, "SELL", latest_price * 1.004)
                    future_stop_loss = executor.submit(self.trader.IntraDayStopLossOrder, self.symbol, quantity, "SELL", latest_price * 0.99)
                    
                    market_order = future_market.result()
                    target_order = future_target.result()
                    stop_loss_order = future_stop_loss.result()
            
            elif action == 2:  # SELL
                orders_logger.info(f"Placing SELL order - Quantity: {quantity}, Price: {latest_price}")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_market = executor.submit(self.trader.IntraMarketOrder, self.symbol, quantity, "SELL")
                    future_target = executor.submit(self.trader.IntraLimitOrder, self.symbol, quantity, "BUY", latest_price * 0.996)
                    future_stop_loss = executor.submit(self.trader.IntraDayStopLossOrder, self.symbol, quantity, "BUY", latest_price * 1.01)
                    
                    market_order = future_market.result()
                    target_order = future_target.result()
                    stop_loss_order = future_stop_loss.result()
            
            order_id['market_order'] = market_order['data']['order_id']
            order_id['target_order'] = target_order['data']['order_id']
            order_id['stop_loss_order'] = stop_loss_order['data']['order_id']
            
            execution_time = time.time() - t_start
            orders_logger.info(f"Orders placed successfully in {execution_time:.2f}s")
            orders_logger.info(f"Order IDs - Market: {order_id['market_order']}, Target: {order_id['target_order']}, Stop Loss: {order_id['stop_loss_order']}")
            
            # Start monitoring orders
            self.bot.start_order_monitoring(order_id)
            
            return order_id
            
        except Exception as e:
            orders_logger.error(f"Failed to execute trade: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources and exit positions"""
        try:
            system_logger.info("Starting cleanup process...")
            
            # Stop order monitoring
            self.bot.stop_order_monitoring()
            
            # Exit all positions
            self.trader.exitAllPositions()
            
            # Cancel any pending orders (if any)
            try:
                if hasattr(self, 'current_orders') and self.current_orders:
                    self.trader.CancelOrder(order_id=self.current_orders.get('stop_loss_order'))
                    self.trader.CancelOrder(order_id=self.current_orders.get('target_order'))
            except:
                pass
            
            system_logger.info("Cleanup completed successfully")
            
        except Exception as e:
            system_logger.error(f"Error during cleanup: {e}")
    
    def run(self):
        """Main trading loop"""
        try:
            system_logger.info("Starting Suzlon Trading Bot...")
            
            # Setup components
            self.setup_csv_logging()
            self.start_ws_loop()
            
            # Wait for WebSocket connection
            time.sleep(10)
            ws_logger.info("WebSocket connection established")
            
            # Get market close time
            now_ist = datetime.now(self.india_tz)
            market_close_time = now_ist.replace(hour=15, minute=9, second=0, microsecond=0)
            
            iteration_count = 0
            
            while self.running:
                iteration_count += 1
                iteration_start = time.time()
                
                now_ist = datetime.now(self.india_tz)
                
                # Check market status
                try:
                    market_status = self.trader.MarketStatus()
                except Exception as e:
                    trading_logger.error(f"Error fetching market status: {e}")
                    market_status = False
                
                if not market_status or now_ist >= market_close_time:
                    trading_logger.info("Market is closed, shutting down...")
                    break
                
                # Get latest market data
                try:
                    new_bar = get_latest_ohlc_volume(symbol=self.symbol)
                    self.log_market_data(new_bar)
                    
                    # Update technical indicators
                    self.ti_window.update(new_bar)
                    features = self.ti_window.get_feature_row()
                    
                    if features is None:
                        trading_logger.info(f"Iteration {iteration_count}: No features available yet, waiting...")
                        time.sleep(60)
                        continue
                    
                except Exception as e:
                    trading_logger.error(f"Error getting market data: {e}")
                    time.sleep(60)
                    continue
                
                # Check if we have active orders
                if not self.bot.has_active_orders():
                    try:
                        # Prepare state for model
                        features = features.rename({'timestamp': 'time'})
                        normalized_row = normalize_new_row_with_mean_std(features, self.norm_params)
                        state = self.get_state(normalized_row)
                        
                        # Get model prediction
                        action = self.get_model_prediction(state)
                        
                        if action != 0:  # If action is BUY or SELL
                            # Get required data for trade execution
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                future_cash = executor.submit(self.trader.get_margin)
                                future_price = executor.submit(get_market_summary, self.symbol)
                                
                                available_cash = future_cash.result()
                                latest_price = future_price.result()['latest_price']
                            
                            # Execute trade
                            self.current_orders = self.execute_trade(action, latest_price, available_cash)
                        else:
                            trading_logger.info(f"Iteration {iteration_count}: Model prediction is HOLD")
                    
                    except Exception as e:
                        trading_logger.error(f"Error in trading logic: {e}")
                
                else:
                    trading_logger.info(f"Iteration {iteration_count}: Orders being monitored in background")
                
                # Performance logging
                iteration_time = time.time() - iteration_start
                trading_logger.info(f"Iteration {iteration_count} completed in {iteration_time:.2f}s")
                
                trade_stats = self.bot.get_trade_stats()
                if trade_stats['target_hits'] > 1 or trade_stats['stop_loss_hits'] > 0: # Exit if significant trades occurred
                    orders_logger.info(f"Trade Stats - Target Hits: {trade_stats['target_hits']}, Stop Loss Hits: {trade_stats['stop_loss_hits']}")
                    break

                # Wait for next minute
                next_minute = (datetime.now(self.india_tz) + timedelta(minutes=1)).replace(second=0, microsecond=0)
                while datetime.now(self.india_tz) < next_minute and self.running:
                    time.sleep(0.5)
                time.sleep(4)  # Additional buffer
        
        except KeyboardInterrupt:
            system_logger.info("Received keyboard interrupt")
        except Exception as e:
            system_logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()
            system_logger.info("Trading bot shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run trading bot for a specific symbol.")
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., SUZLON)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--norm_path', type=str, required=True, help='Path to normalization params (.json)')
    
    args = parser.parse_args()

    try:
        bot = TradingBot(symbol=args.symbol, model_path=args.model_path, norm_path=args.norm_path)
        bot.run()
    except Exception as e:
        system_logger.error(f"Failed to start trading bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

## Example usage:
# python trading_bot.py --symbol SUZLON \
#     --model_path "C:/Users/srija/Assignment/Trading/Models/trained_models/suzlon_14_june/suzlon_14_june_2866.pth" \
#     --norm_path "C:/Users/srija/Assignment/Trading/json_files/suzlon_14_june_norm_params.json"
