import requests
import pandas as pd
import ta
from tqdm import tqdm as tq  
import os
from config import STOCKS
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import threading

class StockDataFetcher:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = "https://data.alpaca.markets/v2/stocks/bars"
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str,timeframe: str = "1Min", limit: int = 5000):
        url = f"{self.base_url}?symbols={symbol}&timeframe={timeframe}&start={start_date}&end={end_date}&limit={limit}&adjustment=raw&feed=sip&sort=asc"
        
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return self.create_dataframe(data, symbol)
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    
    def create_dataframe(self, data, symbol):
        if "bars" not in data or symbol not in data["bars"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(data["bars"][symbol])
        df["t"] = pd.to_datetime(df["t"])
        df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trades"}, inplace=True)
    
        df['time'] = pd.to_datetime(df['time'])

        df['time_et'] = df['time'].dt.tz_convert('US/Eastern')

        df_market_hours = df[(df['time_et'].dt.time >= pd.to_datetime("09:30:00").time()) & 
                            (df['time_et'].dt.time <= pd.to_datetime("16:00:00").time())]
        df_market_hours.drop(columns=['time'], inplace=True)
        df_market_hours.rename(columns={'time_et': 'time'}, inplace=True)
        df_market_hours.reset_index(drop=True, inplace=True)

        return df_market_hours

# fetcher = StockDataFetcher()

# stock_symbol = "TSLA"
# start_date = "2025-02-25T00:00:00Z"
# end_date = "2025-03-02T09:30:00-04:00"

# df = fetcher.get_stock_data(stock_symbol, start_date, end_date)

def getNextStateBuySide(current_state, target, stop_loss, df):
    '''
    This will return the next state for the buy side
    '''
    current_price = current_state['close']
    current_date = current_state['time'].date()
    target_price = current_price * target
    stop_loss_price = current_price * stop_loss

    for i in range(len(df)):
        if df.iloc[i]['time'].date() == current_date:  # Fixed missing iloc
            if df.iloc[i]['high'] >= target_price:  # Fixed missing iloc
                result = df.iloc[i].copy()
                result['action'] = 'Target Hit'
                result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
                return result,i
            if df.iloc[i]['low'] <= stop_loss_price:  # Fixed missing iloc
                result = df.iloc[i].copy()
                result['action'] = 'Stop Loss Hit'
                result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
                return result,i
        else:
            
            result = df.iloc[i - 1].copy()
            
            result['action'] = 'End of Day'
            result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
            return result,i-1

    result = df.iloc[-1].copy()
    result['action'] = 'End of Day'
    result['delay'] = (df.iloc[-1]['time'] - current_state['time']).total_seconds() // 60
    return result,len(df)-1


def getNextStateSellSide(current_state, target, stop_loss, df):
    '''
    This will return the next state for the sell side
    '''
    current_price = current_state['close']
    current_date = current_state['time'].date()
    target_price = current_price * target
    stop_loss_price = current_price * stop_loss

    for i in range(len(df)):
        if df.iloc[i]['time'].date() == current_date:  # Fixed missing iloc
            if df.iloc[i]['low'] <= target_price:  # Fixed missing iloc
                result = df.iloc[i].copy()
                result['action'] = 'Target Hit'
                result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
                return result,i
            if df.iloc[i]['high'] >= stop_loss_price:  # Fixed missing iloc
                result = df.iloc[i].copy()
                result['action'] = 'Stop Loss Hit'
                result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
                return result,i
        else:
            
            result = df.iloc[i - 1].copy()
            
            result['action'] = 'End of Day'
            result['delay'] = (df.iloc[i]['time'] - current_state['time']).total_seconds() // 60
            return result,i-1

    result = df.iloc[-1].copy()
    result['action'] = 'End of Day'
    result['delay'] = (df.iloc[-1]['time'] - current_state['time']).total_seconds() // 60
    return result,len(df)-1

def generateTargetDataBuySide(df, target, stop_loss):
    '''
    This will generate the target data for the buy side
    '''
    target_data = []
    for i in tq(range(len(df)), desc="Processing Buy Side Data"):
        temp,next_step = getNextStateBuySide(df.iloc[i], target, stop_loss, df.iloc[i:].reset_index(drop=True))
        temp["next_state_index"] = next_step + i
        target_data.append(temp)
    df = pd.DataFrame(target_data)
    df.reset_index(drop=True, inplace=True)
    return df
   


def generateTargetDataSellSide(df,target,stop_loss):
    '''
    This will generate the target data for the sell side
    '''
    target_data = []
    for i in tq(range(len(df)), desc="Processing Sell Side Data"):
        temp,next_step = getNextStateSellSide(df.iloc[i], target, stop_loss, df.iloc[i:].reset_index(drop=True))
        temp["next_state_index"] = next_step +i
        target_data.append(temp)
    
    df = pd.DataFrame(target_data)
    df.reset_index(drop=True, inplace=True)
    return df

def getTechnicalIndicators(data):
    '''
    This will return the technical indicators for the stock data'
    '''
    # Calculate technical indicators
    data['MA50'] = ta.trend.sma_indicator(data['close'], window=28)
    data['RSI'] = ta.momentum.rsi(data['close'], window=28)
    data['MACD'] = ta.trend.macd(data['close'])
    data['BB_upper'] = ta.volatility.bollinger_hband(data['close'])
    data['BB_lower'] = ta.volatility.bollinger_lband(data['close'])
    data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'])
    data['CCI'] = ta.trend.cci(data['high'], data['low'], data['close'])
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    data['ROC'] = ta.momentum.roc(data['close'])
    data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])

    # Drop rows with NaN values (resulting from the indicator calculation)
    data = data.dropna()
    data.reset_index(drop=True, inplace=True)
    return data

def normalize_dataframe_with_mean_std(df):
    """
    Normalize all columns in the dataframe except the 'time' column using Z-score normalization.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Normalized dataframe.
        dict: Mean and standard deviation values for each column (used for normalizing new rows).
    """
    # Create a copy of the dataframe to avoid modifying the original
    normalized_df = df.copy()
    
    # Dictionary to store mean and standard deviation values for each column
    normalization_params = {}
    
    # Normalize all columns except 'time'
    for col in normalized_df.columns:
        if col != 'time' and col != 'action' and col != 'delay' and col !='close' and col != 'high' and col != 'low' and col != 'open':
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            normalization_params[col] = {'mean': col_mean, 'std': col_std}
            
            # Apply Z-score normalization
            normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        elif col == 'time':
            for i in range(len(normalized_df[col])) :
                normalized_df[col].iloc[i] = normalized_df[col].iloc[i].hour*60 + normalized_df[col].iloc[i].minute-540
            
            
    
    return normalized_df, normalization_params


def normalize_new_row_with_mean_std(row, normalization_params):
    """
    Normalize a new row using previously computed mean and standard deviation values.
    
    Args:
        row (pd.Series): New row of data.
        normalization_params (dict): Dictionary containing mean and standard deviation values for each column.
        
    Returns:
        pd.Series: Normalized row.
    """
    normalized_row = row.copy()
    
    for col in row.index:
        if col in normalization_params:
            col_mean = normalization_params[col]['mean']
            col_std = normalization_params[col]['std']
            
            # Apply Z-score normalization
            normalized_row[col] = (row[col] - col_mean) / col_std
        elif col == 'time':
            normalized_row[col] = row[col].hour*60 + row[col].minute-540
    
    return normalized_row

class UpstoxStockDataFetcher:
    def __init__(self):
        self.base_url = "https://api.upstox.com/v2/historical-candle/NSE_EQ%7C"
        
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str,timeframe: str = "1minute"):
        isin_number = STOCKS[symbol]

        url = f"{self.base_url}{isin_number}/{timeframe}/{end_date}/{start_date}"
        
        headers = {
                 'Accept': 'application/json'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()['data']['candles']
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'Open_Interest'])
            df.drop(columns=['Open_Interest'], inplace=True)
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse the order of rows
            df['time'] = pd.to_datetime(df['time'])
            return df
        else:
            print(url)
            return {"error": f"Request failed with status code {response.status_code} "}
        
# https://upstox.com/developer/api-documentation/open-api

class UpstoxTrader():
    def __init__(self):
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    
    def get_margin(self):
        url = 'https://api.upstox.com/v2/user/get-funds-and-margin'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data['data']['equity']['available_margin']
        else:
            print(f"Error fetching margin: {response.status_code}")
            return None
    
    def MarketStatus(self):
        url = 'https://api.upstox.com/v2/market/status/NSE'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()['data']
            status = data['status']
            
            if status == 'NORMAL_OPEN':
                return True
            else:
                return False 
        else:
            print(f"Error fetching market status: {response.status_code}")
            return None
        
    def getPositions(self):
        url = 'https://api.upstox.com/v2/portfolio/short-term-positions'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error fetching position: {response.status_code}")
            return None
    
    def IntraMarketOrder(self, symbol, quantity, action):
        symbol = STOCKS[symbol]  # Get the ISIN number from the STOCKS dictionary
        symbol= "NSE_EQ|" + symbol
        url = 'https://api-hft.upstox.com/v2/order/place'
        headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.access_token}',
                     }
        data = {
            'quantity': quantity,
            'product': 'I',
            'validity': 'DAY',
            'price': 0,
            'tag': 'string',
            'instrument_token': symbol,
            'order_type': 'MARKET',
            'transaction_type': action,  # 'BUY' or 'SELL'
            'disclosed_quantity': 0,
            'trigger_price': 0,
            'is_amo': False,
            }

        try:
            # Send the POST request
            response = requests.post(url, json=data, headers=headers)
            # Print the response status code and body
            print('Response Code:', response.status_code)
            print('Response Body:', response.json())
            return response.json()

        except Exception as e:
            # Handle exceptions
            print('Error:', str(e))
            return None
        
    def IntraLimitOrder(self, symbol, quantity, action, price):
        symbol = STOCKS[symbol]
        symbol= "NSE_EQ|" + symbol
        url = 'https://api-hft.upstox.com/v2/order/place'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }

        data = {
            'quantity': quantity,
            'product': 'I',
            'validity': 'DAY',
            'price': price,
            'tag': 'string',
            'instrument_token': symbol,
            'order_type': 'LIMIT',
            'transaction_type': action,
            'disclosed_quantity': 0,
            'trigger_price': 20.1,
            'is_amo': False,
        }

        try:
            # Send the POST request
            response = requests.post(url, json=data, headers=headers)

            # Print the response status code and body
            print('Response Code:', response.status_code)
            print('Response Body:', response.json())

            return response.json()

        except Exception as e:
            # Handle exceptions
            print('Error:', str(e))
            return None

    def IntraDayStopLossOrder(self, symbol, quantity, action, price):
        symbol = STOCKS[symbol]
        symbol= "NSE_EQ|" + symbol
        url = 'https://api-hft.upstox.com/v2/order/place'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }

        data = {
            'quantity': quantity,
            'product': 'I',
            'validity': 'DAY',
            'price': 0.0,
            'tag': 'string',
            'instrument_token': symbol,
            'order_type': 'SL-M',
            'transaction_type': action,  
            'disclosed_quantity': 0,
            'trigger_price': price,
            'is_amo': False,
        }

        try:
            # Send the POST request
            response = requests.post(url, json=data, headers=headers)

            # Print the response status code and body
            print('Response Code:', response.status_code)
            print('Response Body:', response.json())

            return response.json()

        except Exception as e:
            # Handle exceptions
            print('Error:', str(e))
            return None

    def CancelOrder(self, order_id):
        """
        Cancel an order by its order ID.

        Args:
            order_id (str): The order ID to cancel.

        Returns:
            requests.Response: The response from the API.
        """
        
        url = f'https://api-hft.upstox.com/v2/order/cancel?order_id={order_id}'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        try:
            response = requests.delete(url, headers=headers)

            print(response.text)

            return response.json()

        except Exception as e:
            print('Error:', str(e))
            return None
        
    def getCharges(self,start_date, end_date):
        """
        Get the charges for a given date range.

        Args:
            start_date (str): The start date in 'dd-mm-yyyy' format.
            end_date (str): The end date in 'dd-mm-yyyy' format.

        Returns:
            dict: A dictionary containing the charges.
        """
        url = 'https://api.upstox.com/v2/trade/profit-loss/charges'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}' 
        }

        params = {
            'from_date': start_date,
            'to_date': end_date,
            'segment': 'EQ',
            'financial_year': '2526'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)

            return response.json() 
        except Exception as e:
            print('Error:', str(e))
            return None
        
    def getProfitLoss(self,start_date, end_date):
        """
        Get the profit and loss for a given date range.

        Args:
            start_date (str): The start date in 'DD-MM-YYYY' format.
            end_date (str): The end date in 'DD-MM-YYYY' format.

        Returns:
            dict: A dictionary containing the profit and loss data.
        """
        url = 'https://api.upstox.com/v2/trade/profit-loss/data'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}' 
        }

        params = {
            'from_date': start_date,
            'to_date': end_date,
            'segment': 'EQ',
            'financial_year': '2526',   ## TO do make this a func of startdate and end date
            'page_number': '1',
            'page_size': '4'
            }
        
        try:
            response = requests.get(url, headers=headers, params=params)

            return response.json() 
        except Exception as e:
            print('Error:', str(e))
            return None
    
    def getOrderDetails(self,order_id):
        url = 'https://api.upstox.com/v2/order/details'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

        params = {'order_id': order_id}
        response = requests.get(url, headers=headers, params=params)

        return response.json()
    
    def exitAllPositions(self):
        """
        Exit all positions by selling them.
        """
        url = 'https://api.upstox.com/v2/order/positions/exit'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }

        data = {}

        try:
            # Send the POST request
            response = requests.post(url, json=data, headers=headers)
            # Print the response status code and body
            print('Response Code:', response.status_code)
            print('Response Body:', response.json())

        except Exception as e:
            # Handle exceptions
            print('Error:', str(e))
            return None

        
class TechnicalIndicatorWindow:
    def __init__(self, window_size=28):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)

    def update(self, ohlcv_bar):
        """
        Add a new OHLCV bar to the window.
        ohlcv_bar = {
            'timestamp': ..., 
            'open': ..., 
            'high': ..., 
            'low': ..., 
            'close': ..., 
            'volume': ...
        }
        """
        self.data.append(ohlcv_bar)

    def _get_dataframe(self):
        df = pd.DataFrame(list(self.data))
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')  # convert to float/int
        return df

    def get_feature_row(self):
        """
        Returns the latest feature vector (technical indicators) for the model.
        Returns None if not enough data for indicators.
        """
        df = self._get_dataframe()
        if len(df) < 28:  # Minimum required for indicators like ADX, RSI, etc.
            return None
        
        df['MA50'] = ta.trend.sma_indicator(df['close'], window=self.window_size)
        df['RSI'] = ta.momentum.rsi(df['close'], window=self.window_size)
        df['MACD'] = ta.trend.macd(df['close'])
        df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'])
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['ROC'] = ta.momentum.roc(df['close'])
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

        df = df.dropna()
        if df.empty:
            print("Not enough data to compute indicators.")
            return None

        latest_features = df.iloc[-1]
        return latest_features

class TradeMonitoringBot:
    '''Class to monitor active orders and manage their execution status in a separate thread'''
    def __init__(self,trader):
        self.order_monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.active_orders = {}
        self.trader = trader
        self.target_hits = 0
        self.stop_loss_hits = 0
        
    def monitor_orders(self, order_id):
        """Run in separate thread to monitor order execution"""
        while not self.stop_monitoring.is_set():
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_stat1 = executor.submit(self.trader.getOrderDetails, order_id['target_order'])
                    future_stat2 = executor.submit(self.trader.getOrderDetails, order_id['stop_loss_order'])
                    stat1 = future_stat1.result()
                    stat2 = future_stat2.result()
                
                if stat1['data']['status'] == 'complete':
                    self.trader.CancelOrder(order_id['stop_loss_order'])
                    print("Target order executed, stop loss order cancelled.")
                    self.target_hits += 1
                    self.active_orders.clear()
                    break
                elif stat2['data']['status'] == 'complete':
                    self.trader.CancelOrder(order_id['target_order'])
                    print("Stop loss order executed, target order cancelled.")
                    self.stop_loss_hits += 1
                    self.active_orders.clear()
                    break
                    
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.active_orders.clear()
                print(f"Error monitoring orders: {e}")
                time.sleep(5)
    
    def start_order_monitoring(self, order_id):
        """Start monitoring orders in background"""
        if self.order_monitoring_thread and self.order_monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.order_monitoring_thread.join()
        
        self.stop_monitoring.clear()
        self.active_orders = order_id.copy()
        self.order_monitoring_thread = threading.Thread(target=self.monitor_orders, args=(order_id,))
        self.order_monitoring_thread.daemon = True
        self.order_monitoring_thread.start()
    
    def has_active_orders(self):
        """Check if there are active orders being monitored"""
        return bool(self.active_orders)
    
    def stop_order_monitoring(self):
        """Stop monitoring thread gracefully"""
        if self.order_monitoring_thread and self.order_monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.order_monitoring_thread.join()
            self.active_orders.clear()
            print("Stopped order monitoring.")

    def get_trade_stats(self):
        """Return current count of trades that hit target or stop loss"""
        return {
            'target_hits': self.target_hits,
            'stop_loss_hits': self.stop_loss_hits
        }