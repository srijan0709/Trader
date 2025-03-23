import requests
import pandas as pd
import ta
from tqdm import tqdm as tq  
class StockDataFetcher:
    def __init__(self):
        self.api_key = "PKW0YNTXIQBYEO0GHQS4"
        self.api_secret = "r0obc1M6nmK1k9aTF7OixLHtGB0PCLBR9RFAJ2Rm"
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
        temp["next_state_index"] = next_step +i
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
    data['MA50'] = ta.trend.sma_indicator(data['close'], window=30)
    data['RSI'] = ta.momentum.rsi(data['close'], window=30)
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