# Import necessary modules
import asyncio
import json
import ssl
import websockets
import requests
from google.protobuf.json_format import MessageToDict
import os
import MarketDataFeedV3_pb2 as pb
market_data_store = {}
import threading
from datetime import datetime, timedelta
market_data_store = {}
data_lock = threading.Lock()

def get_market_data_feed_authorize_v3():
    """Get authorization for market data feed."""
    access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    api_response = requests.get(url=url, headers=headers)
    return api_response.json()


def decode_protobuf(buffer):
    """Decode protobuf message."""
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response



def update_market_data(data_dict):
    """Update global market data store with latest data."""
    with data_lock:
        if 'feeds' in data_dict:
            for instrument_key, feed_data in data_dict['feeds'].items():
                if instrument_key not in market_data_store:
                    market_data_store[instrument_key] = {
                        'trades': [],
                        'latest_price': None,
                        'latest_timestamp': None,
                        'ohlc_1min': None
                    }
                
                # Extract trade data
                if 'fullFeed' in feed_data and 'marketFF' in feed_data['fullFeed']:
                    market_ff = feed_data['fullFeed']['marketFF']
                    
                    # Update latest price and timestamp
                    if 'ltpc' in market_ff:
                        ltpc = market_ff['ltpc']
                        current_time = datetime.now()
                        
                        market_data_store[instrument_key]['latest_price'] = ltpc.get('ltp')
                        market_data_store[instrument_key]['latest_timestamp'] = current_time
                        
                        # Store trade data for OHLC calculation
                        trade_data = {
                            'price': ltpc.get('ltp'),
                            'quantity': ltpc.get('ltq', 0),
                            'timestamp': current_time
                        }
                        
                        # Add trade to list
                        market_data_store[instrument_key]['trades'].append(trade_data)
                        
                        # Keep only last minute data
                        one_minute_ago = current_time - timedelta(minutes=1)
                        market_data_store[instrument_key]['trades'] = [
                            trade for trade in market_data_store[instrument_key]['trades']
                            if trade['timestamp'] > one_minute_ago
                        ]
                    
                    # Extract existing OHLC if available
                    if 'marketOHLC' in market_ff and 'ohlc' in market_ff['marketOHLC']:
                        ohlc_data = market_ff['marketOHLC']['ohlc']
                        # Look for 1-minute interval data
                        for ohlc in ohlc_data:
                            if ohlc.get('interval') == 'I1':  # I1 typically represents 1-minute
                                market_data_store[instrument_key]['ohlc_1min'] = {
                                    'open': ohlc.get('open'),
                                    'high': ohlc.get('high'),
                                    'low': ohlc.get('low'),
                                    'close': ohlc.get('close'),
                                    'volume': ohlc.get('vol'),
                                    'timestamp': datetime.fromtimestamp(int(ohlc.get('ts', 0)) / 1000)
                                }
                                break

def get_latest_ohlc_volume(instrument_key="NSE_EQ|INE155A01022"):
    """
    Get the latest OHLC volume data for the last minute.
    
    Args:
        instrument_key (str): The instrument key to get data for
        
    Returns:
        dict: Dictionary containing OHLC volume data or None if no data available
    """
    with data_lock:
        if instrument_key not in market_data_store:
            return None
        
        data = market_data_store[instrument_key]
        
        # If we have pre-calculated OHLC from feed, return that
        if data['ohlc_1min']:
            return {
                'instrument_key': instrument_key,
                'open': data['ohlc_1min']['open'],
                'high': data['ohlc_1min']['high'],
                'low': data['ohlc_1min']['low'],
                'close': data['ohlc_1min']['close'],
                'volume': data['ohlc_1min']['volume'],
                'timestamp': data['ohlc_1min']['timestamp'],
                'data_source': 'feed_ohlc'
            }
        
        # Otherwise calculate from trade data
        trades = data['trades']
        if not trades:
            return None
        
        # Calculate OHLC from trades in last minute
        prices = [trade['price'] for trade in trades if trade['price'] is not None]
        volumes = [float(trade['quantity']) for trade in trades if trade['quantity'] is not None]
        
        if not prices:
            return None
        
        return {
            'instrument_key': instrument_key,
            'open': prices[0],  # First price in the minute
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],  # Last price in the minute
            'volume': sum(volumes),
            'timestamp': datetime.now(),
            'trade_count': len(trades),
            'data_source': 'calculated_from_trades'
        }

def get_market_summary(instrument_key="NSE_EQ|INE155A01022"):
    """Get a summary of market data for the instrument."""
    with data_lock:
        if instrument_key not in market_data_store:
            return None
        
        data = market_data_store[instrument_key]
        return {
            'instrument_key': instrument_key,
            'latest_price': data['latest_price'],
            'latest_timestamp': data['latest_timestamp'],
            'trades_in_last_minute': len(data['trades']),
            'has_ohlc_data': data['ohlc_1min'] is not None
        }


async def fetch_market_data():
    """Fetch market data using WebSocket and print it."""

    # Create default SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Get market data feed authorization
    response = get_market_data_feed_authorize_v3()
    # Connect to the WebSocket with SSL context
    print(response)
    async with websockets.connect(response["data"]["authorized_redirect_uri"], ssl=ssl_context) as websocket:
        print('Connection established')

        await asyncio.sleep(1)  # Wait for 1 second

        # Data to be sent over the WebSocket
        data = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_EQ|INE155A01022"] # can add a list of instrument keys
            }
        }

        # Convert data to binary and send over WebSocket
        binary_data = json.dumps(data).encode('utf-8')
        await websocket.send(binary_data)

        # Continuously receive and decode data from WebSocket
        while True:
            message = await websocket.recv()
            decoded_data = decode_protobuf(message)

            # Convert the decoded data to a dictionary
            data_dict = MessageToDict(decoded_data)

            update_market_data(data_dict)

            # Print the dictionary representation
            # print(json.dumps(data_dict))

# Execute the function to fetch market data
# asyncio.run(fetch_market_data())
