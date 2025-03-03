import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException

from .base_collector import BaseDataCollector
from config.config import Config

class BinanceDataCollector(BaseDataCollector):
    """Data collector for Binance cryptocurrency exchange"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        
    async def initialize(self):
        """Initialize Binance client"""
        self.client = await AsyncClient.create(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET
        )
        
    async def close(self):
        """Close Binance client connection"""
        if self.client:
            await self.client.close_connection()
            
    async def fetch_current_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price and 24h statistics for a given symbol"""
        try:
            ticker = await self.client.get_ticker(symbol=symbol)
            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'change_24h': float(ticker['priceChangePercent']),
                'volume_24h': float(ticker['volume']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'timestamp': datetime.now()
            }
        except BinanceAPIException as e:
            print(f"Error fetching current price for {symbol}: {str(e)}")
            return None
            
    async def fetch_historical_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch historical kline/candlestick data"""
        try:
            klines = await self.client.get_historical_klines(
                symbol=symbol,
                interval=self.client.KLINE_INTERVAL_1HOUR,
                start_str=str(int(start_time.timestamp() * 1000)),
                end_str=str(int(end_time.timestamp() * 1000))
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df if self.validate_data(df) else pd.DataFrame()
            
        except BinanceAPIException as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    async def fetch_market_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetch market indicators including order book and recent trades"""
        try:
            # Get order book
            depth = await self.client.get_order_book(symbol=symbol)
            
            # Get recent trades
            trades = await self.client.get_recent_trades(symbol=symbol)
            
            # Calculate market indicators
            bid_ask_spread = (float(depth['asks'][0][0]) - float(depth['bids'][0][0]))
            order_book_imbalance = self._calculate_order_book_imbalance(depth)
            
            return {
                'symbol': symbol,
                'bid_ask_spread': bid_ask_spread,
                'order_book_imbalance': order_book_imbalance,
                'order_book_depth': len(depth['bids']),
                'recent_trades_count': len(trades),
                'timestamp': datetime.now()
            }
            
        except BinanceAPIException as e:
            print(f"Error fetching market indicators for {symbol}: {str(e)}")
            return None
            
    def _calculate_order_book_imbalance(self, depth: Dict) -> float:
        """Calculate order book imbalance"""
        bids_volume = sum(float(bid[1]) for bid in depth['bids'][:10])
        asks_volume = sum(float(ask[1]) for ask in depth['asks'][:10])
        
        if asks_volume == 0:
            return 0
            
        return (bids_volume - asks_volume) / (bids_volume + asks_volume)
