import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from binance.client import AsyncClient, Client
from binance.exceptions import BinanceAPIException
import asyncio

from .base_collector import BaseDataCollector
from config.config import Config
from src.data.cache import MarketDataCache

class BinanceDataCollector(BaseDataCollector):
    """Data collector for Binance cryptocurrency exchange"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.cache = MarketDataCache()
        
    async def initialize(self):
        """Initialize Binance client"""
        self.client = Client(
            Config.BINANCE_API_KEY,
            Config.BINANCE_API_SECRET
        )
        
    async def close(self):
        """Close Binance client connection"""
        if self.client:
            await self.client.close_connection()
            
    async def fetch_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current price data for symbol"""
        # Check cache first
        cached_data = self.cache.get_cached_data(symbol)
        if cached_data:
            return cached_data
            
        try:
            # Fetch from Binance
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price_data = {
                'symbol': symbol,
                'price': float(ticker['price']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self.cache.set_cached_data(symbol, price_data)
            return price_data
            
        except BinanceAPIException as e:
            print(f"Error fetching price for {symbol}: {str(e)}")
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
            
    async def fetch_market_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market indicators for symbol"""
        try:
            # Get 24h ticker
            ticker_24h = self.client.get_ticker(symbol=symbol)
            
            indicators = {
                'price_change_24h': float(ticker_24h['priceChangePercent']),
                'volume_24h': float(ticker_24h['volume']),
                'high_24h': float(ticker_24h['highPrice']),
                'low_24h': float(ticker_24h['lowPrice']),
                'timestamp': datetime.now().isoformat()
            }
            
            return indicators
            
        except BinanceAPIException as e:
            print(f"Error fetching indicators for {symbol}: {str(e)}")
            return None
            
    def _calculate_order_book_imbalance(self, depth: Dict) -> float:
        """Calculate order book imbalance"""
        bids_volume = sum(float(bid[1]) for bid in depth['bids'][:10])
        asks_volume = sum(float(ask[1]) for ask in depth['asks'][:10])
        
        if asks_volume == 0:
            return 0
            
        return (bids_volume - asks_volume) / (bids_volume + asks_volume)
