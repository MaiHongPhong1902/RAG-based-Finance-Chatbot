from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class BaseDataCollector(ABC):
    """Base class for all data collectors"""
    
    def __init__(self):
        self.data = pd.DataFrame()
        
    @abstractmethod
    async def fetch_current_price(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price for a given symbol"""
        pass
        
    @abstractmethod
    async def fetch_historical_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical data for a given symbol and time range"""
        pass
        
    @abstractmethod
    async def fetch_market_indicators(self, symbol: str) -> Dict[str, Any]:
        """Fetch market indicators for a given symbol"""
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the collected data"""
        if df.empty:
            return False
        if df.isnull().values.any():
            return False
        return True
