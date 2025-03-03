from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

from config.config import Config

Base = declarative_base()

class PriceData(Base):
    """Model for storing price data"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    high = Column(Float)
    low = Column(Float)
    change_24h = Column(Float)
    indicators = Column(JSON)  # Store market indicators as JSON

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        
    def initialize(self):
        """Initialize database connection"""
        db_url = f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def store_price_data(self, data: Dict[str, Any]):
        """Store price data in the database"""
        session = self.Session()
        try:
            price_data = PriceData(
                symbol=data['symbol'],
                timestamp=data['timestamp'],
                price=data['price'],
                volume=data.get('volume_24h'),
                high=data.get('high_24h'),
                low=data.get('low_24h'),
                change_24h=data.get('change_24h'),
                indicators=data.get('indicators', {})
            )
            session.add(price_data)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error storing price data: {str(e)}")
        finally:
            session.close()
            
    def store_historical_data(self, symbol: str, df: pd.DataFrame):
        """Store historical data in the database"""
        session = self.Session()
        try:
            for index, row in df.iterrows():
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=index,
                    price=row['close'],
                    volume=row['volume'],
                    high=row['high'],
                    low=row['low'],
                    change_24h=None,  # Not available in historical data
                    indicators={}
                )
                session.add(price_data)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error storing historical data: {str(e)}")
        finally:
            session.close()
            
    def get_latest_price(self, symbol: str) -> Dict[str, Any]:
        """Get the latest price data for a symbol"""
        session = self.Session()
        try:
            result = session.query(PriceData)\
                .filter(PriceData.symbol == symbol)\
                .order_by(PriceData.timestamp.desc())\
                .first()
            
            if result:
                return {
                    'symbol': result.symbol,
                    'price': result.price,
                    'timestamp': result.timestamp,
                    'volume': result.volume,
                    'change_24h': result.change_24h,
                    'indicators': result.indicators
                }
            return None
        finally:
            session.close()
            
    def get_historical_prices(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get historical price data for a symbol"""
        session = self.Session()
        try:
            results = session.query(PriceData)\
                .filter(
                    PriceData.symbol == symbol,
                    PriceData.timestamp.between(start_time, end_time)
                )\
                .order_by(PriceData.timestamp.asc())\
                .all()
            
            if results:
                data = []
                for result in results:
                    data.append({
                        'timestamp': result.timestamp,
                        'price': result.price,
                        'volume': result.volume,
                        'high': result.high,
                        'low': result.low
                    })
                return pd.DataFrame(data).set_index('timestamp')
            return pd.DataFrame()
        finally:
            session.close()
