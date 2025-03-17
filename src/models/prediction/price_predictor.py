import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class PricePredictor:
    """LSTM-based price predictor"""
    
    def __init__(self):
        self.model = None
        self.sequence_length = 24  # 24 hours of data
        self.feature_dim = 5  # price, change, volume, high, low
        
    def build_model(self):
        """Build LSTM model"""
        if self.model is not None:
            return
            
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, self.feature_dim)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)  # price and confidence
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def prepare_sequence(self, data: np.ndarray) -> np.ndarray:
        """Prepare sequence data for prediction"""
        if len(data) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = np.zeros((self.sequence_length - len(data), self.feature_dim))
            data = np.vstack([padding, data])
            
        # Reshape for LSTM
        return data.reshape(1, self.sequence_length, self.feature_dim)
        
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price for single symbol"""
        # Extract features
        features = np.array([
            data.get('price', 0),
            data.get('price_change_24h', 0),
            data.get('volume_24h', 0),
            data.get('high_24h', 0),
            data.get('low_24h', 0)
        ])
        
        # Prepare sequence
        sequence = self.prepare_sequence(features)
        
        # Get prediction
        prediction = self.predict_sequence(sequence)
        
        return {
            'price': float(prediction[0]),
            'confidence': float(prediction[1])
        }
        
    def predict_batch(self, batch_data: np.ndarray) -> List[Dict[str, Any]]:
        """Predict prices for batch of data"""
        if self.model is None:
            self.build_model()
            
        # Prepare sequences
        sequences = []
        for data in batch_data:
            sequence = self.prepare_sequence(data)
            sequences.append(sequence)
            
        # Stack sequences
        sequences = np.vstack(sequences)
        
        # Get predictions
        predictions = self.model.predict(sequences)
        
        # Format results
        results = []
        for pred in predictions:
            results.append({
                'price': float(pred[0]),
                'confidence': float(pred[1])
            })
            
        return results
        
    def predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Predict price for sequence data"""
        if self.model is None:
            self.build_model()
            
        return self.model.predict(sequence)[0]
        
    def train(self, data: Dict[str, Any]):
        """Train model with new data"""
        if self.model is None:
            self.build_model()
            
        # Extract features
        features = np.array([
            data.get('price', 0),
            data.get('price_change_24h', 0),
            data.get('volume_24h', 0),
            data.get('high_24h', 0),
            data.get('low_24h', 0)
        ])
        
        # Prepare sequence
        sequence = self.prepare_sequence(features)
        
        # Create target (next price)
        target = np.array([data.get('price', 0), 1.0])  # price and confidence
        
        # Train model
        self.model.fit(
            sequence,
            target.reshape(1, 2),
            epochs=1,
            verbose=0
        )
        
    def _prepare_data(
        self,
        data: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """Prepare data for prediction"""
        # Extract features
        price_history = np.array(data['price_history'])
        volume_history = np.array(data['volume_history'])
        
        # Combine features
        features = np.column_stack((price_history, volume_history))
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X = []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            
        return np.array(X), data['current_price']
        
    def analyze_market_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment based on available data"""
        price_history = np.array(data['price_history'])
        current_price = data['current_price']
        
        # Calculate basic technical indicators
        sma_24h = np.mean(price_history[-24:])
        price_change_24h = data['change_24h']
        
        # Analyze order book if available
        order_book_sentiment = 0
        if 'market_indicators' in data and 'order_book_imbalance' in data['market_indicators']:
            order_book_sentiment = data['market_indicators']['order_book_imbalance']
            
        # Determine trend
        trend = "neutral"
        if current_price > sma_24h and price_change_24h > 0:
            trend = "bullish"
        elif current_price < sma_24h and price_change_24h < 0:
            trend = "bearish"
            
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (
            np.tanh(price_change_24h / 10) * 0.5 +  # Price change contribution
            np.tanh((current_price - sma_24h) / sma_24h) * 0.3 +  # SMA contribution
            order_book_sentiment * 0.2  # Order book contribution
        )
        
        return {
            'trend': trend,
            'sentiment_score': sentiment_score,
            'technical_indicators': {
                'sma_24h': sma_24h,
                'price_change_24h': price_change_24h
            }
        }
