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
    """Price prediction model using LSTM"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 24  # Use 24 hours of data to predict
        self.feature_columns = ['price', 'volume']
        
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
        
    def _build_model(self, input_shape: tuple):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return model
        
    def train(self, training_data: Dict[str, Any]):
        """Train the prediction model"""
        X, current_price = self._prepare_data(training_data)
        
        if len(X) < self.sequence_length:
            raise ValueError("Insufficient data for training")
            
        if self.model is None:
            self.model = self._build_model(
                input_shape=(self.sequence_length, len(self.feature_columns))
            )
            
        # Use the last sequence for validation
        X_train = X[:-1]
        y_train = X[1:, -1, 0]  # Use the next price as target
        
        self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
    def predict(
        self,
        data: Dict[str, Any],
        prediction_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Generate price predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        X, current_price = self._prepare_data(data)
        
        if len(X) < 1:
            raise ValueError("Insufficient data for prediction")
            
        # Use the last sequence for prediction
        last_sequence = X[-1:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(prediction_hours):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            
            # Scale back to original range
            scaled_pred = np.zeros((1, len(self.feature_columns)))
            scaled_pred[0, 0] = next_pred[0]  # Price prediction
            # Use last volume for simplicity
            scaled_pred[0, 1] = current_sequence[0, -1, 1]
            
            original_scale_pred = self.scaler.inverse_transform(scaled_pred)
            predicted_price = original_scale_pred[0, 0]
            
            # Calculate confidence based on historical volatility
            price_history = np.array(data['price_history'])
            volatility = np.std(price_history) / np.mean(price_history)
            confidence = max(0, min(1, 1 - volatility))
            
            # Store prediction
            prediction_time = datetime.now() + timedelta(hours=i+1)
            predictions.append({
                'timestamp': prediction_time.isoformat(),
                'price': predicted_price,
                'confidence': confidence
            })
            
            # Update sequence for next prediction
            new_sequence = np.zeros((1, self.sequence_length, len(self.feature_columns)))
            new_sequence[0, :-1] = current_sequence[0, 1:]
            new_sequence[0, -1, 0] = next_pred[0]  # Predicted price
            new_sequence[0, -1, 1] = current_sequence[0, -1, 1]  # Last volume
            current_sequence = new_sequence
            
        return predictions
        
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
