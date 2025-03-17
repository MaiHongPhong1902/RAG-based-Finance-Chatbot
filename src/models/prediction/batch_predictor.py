from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from src.models.prediction.price_predictor import PricePredictor

class BatchPredictor:
    """Batch processor for price predictions"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.predictor = PricePredictor()
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
    def prepare_batch(self, data_list: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare batch data for prediction"""
        batch_data = []
        for data in data_list:
            # Extract features
            features = [
                data.get('price', 0),
                data.get('price_change_24h', 0),
                data.get('volume_24h', 0),
                data.get('high_24h', 0),
                data.get('low_24h', 0)
            ]
            batch_data.append(features)
            
        return np.array(batch_data)
        
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict prices for a batch of symbols"""
        results = []
        
        # Process in batches
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batch_data = self.prepare_batch(batch)
            
            # Get predictions
            predictions = self.predictor.predict_batch(batch_data)
            
            # Format results
            for j, pred in enumerate(predictions):
                symbol = batch[j].get('symbol', '')
                results.append({
                    'symbol': symbol,
                    'predicted_price': float(pred['price']),
                    'confidence': float(pred['confidence']),
                    'timestamp': datetime.now().isoformat()
                })
                
        return results
        
    def get_cached_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get cached prediction for symbol"""
        if symbol in self.cache:
            cached_data = self.cache[symbol]
            if datetime.now() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['prediction']
        return None
        
    def cache_prediction(self, symbol: str, prediction: Dict[str, Any]):
        """Cache prediction for symbol"""
        self.cache[symbol] = {
            'prediction': prediction,
            'timestamp': datetime.now()
        }
        
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear prediction cache"""
        if symbol:
            self.cache.pop(symbol, None)
        else:
            self.cache.clear() 