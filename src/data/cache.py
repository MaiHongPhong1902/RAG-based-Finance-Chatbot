from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

class MarketDataCache:
    """Cache manager for market data"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(minutes=5)  # Cache TTL in minutes
        
    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for symbol"""
        return self.cache_dir / f"{symbol.lower()}_cache.json"
        
    def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for symbol"""
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > self.cache_ttl:
                return None
                
            return cache_data['data']
            
        except Exception:
            return None
            
    def set_cached_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Cache market data for symbol"""
        cache_path = self._get_cache_path(symbol)
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception:
            pass  # Silently fail if cache write fails
            
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for symbol or all symbols"""
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*_cache.json"):
                cache_file.unlink() 