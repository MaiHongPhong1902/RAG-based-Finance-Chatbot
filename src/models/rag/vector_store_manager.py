from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
from faiss import IndexFlatL2
import json
from pathlib import Path

class VectorStoreManager:
    """Manager for vector store updates"""
    
    def __init__(self, store_dir: str = "vector_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.index = None
        self.metadata = {}
        self.last_update = None
        self.update_threshold = 0.05  # 5% price change threshold
        
    def initialize(self):
        """Initialize or load existing vector store"""
        index_path = self.store_dir / "index.faiss"
        metadata_path = self.store_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            # Load existing index and metadata
            self.index = IndexFlatL2.read(str(index_path))
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.last_update = datetime.fromisoformat(self.metadata.get('last_update', ''))
        else:
            # Create new index
            self.index = IndexFlatL2(768)  # Assuming 768-dimensional vectors
            self.last_update = datetime.now()
            
    def should_update(self, new_data: Dict[str, Any]) -> bool:
        """Check if vector store should be updated based on price changes"""
        if not self.last_update:
            return True
            
        # Check time since last update
        time_since_update = datetime.now() - self.last_update
        if time_since_update > timedelta(hours=1):
            return True
            
        # Check price changes
        for symbol, data in new_data.items():
            if symbol in self.metadata.get('prices', {}):
                old_price = self.metadata['prices'][symbol]
                new_price = data.get('price', 0)
                price_change = abs(new_price - old_price) / old_price
                
                if price_change > self.update_threshold:
                    return True
                    
        return False
        
    def update(self, vectors: np.ndarray, metadata: Dict[str, Any]):
        """Update vector store with new data"""
        if self.index is None:
            self.initialize()
            
        # Update index
        self.index = IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.astype('float32'))
        
        # Update metadata
        self.metadata = {
            'last_update': datetime.now().isoformat(),
            'prices': metadata.get('prices', {}),
            'symbols': metadata.get('symbols', [])
        }
        
        # Save to disk
        self.index.write(str(self.store_dir / "index.faiss"))
        with open(self.store_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.index is None:
            return []
            
        distances, indices = self.index.search(
            query_vector.astype('float32').reshape(1, -1),
            k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata.get('symbols', [])):
                results.append({
                    'symbol': self.metadata['symbols'][idx],
                    'distance': float(distances[0][i])
                })
                
        return results 