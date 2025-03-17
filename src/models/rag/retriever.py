from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

from config.config import Config
from src.data.storage.database import DatabaseManager
from src.models.rag.vector_store_manager import VectorStoreManager

class FinanceRetriever:
    """Retriever class for financial data using RAG"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.vector_store = VectorStoreManager()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
    def initialize(self):
        """Initialize the retriever"""
        self.db.initialize()
        self.vector_store.initialize()
        
    def _create_market_context(self, symbol: str) -> str:
        """Create market context from latest data"""
        latest_data = self.db.get_latest_price(symbol)
        if not latest_data:
            return ""
            
        # Get historical data for the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        historical_data = self.db.get_historical_prices(symbol, start_time, end_time)
        
        context = f"""
        Current market data for {symbol}:
        Current Price: ${latest_data['price']:.2f}
        24h Change: {latest_data['change_24h']:.2f}%
        24h Volume: ${latest_data['volume']:,.2f}
        
        Market Indicators:
        """
        
        if latest_data.get('indicators'):
            indicators = latest_data['indicators']
            context += f"""
            Bid-Ask Spread: ${indicators.get('bid_ask_spread', 0):.4f}
            Order Book Imbalance: {indicators.get('order_book_imbalance', 0):.2f}
            """
            
        if not historical_data.empty:
            price_change = (
                (latest_data['price'] - historical_data.iloc[0]['price']) 
                / historical_data.iloc[0]['price'] * 100
            )
            volatility = historical_data['price'].std()
            context += f"""
            Price Trend Analysis:
            24h Price Change: {price_change:.2f}%
            24h Volatility: {volatility:.2f}
            24h High: ${historical_data['high'].max():.2f}
            24h Low: ${historical_data['low'].min():.2f}
            """
            
        return context
        
    def _create_documents(self, symbol: str) -> List[Document]:
        """Create documents for vector store"""
        context = self._create_market_context(symbol)
        chunks = self.text_splitter.split_text(context)
        
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            )
            documents.append(doc)
            
        return documents
        
    def update_vector_store(self, symbols: List[str]):
        """Update vector store with new market data"""
        # Get current market data
        market_data = {}
        vectors = []
        symbols_list = []
        
        for symbol in symbols:
            data = self.db.get_market_data(symbol)
            if data:
                market_data[symbol] = data
                # Create vector representation
                text = self._format_market_data(data)
                vector = self.encoder.encode([text])[0]
                vectors.append(vector)
                symbols_list.append(symbol)
                
        if not vectors:
            return
            
        # Check if update is needed
        if not self.vector_store.should_update(market_data):
            return
            
        # Update vector store
        vectors_array = np.array(vectors)
        metadata = {
            'prices': {s: d.get('price', 0) for s, d in market_data.items()},
            'symbols': symbols_list
        }
        self.vector_store.update(vectors_array, metadata)
        
    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve relevant context for query"""
        # Encode query
        query_vector = self.encoder.encode([query])[0]
        
        # Search vector store
        results = self.vector_store.search(query_vector)
        
        # Get context from database
        context = []
        for result in results:
            symbol = result['symbol']
            market_data = self.db.get_market_data(symbol)
            if market_data:
                context.append(self._format_market_data(market_data))
                
        return context
        
    def get_price_prediction_context(self, symbol: str) -> Dict[str, Any]:
        """Get context for price prediction"""
        return self.db.get_market_data(symbol)
        
    def _format_market_data(self, data: Dict[str, Any]) -> str:
        """Format market data for vector representation"""
        formatted = f"Symbol: {data.get('symbol', '')}\n"
        formatted += f"Price: ${data.get('price', 0):.2f}\n"
        formatted += f"24h Change: {data.get('price_change_24h', 0):.2f}%\n"
        formatted += f"24h Volume: {data.get('volume_24h', 0):.2f}\n"
        formatted += f"24h High: ${data.get('high_24h', 0):.2f}\n"
        formatted += f"24h Low: ${data.get('low_24h', 0):.2f}\n"
        return formatted
