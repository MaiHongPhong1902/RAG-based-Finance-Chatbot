from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config.config import Config
from src.data.storage.database import DatabaseManager

class FinanceRetriever:
    """Retriever class for financial data using RAG"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDINGS_MODEL
        )
        self.db = DatabaseManager()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
    def initialize(self):
        """Initialize the retriever"""
        self.db.initialize()
        
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
        """Update vector store with latest market data"""
        all_documents = []
        for symbol in symbols:
            documents = self._create_documents(symbol)
            all_documents.extend(documents)
            
        if not all_documents:
            return
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                all_documents,
                self.embeddings
            )
        else:
            self.vector_store.add_documents(all_documents)
            
    def retrieve_context(self, query: str, k: int = None) -> List[str]:
        """Retrieve relevant context for a query"""
        if self.vector_store is None:
            return []
            
        k = k or Config.TOP_K_RESULTS
        documents = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in documents]
        
    def get_price_prediction_context(self, symbol: str) -> Dict[str, Any]:
        """Get context for price prediction"""
        latest_data = self.db.get_latest_price(symbol)
        if not latest_data:
            return {}
            
        # Get historical data for analysis
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)  # Last 7 days
        historical_data = self.db.get_historical_prices(symbol, start_time, end_time)
        
        if historical_data.empty:
            return {}
            
        # Calculate technical indicators
        context = {
            'current_price': latest_data['price'],
            'price_history': historical_data['price'].tolist(),
            'volume_history': historical_data['volume'].tolist(),
            'timestamps': [ts.isoformat() for ts in historical_data.index],
            'market_indicators': latest_data.get('indicators', {}),
            'change_24h': latest_data['change_24h']
        }
        
        return context
