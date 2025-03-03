from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

from src.data.collectors.binance_collector import BinanceDataCollector
from src.models.rag.retriever import FinanceRetriever
from src.models.prediction.price_predictor import PricePredictor
from config.config import Config

class FinanceChatbot:
    """Chatbot for financial predictions and analysis"""
    
    def __init__(self):
        self.collector = BinanceDataCollector()
        self.retriever = FinanceRetriever()
        self.predictor = PricePredictor()
        
        # Initialize language model
        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_length=512
            )
        )
        
        # Initialize prompt templates
        self.prediction_template = PromptTemplate(
            input_variables=["context", "predictions", "sentiment"],
            template="""
            Based on the current market data and analysis:
            
            Market Context:
            {context}
            
            Price Predictions:
            {predictions}
            
            Market Sentiment:
            {sentiment}
            
            Provide a detailed analysis and trading insights in a clear and concise manner.
            """
        )
        
    async def initialize(self):
        """Initialize all components"""
        await self.collector.initialize()
        self.retriever.initialize()
        
    async def update_market_data(self):
        """Update market data for all configured symbols"""
        for symbol in Config.SYMBOLS:
            # Fetch current price and market indicators
            price_data = await self.collector.fetch_current_price(symbol)
            if price_data:
                indicators = await self.collector.fetch_market_indicators(symbol)
                if indicators:
                    price_data['indicators'] = indicators
                    
                # Store data
                self.retriever.db.store_price_data(price_data)
                
        # Update vector store
        self.retriever.update_vector_store(Config.SYMBOLS)
        
    def _format_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """Format predictions for response generation"""
        formatted = "Price Predictions:\n"
        for pred in predictions:
            timestamp = datetime.fromisoformat(pred['timestamp'])
            formatted += f"- {timestamp.strftime('%Y-%m-%d %H:%M')}: "
            formatted += f"${pred['price']:.2f} (Confidence: {pred['confidence']:.2%})\n"
        return formatted
        
    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        """Format sentiment analysis for response generation"""
        formatted = f"Market Trend: {sentiment['trend'].capitalize()}\n"
        formatted += f"Sentiment Score: {sentiment['sentiment_score']:.2f}\n\n"
        
        formatted += "Technical Indicators:\n"
        indicators = sentiment['technical_indicators']
        formatted += f"- 24h SMA: ${indicators['sma_24h']:.2f}\n"
        formatted += f"- 24h Price Change: {indicators['price_change_24h']:.2f}%\n"
        
        return formatted
        
    async def generate_response(self, query: str) -> str:
        """Generate response for user query"""
        try:
            # Extract symbol from query (simple approach, can be improved)
            symbol = None
            for s in Config.SYMBOLS:
                if s.split('/')[0].lower() in query.lower():
                    symbol = s
                    break
                    
            if not symbol:
                return "Please specify a valid cryptocurrency symbol in your query."
                
            # Retrieve relevant context
            context = self.retriever.retrieve_context(query)
            if not context:
                return "I don't have enough market data to answer your query."
                
            # Get prediction context
            pred_context = self.retriever.get_price_prediction_context(symbol)
            if not pred_context:
                return "Unable to access current market data for predictions."
                
            # Train predictor with latest data
            self.predictor.train(pred_context)
            
            # Generate predictions
            predictions = self.predictor.predict(pred_context)
            
            # Analyze sentiment
            sentiment = self.predictor.analyze_market_sentiment(pred_context)
            
            # Generate response using language model
            prompt = self.prediction_template.format(
                context="\n".join(context),
                predictions=self._format_predictions(predictions),
                sentiment=self._format_sentiment(sentiment)
            )
            
            response = self.llm(prompt)
            
            # Add disclaimer
            response += "\n\nDisclaimer: This analysis is for informational purposes only. "
            response += "Cryptocurrency trading involves substantial risk of loss. "
            response += "Past performance does not guarantee future results."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
            
    async def close(self):
        """Clean up resources"""
        await self.collector.close()
