import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import schedule
import threading
import time

from src.models.chatbot import FinanceChatbot
from config.config import Config

app = FastAPI(title="Finance Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = FinanceChatbot()

class Query(BaseModel):
    """Query model for chat endpoint"""
    text: str

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot and start data update scheduler"""
    await chatbot.initialize()
    await chatbot.update_market_data()
    
    # Start scheduler in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    # Schedule market data updates
    schedule.every(Config.UPDATE_INTERVAL).seconds.do(
        lambda: asyncio.run(chatbot.update_market_data())
    )
    
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    await chatbot.close()

@app.post("/chat")
async def chat(query: Query) -> Dict[str, Any]:
    """Chat endpoint for user queries"""
    try:
        response = await chatbot.generate_response(query.text)
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )
