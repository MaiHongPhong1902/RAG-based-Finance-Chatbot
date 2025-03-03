# RAG-based Finance Chatbot

A RAG-based Finance Chatbot for cryptocurrency price predictions that combines real-time data collection, historical analysis, and machine learning to provide informed responses about cryptocurrency markets.

## Project Structure

```
chatbot/
├── src/
│   ├── data/
│   │   ├── collectors/         # Data collection from Binance
│   │   └── storage/           # Database operations
│   ├── models/
│   │   ├── prediction/        # LSTM-based price prediction
│   │   ├── rag/              # RAG implementation
│   │   └── chatbot.py        # Main chatbot logic
│   └── main.py               # FastAPI application
├── config/
│   └── config.py             # Configuration management
├── requirements.txt          # Project dependencies
└── .env.example             # Environment variables template
```

## Key Features

- Real-time cryptocurrency price data collection from Binance
- Historical price analysis and storage in PostgreSQL
- RAG-enhanced responses using FAISS and HuggingFace embeddings
- LSTM-based price predictions with confidence scores
- Market sentiment analysis
- Automated data updates via scheduler
- RESTful API interface using FastAPI

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Binance API credentials

### Installation

```bash
# Clone the repository and navigate to the project directory
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### Database Setup

- Install PostgreSQL if not already installed
- Create a new database named 'finance_bot'
- Update database credentials in .env file

### Running the Application

```bash
# Start the FastAPI server
python -m src.main
```

## API Usage

- The API will be available at http://localhost:8000
- Swagger documentation: http://localhost:8000/docs
- Main endpoints:
  - POST /chat: Send queries to the chatbot
  - GET /health: Check API health status

## Example Queries

- "What's the current price of BTC?"
- "Can you predict BTC price for the next 24 hours?"
- "What's the market sentiment for ETH?"
- "Analyze the current market trends for BNB"

## Important Notes

- The system requires valid Binance API credentials
- Price predictions include confidence scores and disclaimers
- Market data is automatically updated every 5 minutes (configurable)
- All predictions come with risk disclaimers
- The system uses RAG to provide context-aware responses based on current market conditions
#   R A G - b a s e d - F i n a n c e - C h a t b o t  
 