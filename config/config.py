import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'finance_bot')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')

    # Elasticsearch Configuration
    ES_HOST = os.getenv('ES_HOST', 'localhost')
    ES_PORT = os.getenv('ES_PORT', '9200')

    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models')
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))

    # Data Collection Configuration
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '300'))  # 5 minutes
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']  # Default symbols to track

    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
