## RAG-based Finance Chatbot

**A RAG-based Finance Chatbot designed to provide accurate and timely cryptocurrency price predictions. By leveraging real-time data, historical analysis, and machine learning, this chatbot delivers informed responses about cryptocurrency markets.

```markdown
## Project Structure

```plaintext
chatbot/
├── src/
│   ├── data/
│   │   ├── collectors/         # Binance data collection scripts
│   │   └── storage/            # Database interaction modules
│   ├── models/
│   │   ├── prediction/         # LSTM-based price prediction logic
│   │   ├── rag/                # RAG implementation for retrieving and generating answers
│   │   └── chatbot.py          # Core chatbot functionalities
│   └── main.py                 # FastAPI application entry point
├── config/
│   └── config.py               # Configuration settings
├── requirements.txt            # Project dependencies
└── .env.example                # Template for environment variables
```

## Features

- **Real-time data collection** from Binance
- **PostgreSQL integration** for historical data storage and analysis
- **RAG-based answers** using FAISS and HuggingFace embeddings
- **LSTM price prediction** with confidence scores
- **Market sentiment analysis**
- **Automated data updates** (default: every 5 minutes)
- **RESTful API** powered by FastAPI

## Getting Started

### Prerequisites

- Python 3.8 or newer
- A running PostgreSQL database
- Binance API credentials

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MaiHongPhong1902/RAG-based-Finance-Chatbot.git
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include your Binance API keys, database credentials, and other settings.

### Database Setup

- Install PostgreSQL (if not already installed)
- Create a database named `finance_bot`
- Ensure `.env` contains the correct database connection details

### Run the Application

```bash
python -m src.main
```

### Accessing the API

- **Base URL:** `http://localhost:8000`
- **Swagger documentation:** `http://localhost:8000/docs`

**Endpoints:**
- `POST /chat`: Send a query to the chatbot
- `GET /health`: Check if the service is running

## Example Queries

- **General market info:**  
  "What's the current price of BTC?"  
  "What’s the market sentiment for ETH?"
  
- **Price predictions:**  
  "Can you predict BTC price for the next 24 hours?"
  
- **Market trends:**  
  "Analyze the current market trends for BNB."


## License

MIT License

Copyright (c) 2024 Mai Hong Phong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

- **Mai Hong Phong**
  - Email: maihongphong.work@gmail.com
  - Phone: 0865243215
  - GitHub: [@MaiHongPhong1902](https://github.com/MaiHongPhong1902)
  - University: Ho Chi Minh City University of Technology and Education
