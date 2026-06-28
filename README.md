# Sentiment Trade

Real-time sentiment analysis and trading signal generator using BERT, LSTM, CNN, and ANN models.

## Architecture

```
News APIs (Finnhub, GNews, AlphaVantage) ─┐
                                          ├─→ BERT + VADER + TextBlob
Social APIs (Stocktwits) ─────────────────┘       │
                                                  ▼
                                          Signal Generator
                                          (sentiment + volume + momentum)
                                                  │
                                                  ▼
                                          ANN Classifier (optional)
                                          Buy / Sell / Neutral
                                                  │
                                                  ▼
                                          Portfolio Tracker
```

## Features

- **Sentiment Analysis** — Ensemble of BERT (50%), VADER (30%), TextBlob (20%)
- **Trading Signals** — Multi-factor scoring: sentiment, tweet volume, price momentum, news count
- **Deep Learning Models** — BiLSTM, CNN on BERT embeddings, 3-layer ANN classifier
- **Live Dashboard** — Streamlit with 7 tabs: Overview, News, Sentiment, Signals, Portfolio, Models, Pipeline
- **Real Data** — Finnhub, GNews, AlphaVantage for news; Stocktwits for social; yfinance for prices

## Setup

```bash
# Clone and install
git clone <repo-url>
cd sentrade
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env
echo "FINNHUB_API_KEY=your_key" >> .env
echo "GNEWS_API_KEY=your_key" >> .env
echo "ALPHAVANTAGE_API_KEY=your_key" >> .env

# Run
streamlit run app.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit, Plotly |
| NLP | BERT (DistilBERT), VADER, TextBlob |
| Deep Learning | PyTorch (LSTM, CNN, ANN) |
| Summarization | BART (facebook/bart-large-cnn) |
| Data | Finnhub, GNews, AlphaVantage, yfinance, Stocktwits |
