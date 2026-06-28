import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.helpers import compute_signal_score, TICKERS

class SignalGenerator:
    def __init__(self):
        self.history = []
    def compute_signal(self, ticker, sentiment_score, tweet_df, news_df, price_df):
        tweet_volume = len(tweet_df) if tweet_df is not None else 0
        news_count = len(news_df) if news_df is not None else 0
        if price_df is not None and len(price_df) > 1:
            momentum = (price_df["Close"].iloc[-1] / price_df["Close"].iloc[0] - 1) * 100
            volatility = price_df["Close"].pct_change().std() * 100
        else:
            momentum, volatility = 0, 0
        raw = compute_signal_score(sentiment_score, tweet_volume, momentum, news_count)
        if raw > 0.4: signal, action = "Strong Buy", "Enter long position"
        elif raw > 0.1: signal, action = "Buy", "Accumulate position"
        elif raw < -0.4: signal, action = "Strong Sell", "Exit / Short position"
        elif raw < -0.1: signal, action = "Sell", "Reduce position"
        else: signal, action = "Neutral", "Hold"
        data = {"ticker": ticker, "timestamp": datetime.now(), "signal": signal, "strength": round(abs(raw), 3), "raw_score": round(raw, 3), "action": action, "sentiment_score": round(sentiment_score, 3), "tweet_volume": tweet_volume, "news_count": news_count, "price_momentum": round(momentum, 2), "volatility": round(volatility, 2)}
        self.history.append(data)
        return data
    def compute_multi_ticker_signals(self, sentiment_results, data_fetcher, tickers=None):
        if tickers is None: tickers = list(TICKERS.keys())
        signals = []
        for ticker in tickers:
            ts = sentiment_results[sentiment_results["ticker"] == ticker]
            sentiment = ts["sentiment_score"].mean() if len(ts) > 0 else 0
            signals.append(self.compute_signal(ticker, sentiment, data_fetcher.get_tweets_for_ticker(ticker), data_fetcher.get_news_for_ticker(ticker), data_fetcher.get_price_data(ticker)))
        return pd.DataFrame(signals)
    def get_signal_history(self, n=20):
        if not self.history: return pd.DataFrame()
        return pd.DataFrame(self.history).tail(n).sort_values("timestamp", ascending=False)
