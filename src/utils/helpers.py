import numpy as np

# Stocks tracked by the system
TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
}

def compute_signal_score(sentiment, tweet_vol, momentum, news_count):
    """
    Calculate a combined trading signal score.

    Parameters:
    sentiment   : sentiment score (-1 to 1)
    tweet_vol   : number of tweets/posts
    momentum    : price momentum (%)
    news_count  : number of news articles

    Returns:
    score between -1 and 1
    """

    score = (
        0.35 * sentiment +
        0.20 * np.tanh(tweet_vol / 20) +
        0.25 * np.tanh(momentum / 5) +
        0.20 * np.tanh(news_count / 10)
    )

    return np.clip(score, -1, 1)