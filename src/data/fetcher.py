import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
GNEWS_BASE_URL = "https://gnews.io/api/v4"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2"

NEWS_COLUMNS = ["ticker", "headline", "summary", "source", "url", "timestamp"]
SOCIAL_COLUMNS = ["ticker", "text", "author", "source", "timestamp", "sentiment"]


class DataFetcher:
    def __init__(self, finnhub_api_key=None, gnews_api_key=None,
                 alphavantage_api_key=None, ticker_name_map=None):
        self.finnhub_api_key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.gnews_api_key = gnews_api_key or os.environ.get("GNEWS_API_KEY", "")
        self.alphavantage_api_key = alphavantage_api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self.ticker_name_map = ticker_name_map or {}

        self.news_data = {}      # keyed by ticker
        self.social_data = {}    # keyed by ticker (was: single twitter_data DataFrame)
        self.price_data = {}

    # ==================== NEWS: individual source fetchers ====================
    # Each returns a normalized DataFrame with NEWS_COLUMNS and never raises -
    # failures are logged and an empty frame is returned so other sources can
    # still succeed.

    def _fetch_finnhub_news(self, ticker, days_back=7):
        if not self.finnhub_api_key:
            print("[Finnhub] Skipped - FINNHUB_API_KEY not set.")
            return pd.DataFrame(columns=NEWS_COLUMNS)

        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days_back)
        params = {
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.finnhub_api_key,
        }

        try:
            resp = requests.get(f"{FINNHUB_BASE_URL}/company-news", params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json() or []
            return pd.DataFrame([{
                "ticker": ticker,
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", "Finnhub"),
                "url": a.get("url", ""),
                "timestamp": datetime.fromtimestamp(a.get("datetime", 0)),
            } for a in articles], columns=NEWS_COLUMNS)
        except requests.exceptions.RequestException as e:
            print(f"[Finnhub] Failed for {ticker}: {e}")
            return pd.DataFrame(columns=NEWS_COLUMNS)

    def _fetch_gnews_news(self, ticker, days_back=7, max_results=25):
        if not self.gnews_api_key:
            print("[GNews] Skipped - GNEWS_API_KEY not set.")
            return pd.DataFrame(columns=NEWS_COLUMNS)

        query = self.ticker_name_map.get(ticker, ticker)
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "q": query,
            "from": from_date,
            "lang": "en",
            "max": max_results,
            "apikey": self.gnews_api_key,
        }

        try:
            resp = requests.get(f"{GNEWS_BASE_URL}/search", params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return pd.DataFrame([{
                "ticker": ticker,
                "headline": a.get("title", ""),
                "summary": a.get("description", ""),
                "source": a.get("source", {}).get("name", "GNews"),
                "url": a.get("url", ""),
                "timestamp": pd.to_datetime(a.get("publishedAt")).to_pydatetime().replace(tzinfo=None)
                             if a.get("publishedAt") else datetime.now(),
            } for a in articles], columns=NEWS_COLUMNS)
        except requests.exceptions.RequestException as e:
            print(f"[GNews] Failed for {ticker}: {e}")
            return pd.DataFrame(columns=NEWS_COLUMNS)

    def _fetch_alphavantage_news(self, ticker, limit=50):
        if not self.alphavantage_api_key:
            print("[AlphaVantage] Skipped - ALPHAVANTAGE_API_KEY not set.")
            return pd.DataFrame(columns=NEWS_COLUMNS)

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": limit,
            "apikey": self.alphavantage_api_key,
        }

        try:
            resp = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "Note" in data or "Information" in data:
                # Rate limit or invalid call - AlphaVantage returns 200 with a message,
                # not an HTTP error, so this has to be checked explicitly.
                print(f"[AlphaVantage] API message: {data.get('Note') or data.get('Information')}")
                return pd.DataFrame(columns=NEWS_COLUMNS)

            feed = data.get("feed", [])
            rows = []
            for a in feed:
                tp = a.get("time_published", "")
                try:
                    ts = datetime.strptime(tp, "%Y%m%dT%H%M%S")
                except ValueError:
                    ts = datetime.now()
                rows.append({
                    "ticker": ticker,
                    "headline": a.get("title", ""),
                    "summary": a.get("summary", ""),
                    "source": a.get("source", "AlphaVantage"),
                    "url": a.get("url", ""),
                    "timestamp": ts,
                })
            return pd.DataFrame(rows, columns=NEWS_COLUMNS)
        except requests.exceptions.RequestException as e:
            print(f"[AlphaVantage] Failed for {ticker}: {e}")
            return pd.DataFrame(columns=NEWS_COLUMNS)

    # ==================== NEWS: unified entry point ====================

    def fetch_news_for_ticker(self, ticker, days_back=7, sources=None):
        """
        Fetches and merges news from multiple real sources for one ticker.
        sources: subset of ["finnhub", "gnews", "alphavantage"]; defaults to all three.
        """
        if sources is None:
            sources = ["finnhub", "gnews", "alphavantage"]

        fetchers = {
            "finnhub": lambda: self._fetch_finnhub_news(ticker, days_back),
            "gnews": lambda: self._fetch_gnews_news(ticker, days_back),
            "alphavantage": lambda: self._fetch_alphavantage_news(ticker),
        }

        frames = []
        for source in sources:
            if source not in fetchers:
                print(f"Unknown news source: {source}")
                continue
            frames.append(fetchers[source]())

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)


        if not combined.empty:
            if "base_sentiment" not in combined.columns:
                from textblob import TextBlob
                combined["base_sentiment"] = combined["headline"].apply(
                    lambda x: TextBlob(x).sentiment.polarity
                )
            combined = combined.sort_values("timestamp", ascending=False).reset_index(drop=True)

        self.news_data[ticker] = combined
        return combined

    def get_news_for_ticker(self, ticker, hours=24):
        if ticker not in self.news_data:
            self.fetch_news_for_ticker(ticker)

        df = self.news_data[ticker]
        if df.empty:
            return df

        cutoff = (datetime.now() - timedelta(hours=hours)).replace(tzinfo=None)
        return df[df["timestamp"] >= cutoff]

    # ==================== SOCIAL: individual source fetchers ====================

    def _fetch_stocktwits_posts(self, ticker, limit=30):
        url = f"{STOCKTWITS_BASE_URL}/streams/symbol/{ticker}.json"

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            messages = data.get("messages", [])[:limit]

            rows = []
            for m in messages:
                entities = m.get("entities") or {}
                sentiment_obj = entities.get("sentiment")
                sentiment = sentiment_obj.get("basic") if sentiment_obj else None
                user = m.get("user") or {}

                rows.append({
                    "ticker": ticker,
                    "text": m.get("body", ""),
                    "author": user.get("username", ""),
                    "source": "Stocktwits",
                "timestamp": pd.to_datetime(m.get("created_at")).to_pydatetime().replace(tzinfo=None)
                             if m.get("created_at") else datetime.now(),
                    "sentiment": sentiment,   # "Bullish", "Bearish", or None
                })
            return pd.DataFrame(rows, columns=SOCIAL_COLUMNS)

        except requests.exceptions.RequestException as e:
            print(f"[Stocktwits] Failed for {ticker}: {e}")
            return pd.DataFrame(columns=SOCIAL_COLUMNS)

    # Reddit not wired up yet - add a _fetch_reddit_posts() here when ready,
    # following the same pattern (normalize to SOCIAL_COLUMNS, never raise).

    # ==================== SOCIAL: unified entry point ====================

    def fetch_social_for_ticker(self, ticker, sources=None):
        """
        Fetches and merges social posts from multiple real sources for one ticker.
        sources: subset of ["stocktwits"] for now; will grow once Reddit is added.
        """
        if sources is None:
            sources = ["stocktwits"]

        fetchers = {
            "stocktwits": lambda: self._fetch_stocktwits_posts(ticker),
        }

        frames = []
        for source in sources:
            if source not in fetchers:
                print(f"Unknown social source: {source}")
                continue
            frames.append(fetchers[source]())

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SOCIAL_COLUMNS)


        if not combined.empty and "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp", ascending=False).reset_index(drop=True)

        self.social_data[ticker] = combined
        return combined

    def get_tweets_for_ticker(self, ticker, hours=24):
        """Kept this name for backward compatibility - now backed by Stocktwits, not Twitter."""
        if ticker not in self.social_data:
            self.fetch_social_for_ticker(ticker)

        df = self.social_data[ticker]
        if df.empty or "timestamp" not in df.columns:
            return df

        cutoff = datetime.now() - timedelta(hours=hours)
        return df[df["timestamp"] >= cutoff]

    def fetch_all(self):
        from src.utils.helpers import TICKERS
        for ticker in TICKERS:
            self.fetch_news_for_ticker(ticker)
            self.fetch_social_for_ticker(ticker)
        return self

    # ==================== PRICES (yfinance - real, unchanged) ====================

    def get_price_data(self, ticker, period="3mo"):
        if ticker not in self.price_data:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                df = pd.DataFrame({"Date": hist.index, "Close": hist["Close"].values})
                df["returns"] = df["Close"].pct_change()
                self.price_data[ticker] = df
            except Exception:
                print(f"[yfinance] Failed for {ticker}"); self.price_data[ticker] = pd.DataFrame()
        return self.price_data[ticker]