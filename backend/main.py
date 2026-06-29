import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Sentiment Trade API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = None
def get_state():
    global _state
    if _state is not None:
        return _state
    from src.data.fetcher import DataFetcher
    from src.models.sentiment_bert import VaderSentiment, TextBlobSentiment, BERT_Sentiment
    from src.models.summarizer import SimpleSummarizer
    from src.signals.generator import SignalGenerator
    from src.signals.portfolio import PortfolioTracker
    from src.utils.helpers import TICKERS

    df = DataFetcher()
    try:
        df.fetch_all()
    except:
        pass
    sg = SignalGenerator()
    pt = PortfolioTracker()
    vader = VaderSentiment()
    tb = TextBlobSentiment()
    bert = BERT_Sentiment()
    bert_loaded = False
    try:
        bert.load()
        bert_loaded = True
    except:
        pass
    _state = {
        "df": df, "sg": sg, "pt": pt,
        "vader": vader, "tb": tb, "bert": bert,
        "bert_loaded": bert_loaded, "TICKERS": TICKERS,
    }
    return _state

@app.get("/api/news/{ticker}")
def get_news(ticker: str, hours: int = 24):
    s = get_state()
    news = s["df"].get_news_for_ticker(ticker, hours=hours)
    if news.empty:
        return {"ticker": ticker, "articles": []}
    return {"ticker": ticker, "count": len(news), "articles": news.to_dict(orient="records")}

@app.get("/api/tweets/{ticker}")
def get_tweets(ticker: str, hours: int = 24):
    s = get_state()
    tweets = s["df"].get_tweets_for_ticker(ticker, hours=hours)
    if tweets.empty:
        return {"ticker": ticker, "posts": []}
    posts = tweets.to_dict(orient="records")
    for p in posts:
        p["textblob_sentiment"] = round(s["tb"].analyze(p.get("text", "")), 3)
        p["vader_sentiment"] = round(s["vader"].analyze(p.get("text", "")), 3)
    return {"ticker": ticker, "count": len(posts), "posts": posts}

@app.get("/api/price/{ticker}")
def get_price(ticker: str):
    s = get_state()
    price = s["df"].get_price_data(ticker)
    if price.empty:
        raise HTTPException(404, "No price data")
    price = price.tail(90).replace({np.nan: None})
    data = price.to_dict(orient="records")
    for d in data:
        if isinstance(d.get("Date"), pd.Timestamp):
            d["Date"] = d["Date"].isoformat()
    return {"ticker": ticker, "data": data}

@app.get("/api/sentiment/{ticker}")
def get_sentiment(ticker: str, hours: int = 24):
    s = get_state()
    news = s["df"].get_news_for_ticker(ticker, hours=hours)
    tweets = s["df"].get_tweets_for_ticker(ticker, hours=hours)
    articles = news.to_dict(orient="records") if not news.empty else []
    vader, tb, bert, bert_loaded = s["vader"], s["tb"], s["bert"], s["bert_loaded"]

    for a in articles:
        a["vader"] = round(vader.analyze(a.get("headline", "")), 3)
        a["textblob"] = round(tb.analyze(a.get("headline", "")), 3)
        if bert_loaded:
            a["bert"] = round(float(bert.analyze([a["headline"]])[0]), 3)
        else:
            a["bert"] = None
        a["ensemble"] = round(0.5 * (a["bert"] if a["bert"] else 0) + 0.3 * a["vader"] + 0.2 * a["textblob"], 3)

    tweet_scores = []
    if not tweets.empty:
        for t in tweets["text"]:
            tweet_scores.append({
                "vader": round(vader.analyze(t), 3),
                "textblob": round(tb.analyze(t), 3),
            })

    all_scores = [a["ensemble"] for a in articles if "ensemble" in a]
    if tweet_scores:
        all_scores.extend([t["textblob"] for t in tweet_scores])
    avg = round(np.mean(all_scores), 3) if all_scores else 0
    label = "Bullish" if avg > 0.15 else "Bearish" if avg < -0.15 else "Neutral"

    return {"ticker": ticker, "avg_sentiment": avg, "label": label,
            "articles": articles, "tweet_scores": tweet_scores}

@app.get("/api/signals")
def get_signals():
    s = get_state()
    all_news = s["df"].news_data
    if isinstance(all_news, dict):
        frames = [v for v in all_news.values() if isinstance(v, pd.DataFrame) and not v.empty]
        all_news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_news.empty and "base_sentiment" in all_news.columns:
        all_news["vader_sentiment"] = all_news["headline"].apply(lambda x: s["vader"].analyze(x))
        all_news["textblob_sentiment"] = all_news["headline"].apply(lambda x: s["tb"].analyze(x))
        all_news["sentiment_score"] = 0.4 * all_news["base_sentiment"] + 0.3 * all_news["vader_sentiment"] + 0.3 * all_news["textblob_sentiment"]
        signals = s["sg"].compute_multi_ticker_signals(all_news, s["df"])
        cp = {t: (s["df"].get_price_data(t)["Close"].iloc[-1] if not s["df"].get_price_data(t).empty else 100) for t in s["TICKERS"]}
        for _, sig in signals.iterrows():
            s["pt"].execute_signal(sig.to_dict(), cp.get(sig["ticker"], 100))
        s["pt"].update_equity(cp)
        return {"count": len(signals), "signals": signals.to_dict(orient="records")}
    return {"count": 0, "signals": []}

@app.get("/api/signals/history")
def signal_history():
    s = get_state()
    hist = s["sg"].get_signal_history(30)
    if hist.empty:
        return {"count": 0, "history": []}
    h = hist.replace({np.nan: None}).to_dict(orient="records")
    for r in h:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
    return {"count": len(h), "history": h}

@app.get("/api/portfolio")
def get_portfolio():
    s = get_state()
    cp = {t: (s["df"].get_price_data(t)["Close"].iloc[-1] if not s["df"].get_price_data(t).empty else 100) for t in s["TICKERS"]}
    perf = s["pt"].get_performance_summary(cp)
    def df_to_rows(d):
        return d.replace({np.nan: None}).to_dict(orient="records") if isinstance(d, pd.DataFrame) and not d.empty else []
    perf["positions_detail"] = df_to_rows(perf["positions_detail"])
    perf["trades"] = df_to_rows(perf["trades"])
    perf["equity_curve"] = df_to_rows(perf["equity_curve"])
    for r in perf["equity_curve"]:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
    for r in perf["trades"]:
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
    return perf

@app.post("/api/portfolio/reset")
def reset_portfolio():
    get_state()["pt"].reset()
    return {"status": "ok"}

@app.get("/api/summarize/{ticker}")
def summarize(ticker: str, hours: int = 24):
    s = get_state()
    news = s["df"].get_news_for_ticker(ticker, hours=hours)
    if news.empty:
        return {"ticker": ticker, "summary": "No news available."}
    summarizer = SimpleSummarizer()
    text = ". ".join(news["headline"].tolist()[:10])
    summary = summarizer.summarize(text, max_len=120)
    return {"ticker": ticker, "summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
