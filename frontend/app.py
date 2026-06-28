import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
from urllib.parse import urljoin

API_BASE = "http://localhost:8000"

def api(path, method="GET", data=None):
    url = urljoin(API_BASE, path)
    try:
        if method == "GET":
            r = requests.get(url, timeout=15)
        else:
            r = requests.post(url, json=data, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

st.set_page_config(page_title="Sentiment Trade", layout="wide")
TICKER_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"]

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", TICKER_LIST)
    window = st.selectbox("Time Window", [6, 12, 24, 48], index=2)
    if st.button("Refresh Data"):
        api("/api/signals")
        st.rerun()
    st.divider()

    port = api("/api/portfolio")
    if port:
        st.metric("Cash", f"${port.get('cash', 0):,.2f}")
        st.metric("Positions", port.get("open_positions", 0))
        if st.button("Reset Portfolio"):
            api("/api/portfolio/reset", method="POST")
            st.rerun()

price_resp = api(f"/api/price/{ticker}")
price = pd.DataFrame(price_resp.get("data", [])) if price_resp and price_resp.get("data") else pd.DataFrame()
news_resp = api(f"/api/news/{ticker}?hours={window}")
news = pd.DataFrame(news_resp.get("articles", [])) if news_resp else pd.DataFrame()
tweets_resp = api(f"/api/tweets/{ticker}?hours={window}")
tweets = pd.DataFrame(tweets_resp.get("posts", [])) if tweets_resp else pd.DataFrame()

sent_resp = api(f"/api/sentiment/{ticker}?hours={window}")
avg_sent = sent_resp.get("avg_sentiment", 0) if sent_resp else 0
sent_label = sent_resp.get("label", "Neutral") if sent_resp else "Neutral"

latest_price = price["Close"].iloc[-1] if not price.empty and "Close" in price.columns else 0

st.title("Sentiment Trade — Backend Transparency")
st.caption(f"{ticker} · ${latest_price:.2f} · Overall Sentiment: {sent_label} ({avg_sent:.3f})")

tabs = st.tabs([
    "Dashboard", "Raw News", "Sentiment Breakdown",
    "Signal Engine", "Portfolio", "Model Architectures", "Data Pipeline",
])

# ===================== TAB 0: DASHBOARD =====================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Sentiment", sent_label, f"{avg_sent:.3f}")
    col2.metric("News Articles", len(news))
    col3.metric("Social Posts", len(tweets))
    mom = 0
    if not price.empty and "returns" in price.columns and len(price) > 5:
        mom = price["returns"].tail(5).mean() * 100
    col4.metric("Price Momentum (5d)", f"{mom:+.2f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    if not price.empty and "Date" in price.columns and "Close" in price.columns:
        fig.add_trace(go.Scatter(x=price["Date"], y=price["Close"], mode="lines", name="Price",
                                 line=dict(color="#00bfff")), row=1, col=1)
    if not news.empty and "base_sentiment" in news.columns and not price.empty:
        colors = ["#00cc66" if s > 0 else "#ff4444" for s in news["base_sentiment"]]
        fig.add_trace(go.Scatter(x=news["timestamp"], y=[latest_price] * len(news),
                                 mode="markers", marker=dict(size=10, color=colors, symbol="diamond"),
                                 text=news["headline"], name="News Sentiment",
                                 hovertemplate="<b>%{text}</b><br>Sentiment: %{marker.color}<extra></extra>"),
                      row=1, col=1)
    if not price.empty and "returns" in price.columns:
        returns = price["returns"].fillna(0) * 100
        fig.add_trace(go.Bar(x=price["Date"], y=returns, name="Return %",
                             marker_color=["#00cc66" if r >= 0 else "#ff4444" for r in returns]),
                      row=2, col=1)
    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    all_scores = []
    if not news.empty and "base_sentiment" in news.columns:
        all_scores.extend(news["base_sentiment"].tolist())
    if tweets_resp and tweets_resp.get("posts"):
        for p in tweets_resp["posts"]:
            all_scores.append(p.get("textblob_sentiment", 0))

    if all_scores:
        st.subheader("Sentiment Distribution")
        fig2 = px.histogram(pd.DataFrame({"sentiment": all_scores}), x="sentiment",
                           nbins=20, template="plotly_dark",
                           color_discrete_sequence=["#00bfff"])
        fig2.add_vline(x=0, line_dash="dash", line_color="#ff4444", annotation_text="Neutral")
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"**Sentiment Signal:** {sent_label} (score: {avg_sent:.3f})")

# ===================== TAB 1: RAW NEWS =====================
with tabs[1]:
    st.subheader(f"Raw News Feed — {ticker}")
    st.caption(f"Showing {len(news)} articles from the past {window} hours")
    st.markdown("---")

    if not news.empty:
        for i, (_, row) in enumerate(news.iterrows()):
            base_s = row.get("base_sentiment", 0)
            sent_c = "#00cc66" if base_s > 0.1 else "#ff4444" if base_s < -0.1 else "#ffaa00"
            with st.expander(f"#{i+1} {str(row.get('headline', ''))[:100]}...", expanded=(i < 3)):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Headline:** {row.get('headline', '')}")
                    if "summary" in row.index and row["summary"]:
                        st.markdown(f"**Summary:** {row['summary']}")
                    if "url" in row.index and row["url"]:
                        st.markdown(f"**URL:** {row['url']}")
                with cols[1]:
                    st.markdown(f"**Source:** {row.get('source', '')}")
                    st.markdown(f"**Time:** {row.get('timestamp', '')}")
                    st.markdown(f"**Sentiment:** <span style='color:{sent_c}'>{base_s:.3f}</span>",
                               unsafe_allow_html=True)
    else:
        st.warning(f"No news articles found for {ticker} in the past {window} hours.")

    st.markdown("---")
    st.subheader(f"Social Media Posts — {ticker}")
    if tweets_resp and tweets_resp.get("posts"):
        for p in tweets_resp["posts"]:
            s = p.get("textblob_sentiment", 0)
            c = "#00cc66" if s > 0.1 else "#ff4444" if s < -0.1 else "#ffaa00"
            st.markdown(f"**@{p.get('author', '')}**: {p.get('text', '')}")
            st.markdown(f"<span style='color:{c}'>Sentiment: {s:.3f}</span> | Source: {p.get('source', '')} | {p.get('timestamp', '')}",
                       unsafe_allow_html=True)
            st.divider()
    else:
        st.info(f"No social posts for {ticker}.")

# ===================== TAB 2: SENTIMENT BREAKDOWN =====================
with tabs[2]:
    st.subheader("How Sentiment Score Is Calculated")
    st.markdown("""
    The final sentiment score is an **ensemble** of three models:

    ```
    Final Score = 0.5 × BERT + 0.3 × VADER + 0.2 × TextBlob
    ```

    | Model | Type | Weight | Range | Description |
    |---|---|---|---|---|
    | **BERT** | Transformer (Deep Learning) | 50% | -1 to 1 | Pre-trained DistilBERT, fine-tuned on SST-2 |
    | **VADER** | Rule-based Lexicon | 30% | -1 to 1 | Valence Aware Dictionary, good for social media |
    | **TextBlob** | Polarity Lexicon | 20% | -1 to 1 | Simple polarity lookup, fast fallback |
    """)

    articles = sent_resp.get("articles", []) if sent_resp else []
    if articles:
        st.subheader("Per-Article Sentiment Analysis")
        rows = []
        for a in articles:
            rows.append({
                "Headline": str(a.get("headline", ""))[:70] + "...",
                "BERT": round(a.get("bert", 0), 3) if a.get("bert") else "N/A",
                "VADER": round(a.get("vader", 0), 3),
                "TextBlob": round(a.get("textblob", 0), 3),
                "Ensemble": round(a.get("ensemble", 0), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Sentiment Score Distribution by Model")
        plot_df = pd.DataFrame(rows)
        for col in ["BERT", "VADER", "TextBlob"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        fig3 = go.Figure()
        for model in ["BERT", "VADER", "TextBlob"]:
            if model in plot_df.columns and plot_df[model].notna().any():
                fig3.add_trace(go.Box(y=plot_df[model], name=model))
        fig3.update_layout(template="plotly_dark", height=350, title="Sentiment Score Distribution by Model")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No news data available for sentiment analysis.")

    tw_scores = sent_resp.get("tweet_scores", []) if sent_resp else []
    if tw_scores:
        st.subheader("Social Media Sentiment")
        tweet_rows = []
        for t in tw_scores:
            tweet_rows.append({"Text": "(from API)", "VADER": round(t.get("vader", 0), 3), "TextBlob": round(t.get("textblob", 0), 3)})
        st.dataframe(pd.DataFrame(tweet_rows), use_container_width=True, hide_index=True)

# ===================== TAB 3: SIGNAL ENGINE =====================
with tabs[3]:
    st.subheader("Trading Signal Formula")
    st.markdown("""
    ```
    Signal Score = 0.35 × Sentiment + 0.20 × tanh(TweetVolume/20)
                  + 0.25 × tanh(PriceMomentum/5) + 0.20 × tanh(NewsCount/10)

    Score > 0.4  → Strong Buy
    Score > 0.1  → Buy
    Score < -0.4 → Strong Sell
    Score < -0.1 → Sell
    Otherwise    → Neutral
    ```
    """)

    if st.button("Generate Signals Now", type="primary"):
        result = api("/api/signals")
        if result and result.get("count", 0) > 0:
            st.success(f"{result['count']} signals generated across all tickers!")
        else:
            st.error("No signals generated — insufficient news data.")

    st.subheader("Current Signal Breakdown")
    tweet_vol = len(tweets) if not tweets.empty else 0
    news_count = len(news) if not news.empty else 0
    price_mom = price["returns"].tail(5).mean() * 100 if not price.empty and "returns" in price.columns and len(price) > 5 else 0
    sentiment = avg_sent

    def compute_signal_score(s, tv, pm, nc):
        return 0.35 * s + 0.20 * np.tanh(tv / 20) + 0.25 * np.tanh(pm / 5) + 0.20 * np.tanh(nc / 10)

    st.markdown(f"""
    | Component | Value | Weight | Contribution |
    |---|---|---|---|
    | Sentiment | {sentiment:.3f} | 35% | {0.35 * sentiment:.3f} |
    | Tweet Volume | {tweet_vol} | 20% | {0.20 * np.tanh(tweet_vol / 20):.3f} |
    | Price Momentum | {price_mom:.2f}% | 25% | {0.25 * np.tanh(price_mom / 5):.3f} |
    | News Count | {news_count} | 20% | {0.20 * np.tanh(news_count / 10):.3f} |
    | **Total Score** | | | **{compute_signal_score(sentiment, tweet_vol, price_mom, news_count):.3f}** |
    """)

    hist_resp = api("/api/signals/history")
    if hist_resp and hist_resp.get("history"):
        hist = pd.DataFrame(hist_resp["history"])
        st.subheader("Signal History")
        cols = [c for c in ["timestamp", "ticker", "signal", "strength", "sentiment_score", "tweet_volume", "news_count", "price_momentum", "action"] if c in hist.columns]
        if cols:
            st.dataframe(hist[cols], use_container_width=True, hide_index=True)
        if "ticker" in hist.columns and "strength" in hist.columns and "signal" in hist.columns:
            fig4 = px.bar(hist, x="ticker", y="strength", color="signal",
                         color_discrete_map={"Strong Buy": "#00cc66", "Buy": "#66ff99",
                                            "Neutral": "#ffaa00", "Sell": "#ff6666",
                                            "Strong Sell": "#ff4444"},
                         template="plotly_dark", title="Signal Strength by Ticker")
            st.plotly_chart(fig4, use_container_width=True)

# ===================== TAB 4: PORTFOLIO =====================
with tabs[4]:
    port = api("/api/portfolio")
    if port:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${port.get('current_value', 0):,.2f}", f"{port.get('total_return_pct', 0):+.2f}%")
        col2.metric("Cash", f"${port.get('cash', 0):,.2f}")
        col3.metric("Open Positions", port.get("open_positions", 0))
        col4.metric("Total Trades", port.get("total_trades", 0))

        eq = port.get("equity_curve", [])
        if eq:
            st.subheader("Equity Curve")
            eq_df = pd.DataFrame(eq)
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=eq_df["timestamp"], y=eq_df["equity"], fill="tozeroy",
                                      mode="lines", line=dict(color="#00bfff", width=2)))
            fig5.add_hline(y=port.get("initial_capital", 100000), line_dash="dash", line_color="#ffaa00",
                          annotation_text="Initial Capital")
            fig5.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig5, use_container_width=True)

        positions = port.get("positions_detail", [])
        if positions:
            st.subheader("Open Positions")
            st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)

        trades = port.get("trades", [])
        if trades:
            st.subheader("Trade History")
            st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)

        total = port.get("total_trades", 0)
        if total > 0:
            w = port.get("winning_trades", 0)
            l = port.get("losing_trades", 0)
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            st.metric("Win Rate", f"{wr:.1f}%")

# ===================== TAB 5: MODEL ARCHITECTURES =====================
with tabs[5]:
    st.subheader("Model Architecture Details")

    st.markdown("### 1. BERT Sentiment Classifier")
    st.markdown("""
    ```
    Input Text → [CLS] Tokenize → DistilBERT (6 layers, 768 hidden)
              → [CLS] Pooling → Linear(768, 2) → Softmax → Positive/Negative
    ```
    - **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
    - **Parameters:** 67M
    - **Vocabulary:** 30,522 tokens
    - **Max Length:** 512 tokens
    - **Output:** Confidence score mapped to [-1, 1]
    """)

    st.markdown("### 2. LSTM Text Classifier")
    st.markdown("""
    ```
    Input Text → Embedding(256d) → BiLSTM(2 layers, 128 hidden)
              → Concatenate(hidden_f, hidden_b) → Dropout(0.3) → Linear → Sigmoid
    ```
    - **Embedding:** 256-dimensional trainable
    - **LSTM:** Bidirectional, 2 layers, 128 hidden dims
    - **Regularization:** Dropout 0.3
    """)

    st.markdown("### 3. CNN on Transformer Embeddings")
    st.markdown("""
    ```
    Input Text → DistilBERT (frozen) → Embeddings(768d)
              → Conv1D(128 filters, kernel=3) → MaxPool
              → Conv1D(128 filters, kernel=4) → MaxPool
              → Conv1D(128 filters, kernel=5) → MaxPool
              → Concatenate → Dropout(0.3) → Linear → Sigmoid
    ```
    - **Filters:** 128 per kernel size
    - **Kernel Sizes:** 3, 4, 5 (captures n-gram patterns)
    - **Transformer:** Frozen DistilBERT (no fine-tuning)
    """)

    st.markdown("### 4. ANN Signal Classifier")
    st.markdown("""
    ```
    Input(8 features) → Linear(64) → ReLU → BatchNorm → Dropout(0.3)
                     → Linear(32) → ReLU → BatchNorm → Dropout(0.3)
                     → Linear(3) → Softmax → [Strong Sell, Neutral, Strong Buy]
    ```
    - **Input Features:** sentiment_score, tweet_volume, news_count, price_momentum,
      volatility, bert_confidence, vader_sentiment, textblob_sentiment
    - **Output Classes:** Strong Sell (0), Neutral (1), Strong Buy (2)
    """)

    st.markdown("### 5. GenAI Summarizer (BART)")
    st.markdown("""
    ```
    Multiple Headlines → BART Encoder → BART Decoder → Concise Summary
    ```
    - **Model:** `facebook/bart-large-cnn` (406M params)
    - **Architecture:** Seq2Seq with 12 encoder + 12 decoder layers
    - **Training:** Fine-tuned on CNN/DailyMail for news summarization
    """)

# ===================== TAB 6: DATA PIPELINE =====================
with tabs[6]:
    st.subheader("End-to-End Data Pipeline")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        DATA PIPELINE                               │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐  │
    │  │  Finnhub  │   │    GNews     │   │ AlphaVan.  │   │Stocktwits│  │
    │  │  (News)   │   │   (News)     │   │  (News)    │   │ (Social) │  │
    │  └─────┬────┘   └──────┬───────┘   └─────┬──────┘   └────┬─────┘  │
    │        └───────────────┼─────────────────┘               │        │
    │                        │                                  │        │
    │                        ▼                                  ▼        │
    │              ┌─────────────────┐              ┌───────────────┐    │
    │              │  News DataFrame │              │ Social DF     │    │
    │              │  (headline,src, │              │ (text,author, │    │
    │              │   url,ts,sent)  │              │  source,ts)   │    │
    │              └────────┬────────┘              └───────┬───────┘    │
    │                       │                               │           │
    │                       ▼                               ▼           │
    │              ┌──────────────────────────────────────────────┐     │
    │              │        SENTIMENT ANALYSIS                    │     │
    │              │  BERT(50%) + VADER(30%) + TextBlob(20%)     │     │
    │              │        → Ensemble Sentiment Score           │     │
    │              └───────────────────┬──────────────────────────┘     │
    │                                  │                                │
    │                                  ▼                                │
    │              ┌──────────────────────────────────────────────┐     │
    │              │        SIGNAL GENERATOR                      │     │
    │              │  0.35*Sentiment + 0.20*tanh(Tweets/20)      │     │
    │              │  +0.25*tanh(Momentum/5) + 0.20*tanh(News/10)│     │
    │              │        → Buy / Sell / Neutral               │     │
    │              └───────────────────┬──────────────────────────┘     │
    │                                  │                                │
    │                                  ▼                                │
    │              ┌──────────────────────────────────────────────┐     │
    │              │        ANN CLASSIFIER (optional)              │     │
    │              │  [8 features] → 64 → 32 → 3 → StrongBuy/    │     │
    │              │                Neutral/StrongSell            │     │
    │              └───────────────────┬──────────────────────────┘     │
    │                                  │                                │
    │                                  ▼                                │
    │              ┌──────────────────────────────────────────────┐     │
    │              │        PORTFOLIO TRACKER                     │     │
    │              │  Execute trades, track P&L, equity curve    │     │
    │              │  Win rate, positions, trade history         │     │
    │              └──────────────────────────────────────────────┘     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    ```""")

    st.subheader("Current Data State")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**News Data Columns**")
        if not news.empty:
            st.json({col: str(news[col].dtype) for col in news.columns})
        st.markdown("**Social Data Columns**")
        if not tweets.empty:
            st.json({col: str(tweets[col].dtype) for col in tweets.columns})
    with col2:
        st.markdown("**Price Data Stats**")
        if not price.empty and "Close" in price.columns:
            st.json({
                "rows": len(price),
                "latest_price": float(price["Close"].iloc[-1]),
                "min_price": float(price["Close"].min()),
                "max_price": float(price["Close"].max()),
                "volatility": float(price["returns"].std() * 100) if "returns" in price.columns and len(price) > 1 else 0,
            })

    if not news.empty:
        st.subheader("Latest News Summary (GenAI-style)")
        summ = api(f"/api/summarize/{ticker}?hours={window}")
        if summ:
            st.info(summ.get("summary", "No summary available."))
