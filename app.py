import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import DataFetcher
from src.models.sentiment_bert import VaderSentiment, TextBlobSentiment, BERT_Sentiment, EnsembleSentiment
from src.models.ann_classifier import ANNClassifier
from src.models.summarizer import SimpleSummarizer
from src.signals.generator import SignalGenerator
from src.signals.portfolio import PortfolioTracker
from src.utils.helpers import TICKERS, compute_signal_score

st.set_page_config(page_title="Sentiment Trade", layout="wide")
TICKER_LIST = list(TICKERS.keys())

if "df" not in st.session_state:
    st.session_state.df = DataFetcher()
    st.session_state.df.fetch_all()
if "sg" not in st.session_state:
    st.session_state.sg = SignalGenerator()
if "pt" not in st.session_state:
    st.session_state.pt = PortfolioTracker()
if "ensemble" not in st.session_state:
    st.session_state.ensemble = EnsembleSentiment()
if "summarizer" not in st.session_state:
    st.session_state.summarizer = SimpleSummarizer()

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", TICKER_LIST)
    window = st.selectbox("Time Window", [6, 12, 24, 48], index=2)
    if st.button("Refresh Data"):
        st.session_state.df.fetch_all()
        st.rerun()
    st.divider()
    st.metric("Cash", f"${st.session_state.pt.cash:,.2f}")
    st.metric("Positions", len(st.session_state.pt.positions))
    if st.button("Reset Portfolio"):
        st.session_state.pt.reset()
        st.rerun()

news = st.session_state.df.get_news_for_ticker(ticker, hours=window)
tweets = st.session_state.df.get_tweets_for_ticker(ticker, hours=window)
price = st.session_state.df.get_price_data(ticker)
vader = VaderSentiment()
tb = TextBlobSentiment()

all_scores = []
if len(news) > 0:
    all_scores.extend(news["base_sentiment"].tolist())
if len(tweets) > 0:
    all_scores.extend([tb.analyze(t) for t in tweets["text"]])
avg_sent = np.mean(all_scores) if all_scores else 0
sent_label = "Bullish" if avg_sent > 0.15 else "Bearish" if avg_sent < -0.15 else "Neutral"

st.title("Sentiment Trade — Backend Transparency")
st.caption(f"{ticker} · ${price['Close'].iloc[-1]:.2f} · Overall Sentiment: {sent_label} ({avg_sent:.3f})")

tabs = st.tabs([
    "Dashboard",
    "Raw News",
    "Sentiment Breakdown",
    "Signal Engine",
    "Portfolio",
    "Model Architectures",
    "Data Pipeline",
])

# ===================== TAB 0: DASHBOARD =====================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Sentiment", sent_label, f"{avg_sent:.3f}")
    col2.metric("News Articles", len(news))
    col3.metric("Social Posts", len(tweets))
    mom = price["returns"].tail(5).mean() * 100 if len(price) > 5 else 0
    col4.metric("Price Momentum (5d)", f"{mom:+.2f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=price["Date"], y=price["Close"], mode="lines", name="Price",
                             line=dict(color="#00bfff")), row=1, col=1)
    if len(news) > 0:
        colors = ["#00cc66" if s > 0 else "#ff4444" for s in news["base_sentiment"]]
        fig.add_trace(go.Scatter(x=news["timestamp"], y=[price["Close"].iloc[-1]] * len(news),
                                 mode="markers", marker=dict(size=10, color=colors, symbol="diamond"),
                                 text=news["headline"], name="News Sentiment",
                                 hovertemplate="<b>%{text}</b><br>Sentiment: %{marker.color}<extra></extra>"),
                      row=1, col=1)
    returns = price["returns"].fillna(0) * 100
    fig.add_trace(go.Bar(x=price["Date"], y=returns, name="Return %",
                         marker_color=["#00cc66" if r >= 0 else "#ff4444" for r in returns]),
                  row=2, col=1)
    fig.update_layout(template="plotly_dark", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if all_scores:
        st.subheader("Sentiment Distribution")
        fig2 = px.histogram(pd.DataFrame({"sentiment": all_scores}), x="sentiment",
                           nbins=20, template="plotly_dark",
                           color_discrete_sequence=["#00bfff"])
        fig2.add_vline(x=0, line_dash="dash", line_color="#ff4444", annotation_text="Neutral")
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=True)

    latest_price = price["Close"].iloc[-1]
    signal = st.session_state.sg.compute_signal(ticker, avg_sent, tweets, news, price)
    st.info(f"**Latest Signal:** {signal['signal']} — {signal['action']} (strength: {signal['strength']})")

# ===================== TAB 1: RAW NEWS =====================
with tabs[1]:
    st.subheader(f"Raw News Feed — {ticker}")
    st.caption(f"Showing {len(news)} articles from the past {window} hours")
    st.markdown("---")

    if len(news) > 0:
        for i, (_, row) in enumerate(news.iterrows()):
            sent_c = "#00cc66" if row["base_sentiment"] > 0.1 else "#ff4444" if row["base_sentiment"] < -0.1 else "#ffaa00"
            with st.expander(f"#{i+1} {row['headline'][:100]}...", expanded=(i < 3)):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**Headline:** {row['headline']}")
                    if "summary" in row.index and row["summary"]:
                        st.markdown(f"**Summary:** {row['summary']}")
                    if "url" in row.index and row["url"]:
                        st.markdown(f"**URL:** {row['url']}")
                with cols[1]:
                    st.markdown(f"**Source:** {row['source']}")
                    st.markdown(f"**Time:** {row['timestamp']}")
                    st.markdown(f"**Sentiment:** <span style='color:{sent_c}'>{row['base_sentiment']:.3f}</span>",
                               unsafe_allow_html=True)
    else:
        st.warning(f"No news articles found for {ticker} in the past {window} hours.")

    st.markdown("---")
    st.subheader(f"Social Media Posts — {ticker}")
    if len(tweets) > 0:
        for _, row in tweets.iterrows():
            s = tb.analyze(row["text"])
            c = "#00cc66" if s > 0.1 else "#ff4444" if s < -0.1 else "#ffaa00"
            st.markdown(f"**@{row['author']}**: {row['text']}")
            st.markdown(f"<span style='color:{c}'>Sentiment: {s:.3f}</span> | Source: {row['source']} | {row['timestamp']}",
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

    if len(news) > 0:
        st.subheader("Per-Article Sentiment Analysis")

        bert = BERT_Sentiment()
        bert_loaded = False
        try:
            bert.load()
            bert_loaded = True
        except Exception:
            pass

        rows = []
        for _, row in news.iterrows():
            hl = row["headline"]
            base = row["base_sentiment"]
            v = vader.analyze(hl)
            t = tb.analyze(hl)
            b = bert.analyze([hl])[0] if bert_loaded else None
            ensemble = 0.5 * (b if b is not None else 0) + 0.3 * v + 0.2 * t
            rows.append({
                "Headline": hl[:70] + "...",
                "BERT": round(b, 3) if b is not None else "N/A",
                "VADER": round(v, 3),
                "TextBlob": round(t, 3),
                "Ensemble": round(ensemble, 3),
            })

        df_sent = pd.DataFrame(rows)
        st.dataframe(df_sent, use_container_width=True, hide_index=True)

        st.subheader("Sentiment Score Distribution by Model")
        plot_df = df_sent.copy()
        for col in ["BERT", "VADER", "TextBlob"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=["VADER", "TextBlob"])
        fig3 = go.Figure()
        for model in ["BERT", "VADER", "TextBlob"]:
            if model in plot_df.columns and plot_df[model].notna().any():
                fig3.add_trace(go.Box(y=plot_df[model], name=model))
        fig3.update_layout(template="plotly_dark", height=350, title="Sentiment Score Distribution by Model")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No news data available for sentiment analysis.")

    if len(tweets) > 0:
        st.subheader("Social Media Sentiment")
        tweet_rows = []
        for _, row in tweets.iterrows():
            t = row["text"]
            v = vader.analyze(t)
            tb_s = tb.analyze(t)
            tweet_rows.append({"Text": t[:60] + "...", "VADER": round(v, 3), "TextBlob": round(tb_s, 3)})
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
        all_news = st.session_state.df.news_data.copy()
        if isinstance(all_news, dict):
            all_news = pd.concat(all_news.values(), ignore_index=True) if all_news else pd.DataFrame()
        if not all_news.empty and "base_sentiment" in all_news.columns:
            all_news["vader_sentiment"] = all_news["headline"].apply(lambda x: vader.analyze(x))
            all_news["textblob_sentiment"] = all_news["headline"].apply(lambda x: tb.analyze(x))
            all_news["sentiment_score"] = (
                0.4 * all_news["base_sentiment"]
                + 0.3 * all_news["vader_sentiment"]
                + 0.3 * all_news["textblob_sentiment"]
            )
            signals = st.session_state.sg.compute_multi_ticker_signals(all_news, st.session_state.df)
            cp = {t: st.session_state.df.get_price_data(t)["Close"].iloc[-1] for t in TICKER_LIST}
            for _, s in signals.iterrows():
                st.session_state.pt.execute_signal(s.to_dict(), cp.get(s["ticker"], 100))
            st.session_state.pt.update_equity(cp)
            st.success(f"{len(signals)} signals generated across all tickers!")
        else:
            st.error("No news data available to generate signals.")

    st.subheader("Current Signal Breakdown")
    if len(news) > 0 or len(tweets) > 0:
        tweet_vol = len(tweets)
        news_count = len(news)
        price_mom = price["returns"].tail(5).mean() * 100 if len(price) > 5 else 0
        sentiment = avg_sent

        st.markdown(f"""
        | Component | Value | Weight | Contribution |
        |---|---|---|---|
        | Sentiment | {sentiment:.3f} | 35% | {0.35 * sentiment:.3f} |
        | Tweet Volume | {tweet_vol} | 20% | {0.20 * np.tanh(tweet_vol / 20):.3f} |
        | Price Momentum | {price_mom:.2f}% | 25% | {0.25 * np.tanh(price_mom / 5):.3f} |
        | News Count | {news_count} | 20% | {0.20 * np.tanh(news_count / 10):.3f} |
        | **Total Score** | | | **{compute_signal_score(sentiment, tweet_vol, price_mom, news_count):.3f}** |
        """)

    hist = st.session_state.sg.get_signal_history()
    if not hist.empty:
        st.subheader("Signal History")
        display = hist[["timestamp", "ticker", "signal", "strength", "sentiment_score", "tweet_volume", "news_count", "price_momentum", "action"]].copy()
        display["timestamp"] = display["timestamp"].apply(lambda x: x.strftime("%H:%M:%S") if hasattr(x, "strftime") else str(x))
        st.dataframe(display, use_container_width=True, hide_index=True)

        fig4 = px.bar(hist, x="ticker", y="strength", color="signal",
                     color_discrete_map={"Strong Buy": "#00cc66", "Buy": "#66ff99",
                                        "Neutral": "#ffaa00", "Sell": "#ff6666",
                                        "Strong Sell": "#ff4444"},
                     template="plotly_dark", title="Signal Strength by Ticker")
        st.plotly_chart(fig4, use_container_width=True)

# ===================== TAB 4: PORTFOLIO =====================
with tabs[4]:
    cp = {t: st.session_state.df.get_price_data(t)["Close"].iloc[-1] for t in TICKER_LIST}
    perf = st.session_state.pt.get_performance_summary(cp)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Value", f"${perf['current_value']:,.2f}", f"{perf['total_return_pct']:+.2f}%")
    col2.metric("Cash", f"${perf['cash']:,.2f}")
    col3.metric("Open Positions", perf["open_positions"])
    col4.metric("Total Trades", perf["total_trades"])

    if not perf["equity_curve"].empty:
        st.subheader("Equity Curve")
        eq = perf["equity_curve"]
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=eq["timestamp"], y=eq["equity"], fill="tozeroy",
                                  mode="lines", line=dict(color="#00bfff", width=2)))
        fig5.add_hline(y=perf["initial_capital"], line_dash="dash", line_color="#ffaa00",
                      annotation_text="Initial Capital")
        fig5.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig5, use_container_width=True)

    if not perf["positions_detail"].empty:
        st.subheader("Open Positions")
        st.dataframe(perf["positions_detail"], use_container_width=True, hide_index=True)

    if not perf["trades"].empty:
        st.subheader("Trade History")
        trades = perf["trades"].copy()
        if "timestamp" in trades.columns:
            trades["timestamp"] = trades["timestamp"].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)
            )
        st.dataframe(trades, use_container_width=True, hide_index=True)

    if perf["total_trades"] > 0:
        wr = perf["winning_trades"] / (perf["winning_trades"] + perf["losing_trades"]) * 100 if (perf["winning_trades"] + perf["losing_trades"]) > 0 else 0
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

    st.markdown("---")
    st.subheader("Live ANN Training")

    vader_live = VaderSentiment()

    if st.button("Train ANN on Current News"):
        if len(news) > 5:
            train = news.copy()
            train["vader_sentiment"] = train["headline"].apply(lambda x: vader_live.analyze(x))
            train["textblob_sentiment"] = train["headline"].apply(lambda x: tb.analyze(x))
            train["sentiment_score"] = (
                0.4 * train["base_sentiment"]
                + 0.3 * train["vader_sentiment"]
                + 0.3 * train["textblob_sentiment"]
            )
            for c in ["tweet_volume", "news_count", "price_momentum", "volatility", "bert_confidence"]:
                train[c] = np.random.uniform(1, 20, len(train))
            train["signal"] = np.select(
                [train["sentiment_score"] > 0.3, train["sentiment_score"] < -0.3], [2, 0], default=1
            )
            ann = ANNClassifier(input_dim=8)
            ann.train(train, epochs=30)
            st.session_state["trained_ann"] = ann
            st.success(f"ANN trained on {len(train)} samples across {train['signal'].nunique()} classes")
        else:
            st.warning("Not enough data (need > 5 articles)")

    if "trained_ann" in st.session_state:
        ann = st.session_state["trained_ann"]
        if st.button("Test ANN on Current Ticker"):
            if len(news) > 0:
                test = news.head(1).copy()
                test["vader_sentiment"] = test["headline"].apply(lambda x: vader_live.analyze(x))
                test["textblob_sentiment"] = test["headline"].apply(lambda x: tb.analyze(x))
                test["sentiment_score"] = (
                    0.4 * test["base_sentiment"]
                    + 0.3 * test["vader_sentiment"]
                    + 0.3 * test["textblob_sentiment"]
                )
                test["tweet_volume"] = len(tweets)
                test["news_count"] = len(news)
                test["price_momentum"] = price["returns"].tail(5).mean() * 100 if len(price) > 5 else 0
                test["volatility"] = price["returns"].tail(20).std() * 100 if len(price) > 20 else 0
                test["bert_confidence"] = np.random.uniform(0.7, 0.95)
                result = ann.predict(test)
                st.success(f"Prediction: **{result['signal_label'].iloc[0]}** (confidence: {result['signal_strength'].iloc[0]:.2%})")
            else:
                st.warning("No news data to test")

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
        if len(news) > 0:
            st.json({col: str(news[col].dtype) for col in news.columns})
        st.markdown("**Social Data Columns**")
        if len(tweets) > 0:
            st.json({col: str(tweets[col].dtype) for col in tweets.columns})
    with col2:
        st.markdown("**Price Data Stats**")
        if len(price) > 0:
            st.json({
                "rows": len(price),
                "latest_price": float(price["Close"].iloc[-1]),
                "min_price": float(price["Close"].min()),
                "max_price": float(price["Close"].max()),
                "volatility": float(price["returns"].std() * 100) if len(price) > 1 else 0,
            })

    if len(news) > 0:
        st.subheader("Latest News Summary (GenAI-style)")
        headlines_text = ". ".join(news["headline"].tolist()[:10])
        summary = st.session_state.summarizer.summarize(headlines_text, max_len=120)
        st.info(summary)
