import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class BERT_Sentiment:
    def __init__(self, model="distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipeline = None
        self.model_name = model
    def load(self):
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name, truncation=True, max_length=512)
        return self
    def analyze(self, texts):
        self.load()
        if isinstance(texts, str): texts = [texts]
        return np.array([(-r["score"] if r["label"] == "NEGATIVE" else r["score"]) for r in self.pipeline(texts)])

class VaderSentiment:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text):
        return self.analyzer.polarity_scores(text)["compound"]

class TextBlobSentiment:
    def analyze(self, text):
        return TextBlob(text).sentiment.polarity

class EnsembleSentiment:
    def __init__(self):
        self.bert = BERT_Sentiment()
        self.vader = VaderSentiment()
        self.textblob = TextBlobSentiment()
        self.bert_loaded = False
    def analyze(self, text):
        try:
            self.bert.load(); self.bert_loaded = True
        except: pass
        if isinstance(text, str): texts = [text]
        else: texts = text
        scores = []
        for t in texts:
            bert_s = self.bert.analyze([t])[0] if self.bert_loaded else 0.0
            vader_s = self.vader.analyze(t)
            tb_s = self.textblob.analyze(t)
            scores.append(0.5 * bert_s + 0.3 * vader_s + 0.2 * tb_s)
        return np.array(scores)
