import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

class ANN(nn.Module):
    def __init__(self, input_dim, hidden=[64, 32], output_dim=3, dropout=0.3):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class ANNClassifier:
    def __init__(self, input_dim=8, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.label_map = {0: "Strong Sell", 1: "Neutral", 2: "Strong Buy"}
    def build(self):
        self.model = ANN(input_dim=self.input_dim).to(self.device)
        return self
    def train(self, df, epochs=30, batch_size=32, lr=1e-3):
        if self.model is None: self.build()
        cols = ["sentiment_score", "tweet_volume", "news_count", "price_momentum", "volatility", "bert_confidence", "vader_sentiment", "textblob_sentiment"]
        features = self.scaler.fit_transform(df[cols].values)
        labels = df["signal"].values
        dataset = list(zip(features, labels))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            for feats, lbls in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(feats.float().to(self.device)), lbls.long().to(self.device))
                loss.backward()
                optimizer.step()
    def predict(self, df):
        self.model.eval()
        cols = ["sentiment_score", "tweet_volume", "news_count", "price_momentum", "volatility", "bert_confidence", "vader_sentiment", "textblob_sentiment"]
        feats = torch.tensor(self.scaler.transform(df[cols].values)).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(feats), dim=1)
            preds = torch.argmax(probs, dim=1)
        df["signal_label"] = [self.label_map[p.item()] for p in preds]
        df["signal_strength"] = [probs[i, p].item() for i, p in enumerate(preds)]
        return df
