import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=256, hidden=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.sigmoid(self.fc(self.dropout(x))).squeeze(1)

class LSTMSentimentModel:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def build(self):
        self.model = LSTMClassifier().to(self.device)
        return self

    def predict(self, texts):
        if self.model is None: raise ValueError("Model not built")
        self.model.eval()
        if isinstance(texts, str): texts = [texts]
        encodings = self.tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(encodings["input_ids"].to(self.device),
                                 encodings["attention_mask"].to(self.device))
        return outputs.cpu().numpy()