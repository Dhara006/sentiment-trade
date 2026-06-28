import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TransformerEmbeddingCNN(nn.Module):
    def __init__(self, embed_dim=768, num_filters=128, filter_sizes=[3,4,5], dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs, padding=fs//2) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outs = [torch.max(torch.relu(conv(x)), dim=2)[0] for conv in self.convs]
        return self.sigmoid(self.fc(self.dropout(torch.cat(conv_outs, dim=1)))).squeeze(1)

class CNNTextClassifier:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.transformer = AutoModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.transformer.eval()
        for p in self.transformer.parameters(): p.requires_grad = False
        self.cnn = TransformerEmbeddingCNN().to(self.device)

    def predict(self, texts):
        self.cnn.eval()
        if isinstance(texts, str): texts = [texts]
        encodings = self.tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=128, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.transformer(encodings["input_ids"].to(self.device),
                                          encodings["attention_mask"].to(self.device)).last_hidden_state
            outputs = self.cnn(embeddings)
        return outputs.cpu().numpy()