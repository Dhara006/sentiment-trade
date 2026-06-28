import re, nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|\@\w+|\#\w+|\$\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        return " ".join([t for t in tokens if t not in self.stop_words and len(t) > 2])

    def preprocess(self, text):
        return self.remove_stopwords(self.clean_text(text))