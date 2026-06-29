try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

class GenAISummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = None
        self.tokenizer = None
        self.model = None

    def load(self):
        if not _HF_AVAILABLE:
            raise ImportError("transformers not installed")
        if self.summarizer is None:
            try:
                self.summarizer = pipeline("summarization", model=self.model_name)
            except KeyError:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return self

    def summarize(self, text, max_len=60, min_len=20):
        self.load()
        if isinstance(text, str): texts = [text]
        else: texts = text
        results = []
        for t in texts:
            if self.summarizer:
                results.append(self.summarizer(t, max_length=max_len,
                                min_length=min_len, do_sample=False)[0]["summary_text"])
            else:
                inputs = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=1024)
                ids = self.model.generate(inputs["input_ids"], max_length=max_len,
                                          min_length=min_len, do_sample=False)
                results.append(self.tokenizer.decode(ids[0], skip_special_tokens=True))
        return results[0] if isinstance(text, str) else results

class SimpleSummarizer:
    def summarize(self, text, max_len=60):
        sentences = text.replace(". ", ".").split(".")
        return ". ".join(sentences[:3]) + "." if len(sentences) > 3 else text[:max_len]