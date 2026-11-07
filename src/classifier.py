"""One interface for both classifiers: predict(texts) -> [(intent, confidence), ...]."""
from pathlib import Path

import joblib
import numpy as np


class BaselineClassifier:
    def __init__(self, path="models/baseline.joblib"):
        self.pipe = joblib.load(path)

    def predict(self, texts):
        proba = self.pipe.predict_proba(list(texts))
        classes = self.pipe.classes_
        return [(str(classes[i]), float(p[i])) for p, i in zip(proba, proba.argmax(1))]


class TransformerClassifier:
    def __init__(self, path="models/distilbert-intent"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(self, texts):
        out = []
        texts = list(texts)
        with self.torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tok(texts[i:i + 64], truncation=True, padding=True, max_length=64,
                               return_tensors="pt")
                proba = self.torch.softmax(self.model(**enc).logits, -1).numpy()
                for p in proba:
                    k = int(np.argmax(p))
                    out.append((self.model.config.id2label[k], float(p[k])))
        return out


def load_classifier(prefer="distilbert"):
    """Use the fine-tuned transformer if it exists, otherwise the baseline."""
    if prefer == "distilbert" and Path("models/distilbert-intent").exists():
        return TransformerClassifier()
    return BaselineClassifier()
