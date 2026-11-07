"""Knowledge base + retrieval for RAG.

The knowledge base is the training tickets with their reference agent answers.
A new ticket is embedded and matched to the most similar past tickets; their answers
are handed to the LLM as grounding. Uses sentence-transformers embeddings
(all-MiniLM-L6-v2) when installed, otherwise TF-IDF vectors.
"""
import numpy as np
import pandas as pd


class Retriever:
    def __init__(self, kb: pd.DataFrame, backend: str = "auto"):
        self.kb = kb.reset_index(drop=True)
        self.backend = backend
        if backend in ("auto", "embeddings"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.vectors = self._embed(self.kb["text"].tolist())
                self.backend = "embeddings"
            except ImportError:
                if backend == "embeddings":
                    raise
                self.backend = "tfidf"
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.vectors = self.vec.fit_transform(self.kb["text"])

    def _embed(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)

    def search(self, query: str, k: int = 3, intent: str | None = None) -> list[dict]:
        """Top-k most similar knowledge-base entries (cosine similarity).
        If an intent is given, search only entries with that intent (metadata filter)."""
        if self.backend == "embeddings":
            scores = self.vectors @ self._embed([query])[0]
        else:
            scores = (self.vectors @ self.vec.transform([query]).T).toarray().ravel()
        if intent is not None:
            scores = np.where(self.kb["intent"].to_numpy() == intent, scores, -np.inf)
        top = np.argsort(scores)[::-1][:k]
        return [{"text": self.kb.at[i, "text"], "intent": self.kb.at[i, "intent"],
                 "response": self.kb.at[i, "response"], "score": round(float(scores[i]), 4)}
                for i in top if np.isfinite(scores[i])]
