from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfSearch:
    def __init__(self, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.doc_matrix = None
        self.docs: List[str] = []
        self.meta: List[Dict] = []

    def fit(self, docs: List[str], meta: List[Dict]):
        self.docs = docs
        self.meta = meta
        self.doc_matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float, str]]:
        if self.doc_matrix is None:
            raise ValueError("Search index not built. Call fit() first.")

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()

        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append((self.meta[idx], float(score), self.docs[idx]))
        return results
