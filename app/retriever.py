"""
app/retriever.py

Loads the TF-IDF index and catalog items using scikit-learn.
Exposes a single search() function for semantic retrieval.
"""

import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

_vectorizer = None
_catalog = None
_texts = None


def _load():
    """Load TF-IDF components lazily."""
    global _vectorizer, _catalog, _texts
    if _vectorizer is None:
        print("Loading TF-IDF vectorizer and catalog...")
        _vectorizer = joblib.load(os.path.join(_BASE, "vectorizer.joblib"))
        _catalog = joblib.load(os.path.join(_BASE, "catalog_items.joblib"))
        _texts = joblib.load(os.path.join(_BASE, "texts.joblib"))
        print(f"Retriever ready — {len(_catalog)} items indexed.")


_load()


def search(query: str, top_k: int = 20) -> list[dict]:
    """
    Return the top_k most relevant catalog items for the given query using TF-IDF.

    Args:
        query:  Natural-language search string.
        top_k:  Number of results to return (default 20, max capped at catalog size).

    Returns:
        List of catalog item dicts, ordered by relevance (most relevant first).
    """
    _load()
    top_k = min(top_k, len(_catalog))

    # Transform query to TF-IDF vector
    query_vec = _vectorizer.transform([query])

    # Transform corpus to TF-IDF vectors and compute similarity
    corpus_vec = _vectorizer.transform(_texts)
    scores = cosine_similarity(query_vec, corpus_vec)[0]

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [_catalog[i] for i in top_indices]
