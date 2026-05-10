"""
app/retriever.py

Loads the FAISS index and catalog items once at import time.
Exposes a single search() function for semantic retrieval.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
_INDEX_PATH = os.path.join(_BASE, "index.faiss")
_ITEMS_PATH = os.path.join(_BASE, "catalog_items.json")
_MODEL_NAME = "all-MiniLM-L6-v2"

# Load once at module import
print("Loading FAISS index and embedding model...")
_model = SentenceTransformer(_MODEL_NAME)
_index = faiss.read_index(_INDEX_PATH)
with open(_ITEMS_PATH, encoding="utf-8") as f:
    _items: list[dict] = json.load(f)
print(f"Retriever ready — {_index.ntotal} items indexed.")


def search(query: str, top_k: int = 20) -> list[dict]:
    """
    Return the top_k most relevant catalog items for the given query.

    Args:
        query:  Natural-language search string.
        top_k:  Number of results to return (default 20, max capped at index size).

    Returns:
        List of catalog item dicts, ordered by relevance (most relevant first).
    """
    top_k = min(top_k, _index.ntotal)

    # Encode and normalize the query vector
    embedding = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)

    # Search
    scores, indices = _index.search(embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < 0:  # FAISS returns -1 for empty slots
            continue
        results.append(_items[idx])

    return results
