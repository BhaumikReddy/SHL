"""
scripts/build_index.py

Embeds the SHL catalog and builds a FAISS index for semantic search.

Usage:
    python scripts/build_index.py
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
ITEMS_PATH = os.path.join(INDEX_DIR, "catalog_items.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def build_text(item: dict) -> str:
    """Combine name, test_type, and description into a single string for embedding."""
    parts = [item.get("name", ""), item.get("test_type", ""), item.get("description", "")]
    return " | ".join(p for p in parts if p)


def main():
    print("=" * 60)
    print("SHL FAISS Index Builder")
    print("=" * 60)

    # Load catalog
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"\nLoaded {len(catalog)} items from catalog.json")

    # Build text strings
    texts = [build_text(item) for item in catalog]
    print(f"Sample text: {texts[0]}")

    # Load embedding model
    print(f"\nLoading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # Encode
    print("Encoding catalog items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize for cosine similarity (inner product on unit vectors = cosine)
    faiss.normalize_L2(embeddings)

    # Build FAISS IndexFlatIP (inner product)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"\nIndexed {index.ntotal} vectors (dim={dim})")

    # Save
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index to: {os.path.abspath(INDEX_PATH)}")

    with open(ITEMS_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Saved catalog items to: {os.path.abspath(ITEMS_PATH)}")

    print(f"\nDone! {index.ntotal} items indexed.")


if __name__ == "__main__":
    main()
