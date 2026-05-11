"""
scripts/build_index.py

Builds a TF-IDF index for the SHL catalog using scikit-learn.

Usage:
    python scripts/build_index.py
"""

import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


def build_text(item: dict) -> str:
    """Combine all rich fields into a single string for TF-IDF vectorization."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        item.get("duration", ""),
    ]
    return " ".join(filter(None, parts)).strip()


def main():
    print("=" * 60)
    print("SHL TF-IDF Index Builder")
    print("=" * 60)

    # Load catalog
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"\nLoaded {len(catalog)} items from catalog.json")

    # Build text strings
    texts = [build_text(item) for item in catalog]
    print(f"Sample text: {texts[0][:100]}...")

    # Build TF-IDF vectorizer
    print("\nBuilding TF-IDF vectorizer with bigrams (max 10000 features)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    vectorizer.fit(texts)

    # Save index components
    os.makedirs(INDEX_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    joblib.dump(catalog, os.path.join(INDEX_DIR, "catalog_items.joblib"))
    joblib.dump(texts, os.path.join(INDEX_DIR, "texts.joblib"))

    print(f"\nSaved vectorizer to: {os.path.join(INDEX_DIR, 'vectorizer.joblib')}")
    print(f"Saved catalog items to: {os.path.join(INDEX_DIR, 'catalog_items.joblib')}")
    print(f"Saved texts to: {os.path.join(INDEX_DIR, 'texts.joblib')}")
    print(f"\nDone! Index built for {len(catalog)} items.")


if __name__ == "__main__":
    main()

