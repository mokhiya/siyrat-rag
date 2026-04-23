# src/build_bm25.py
# Build a BM25 index from the chunked dataset and persist it as a pickle file.
# Run this script after chunking.py (and optionally after embedder.py).
import json
import pickle
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))
from config import DATASET, CHROMA_DIR
from rank_bm25 import BM25Okapi

BM25_PATH = os.path.join(CHROMA_DIR, "bm25.pkl")


def tokenize(text):
    """Lowercase and extract alphanumeric tokens (supports Latin, extended Latin, apostrophes)."""
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9\u00c0-\u024f\u02bc'']+", text)
    return tokens


def build():
    print("Loading dataset...")
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"   {len(dataset)} chunks found")

    print("Building BM25 index...")
    corpus_tokens = [tokenize(item["matn"]) for item in dataset]
    bm25 = BM25Okapi(corpus_tokens)

    ids     = [str(item["id"])     for item in dataset]
    texts   = [item["matn"]        for item in dataset]
    sources = [item["source"]      for item in dataset]

    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts, "sources": sources}, f)

    print(f"✅ BM25 index saved: {BM25_PATH}")


if __name__ == "__main__":
    build()
