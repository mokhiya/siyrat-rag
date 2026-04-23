# src/chunking.py
# Word-based text chunking with sliding overlap window.
# Run this script to (re)build data/processed/dataset.json from raw PDFs.
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import RAW_DIR, OUT_DIR, DATASET
from loader import load_all_pdfs
from cleaner import clean_text

CHUNK_SIZE = 150   # words per chunk
OVERLAP    = 32    # words of overlap between consecutive chunks


def chunk_by_words(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into word-based chunks with a sliding overlap window."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    dataset = []
    chunk_id = 1

    for text, source in load_all_pdfs(RAW_DIR):
        clean = clean_text(text)
        chunks = chunk_by_words(clean)
        for chunk in chunks:
            dataset.append({
                "id":     chunk_id,
                "matn":   chunk,
                "source": source
            })
            chunk_id += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Total chunks: {len(dataset)}")
    print(f"\nSample chunk:\n{dataset[0]['matn']}")
    print(f"Source: {dataset[0]['source']}")
