# src/config.py
import sys
import os

# Add src/ to the Python path so sibling modules resolve correctly
sys.path.append(os.path.dirname(__file__))

# Root of the repository (one level above src/)
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))

# Data directories (gitignored — raw PDFs, processed chunks, vector DB)
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")

# Chunked dataset produced by chunking.py
DATASET    = os.path.join(OUT_DIR, "dataset.json")
