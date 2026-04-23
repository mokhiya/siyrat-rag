# src/embedder.py
# Generate embeddings for all chunks and store them in ChromaDB.
# Run this script after chunking.py to build the vector index.
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import DATASET, CHROMA_DIR
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the dataset
print("Loading dataset...")
with open(DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(f"   {len(dataset)} chunks found")

# 2. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# 3. Initialize ChromaDB — drop existing collection and recreate
client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("siyrat")
    print("Deleted existing collection")
except Exception:
    pass
collection = client.create_collection(
    name="siyrat",
    metadata={"hnsw:space": "cosine"}
)

# 4. Generate embeddings and store in ChromaDB
print("Embedding chunks...")
BATCH = 64

for i in range(0, len(dataset), BATCH):
    batch    = dataset[i:i + BATCH]
    texts    = [item["matn"] for item in batch]
    ids      = [str(item["id"]) for item in batch]
    # multilingual-e5-base requires "passage:" prefix for documents
    passages = ["passage: " + t for t in texts]
    vectors  = model.encode(passages, show_progress_bar=False).tolist()
    metas    = [{"source": item["source"]} for item in batch]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metas
    )
    print(f"   ✅ {min(i + BATCH, len(dataset))}/{len(dataset)}")

print(f"\nDone! {collection.count()} chunks stored in ChromaDB")
