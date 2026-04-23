# src/hybrid_search.py
# Hybrid retrieval: dense embedding search (ChromaDB) + BM25 keyword search + RRF fusion
import os
import pickle
import sys

sys.path.append(os.path.dirname(__file__))
from config import CHROMA_DIR
from build_bm25 import tokenize, BM25_PATH


def load_bm25():
    """Load the persisted BM25 index from disk."""
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def hybrid_search(query, embed_model, chroma_collection, bm25_data,
                  top_k=20, k_rrf=60):
    """
    Retrieve the most relevant chunks using two complementary methods:

    1. Dense embedding search via ChromaDB (semantic similarity)
    2. BM25 keyword search (lexical matching)
    3. Reciprocal Rank Fusion (RRF) to combine both ranked lists

    Returns a list of dicts: {id, text, source, rrf_score}, sorted by score desc.
    """
    # 1. Dense embedding retrieval
    q_vec = embed_model.encode("query: " + query).tolist()
    emb_result = chroma_collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    emb_ids = emb_result["ids"][0]

    # 2. BM25 keyword retrieval
    bm25       = bm25_data["bm25"]
    all_ids    = bm25_data["ids"]
    all_texts  = bm25_data["texts"]
    all_sources = bm25_data["sources"]

    tokens      = tokenize(query)
    scores      = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    bm25_ids    = [all_ids[i] for i in top_indices]

    # 3. Reciprocal Rank Fusion: score = Σ 1 / (k + rank)
    rrf_scores = {}
    for rank, cid in enumerate(emb_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k_rrf + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k_rrf + rank + 1)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Map chunk IDs back to text and source
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    results = []
    for cid, score in sorted_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            results.append({
                "id":        cid,
                "text":      all_texts[idx],
                "source":    all_sources[idx],
                "rrf_score": score,
            })
    return results
