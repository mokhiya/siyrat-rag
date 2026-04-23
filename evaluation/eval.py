# evaluation/eval.py
# Retrieval evaluation script — tests chunking and hybrid search quality.
# Run after chunking, embedding, and BM25 indexing are complete.
import sys
import os
import json
import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from config import CHROMA_DIR, DATASET
from sentence_transformers import SentenceTransformer
import chromadb
from hybrid_search import hybrid_search, load_bm25

# (question, expected_keyword, context_keyword_or_None)
# context_keyword forces both the expected word AND the context word to appear
# in the same chunk — prevents false positives (e.g. "Abdulloh" the father vs
# "Abdulloh ibn Mas'ud" the companion).
TEST_QUESTIONS = [
    ("Rasulullohning onasi kim edi?",    "Omina",    None),
    ("Rasululloh qaysi yili tug'ilgan?", "571",      None),
    ("Rasululloh qayerda tug'ilgan?",    "Makka",    None),
    ("Rasulullohning otasi kim edi?",    "Abdulloh", "otasi"),
]


def run_eval(note=""):
    with open(DATASET) as f:
        dataset = json.load(f)
    chunk_count = len(dataset)

    model      = SentenceTransformer("intfloat/multilingual-e5-base")
    chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("siyrat")
    bm25_data  = load_bm25()

    print(f"\n{'='*60}")
    print(f"Date:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Chunks: {chunk_count}")
    print(f"Note:   {note}")
    print(f"{'='*60}")

    results = []
    for query, expected, context in TEST_QUESTIONS:
        hybrid_results = hybrid_search(
            query=query,
            embed_model=model,
            chroma_collection=collection,
            bm25_data=bm25_data,
            top_k=20,
        )
        docs     = [r["text"] for r in hybrid_results]
        top_sim  = round(hybrid_results[0]["rrf_score"], 4) if hybrid_results else 0

        if context:
            found = any(
                expected.lower() in doc.lower() and context.lower() in doc.lower()
                for doc in docs
            )
        else:
            found = any(expected.lower() in doc.lower() for doc in docs)

        status = "✅" if found else "❌"
        print(f"\n{status} Query: {query}")
        print(f"   Expected: '{expected}' | Top RRF score: {top_sim}")
        if not found:
            print(f"   Top chunk: {docs[0][:150]}...")

        results.append({"query": query, "found": found, "top_score": top_sim})

    found_count = sum(r["found"] for r in results)
    print(f"\n{'='*60}")
    print(f"Result: {found_count}/{len(TEST_QUESTIONS)} queries answered correctly")
    print(f"{'='*60}\n")

    # Append results to the experiment log
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_log.md")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Chunks: {chunk_count} | {note}\n")
        f.write("| Query | Expected | Found | Score |\n")
        f.write("|-------|----------|-------|-------|\n")
        for r in results:
            status = "✅" if r["found"] else "❌"
            f.write(f"| {r['query']} | — | {status} | {r['top_score']} |\n")
        f.write(f"\n**Total: {found_count}/{len(TEST_QUESTIONS)}**\n")

    print(f"Log saved: experiment_log.md")


if __name__ == "__main__":
    note = sys.argv[1] if len(sys.argv) > 1 else "no note"
    run_eval(note)
