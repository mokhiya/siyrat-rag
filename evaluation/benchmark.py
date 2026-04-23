# evaluation/benchmark.py
# Collect model answers for all benchmark questions using the full RAG pipeline.
# Results are saved to results/benchmark_results.json.
import sys
import os
import json
import time
import datetime
import re

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from config import CHROMA_DIR
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from hybrid_search import load_bm25
from query_expansion import expand_query, multi_query_search

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
QUESTIONS_PATH = os.path.join(BASE_DIR, "evaluation", "questions.json")
RESULTS_PATH   = os.path.join(BASE_DIR, "results", "benchmark_results.json")

MODELS = [
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

PROMPT_TEMPLATE = """Sen Muhammad alayhissalomning hayoti haqida ma'lumot beruvchi yordamchisan.

QOIDALAR:
1. Faqat berilgan MANBA asosida javob ber.
2. Savolga qisqa va aniq javob ber.
3. Faqat MANBAdagi ma'lumotni ishlatib javob ber — o'z bilimingdan foydalanma, taxmin ham qilma.
4. Agar MANBAda javob bo'lmasa — "Manbada ma'lumot yo'q" de.
5. Javobda qaysi kitobdan olganingni ko'rsat.

MANBA:
{context}

SAVOL: {query}

JAVOB:"""


def get_context(query, embed_model, collection, bm25_data, groq_client, model):
    """Retrieve context for a query using query expansion + hybrid search."""
    queries = expand_query(query, groq_client, model=model)
    results = multi_query_search(
        queries=queries,
        embed_model=embed_model,
        chroma_collection=collection,
        bm25_data=bm25_data,
        top_k=20,
    )
    sources = [r["source"] for r in results]
    context = "\n\n".join(
        f"[{r['source'].replace('.pdf', '').replace('_', ' ')}]:\n{r['text'][:600]}"
        for r in results
    )
    return context, sources


def run_benchmark():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        print(f"⚠️  {RESULTS_PATH} already exists!")
        answer = input("Re-run and overwrite? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Cancelled.")
            return

    print("Loading benchmark questions...")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"   {len(questions)} questions")

    print("Loading models...")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chroma      = chromadb.PersistentClient(path=CHROMA_DIR)
    collection  = chroma.get_collection("siyrat")
    bm25_data   = load_bm25()

    results = []

    for q in questions:
        print(f"\n[{q['id']:02d}/{len(questions)}] {q['question']}")
        row = {
            "id":         q["id"],
            "question":   q["question"],
            "expected":   q["expected_answer"],
            "keywords":   q["keywords"],
            "difficulty": q["difficulty"],
            "answers":    {},
        }

        # Retrieve context once and share it across all models
        try:
            context, sources = get_context(
                q["question"], embed_model, collection,
                bm25_data, groq_client, MODELS[0]
            )
        except Exception as e:
            print(f"   ⚠️ Context retrieval error: {e}")
            context, sources = "", []

        row["sources_used"] = list(set(sources))

        for model in MODELS:
            print(f"   🤖 {model.split('/')[0].split('-')[0].upper()}...")
            try:
                prompt = PROMPT_TEMPLATE.format(context=context, query=q["question"])
                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                )
                answer = response.choices[0].message.content
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
                row["answers"][model] = answer
                print(f"      ✅ {answer[:80]}...")
            except Exception as e:
                row["answers"][model] = f"ERROR: {e}"
                print(f"      ❌ {e}")

            time.sleep(2)  # respect API rate limits

        results.append(row)

    output = {
        "date":    datetime.datetime.now().isoformat(),
        "models":  MODELS,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Benchmark results saved: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_benchmark()
