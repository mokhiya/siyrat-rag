# Siyrat RAG — Retrieval-Augmented Generation for Islamic Biography

A production-ready RAG (Retrieval-Augmented Generation) system for answering questions about the life of Prophet Muhammad ﷺ, built on Uzbek Islamic source texts. The system uses hybrid search, query expansion, and an LLM-as-judge evaluation pipeline.

---

## Demo

![Pipeline V2](assets/pipeline_v2.png)

---

## Features

- **Hybrid search** — BM25 keyword matching + dense embedding search, fused with Reciprocal Rank Fusion (RRF)
- **Query expansion** — LLM generates 3 additional search queries from the original question to improve recall
- **Source attribution** — every answer cites which book it was drawn from
- **LLM-as-judge benchmark** — 18 questions × 3 models, scored on faithfulness, accuracy, and completeness
- **Interactive Streamlit dashboard** — Chat, Pipeline diagrams, Embedding space (t-SNE), Retrieval explorer, Benchmark results
- **Docker support** — single `docker-compose up` to run

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | `intfloat/multilingual-e5-base` |
| Vector database | ChromaDB (cosine similarity) |
| Keyword search | BM25 (`rank-bm25`) |
| LLM inference | Groq API (Llama, Qwen) |
| Frontend | Streamlit |
| Containerization | Docker |

---

## Project Structure

```
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example            # API key template
│
├── src/                    # RAG pipeline source code
│   ├── config.py           # Directory paths
│   ├── loader.py           # PDF text extraction
│   ├── cleaner.py          # OCR text cleaning
│   ├── chunking.py         # Word-based chunking (150 words, 32 overlap)
│   ├── embedder.py         # ChromaDB embedding (multilingual-e5-base)
│   ├── build_bm25.py       # BM25 index builder
│   ├── hybrid_search.py    # BM25 + embedding + RRF fusion
│   ├── query_expansion.py  # LLM query expansion + multi-query search
│   └── visualize.py        # Pipeline diagrams and t-SNE plots
│
├── evaluation/             # Benchmarking and evaluation
│   ├── benchmark.py        # Collect model answers for 18 questions
│   ├── judge.py            # LLM-as-judge scoring
│   ├── eval.py             # Retrieval quality evaluation
│   └── questions.json      # 18 benchmark questions (easy/medium/hard)
│
├── results/                # Benchmark output (gitignored)
│   ├── benchmark_results.json
│   └── benchmark_scores.json
│
├── assets/                 # Visualization images (tracked in git)
│   ├── pipeline_v1.png
│   ├── pipeline_v2.png
│   ├── tsne_v1.png
│   ├── tsne_v2.png
│   └── retrieval.png
│
└── data/                   # Gitignored — raw PDFs + generated indexes
    ├── raw/                # Source PDF books
    ├── processed/          # Chunked dataset (dataset.json)
    └── chroma/             # ChromaDB vector store + BM25 index
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/siyrat-rag.git
cd siyrat-rag
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Open .env and add your Groq API key
```

### 3. Add source PDFs

Place your Uzbek Islamic book PDFs in `data/raw/`.

### 4. Build the pipeline

```bash
# Step 1: Chunk the PDFs
python src/chunking.py

# Step 2: Embed chunks into ChromaDB
python src/embedder.py

# Step 3: Build the BM25 keyword index
python src/build_bm25.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# App will be available at http://localhost:8501
```

> **Note:** Place your source PDFs in `data/raw/` and run the pipeline steps before building the Docker image, or mount the `data/` volume after running the pipeline locally.

---

## RAG Pipeline

### V1 — Embedding Only
```
PDF → Loader → Cleaner → Chunker → Embedder → ChromaDB
Query → Embed → Top-5 Chunks → Groq LLM → Answer
```

### V2 — Hybrid Search + Query Expansion (current)
```
PDF → Loader → OCR Cleaner → Chunker → Embedder → ChromaDB
                                      ↘ BM25 Index

Query → Query Expansion (×3) → Hybrid Search (RRF) → Top-20 Chunks → Groq LLM → Answer + Source
```

**Key design decisions:**
- `passage:` / `query:` prefixes used with multilingual-e5-base as required by the model
- RRF constant `k=60` balances dense and sparse retrieval rankings
- Original query weighted `1.0`, expanded queries weighted `0.7` in multi-query merge
- Chunks truncated at 600 characters when passed to LLM context to stay within token limits

---

## Benchmark Results

Evaluated on 18 questions across three difficulty levels using LLM-as-judge scoring (1–5 scale).

| Model | Overall | Easy | Medium | Hard |
|---|---|---|---|---|
| Llama 3.1 8B Instant | **3.80** | 4.33 | 3.72 | 3.33 |
| Llama 4 Scout 17B | 3.74 | **4.39** | 3.67 | 3.17 |
| Qwen3 32B | 2.28 | 1.00 | **3.45** | 2.39 |

> Qwen3-32B scored low on easy questions due to excessive hallucination — it consistently added information from its training data rather than restricting itself to the retrieved context.

![Benchmark](assets/pipeline_v2.png)

---

## Evaluation

Run the retrieval quality check against 4 known-answer questions:

```bash
python evaluation/eval.py "your experiment note"
```

Results are appended to `experiment_log.md`.

---

## Source Books

The system was built using Uzbek-language Islamic biography books (Siyrat literature). Books are not included in this repository due to copyright.

---

## License

MIT
