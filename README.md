---
title: Siyrat RAG
emoji: 📖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Siyrat RAG

A Retrieval-Augmented Generation (RAG) system for answering questions about the life of Prophet Muhammad ﷺ, built on Uzbek-language Islamic source texts.

The system retrieves relevant passages from digitized books using hybrid search (BM25 + dense embeddings), expands queries using an LLM, and generates grounded answers with source attribution. A built-in benchmark evaluates three LLMs using an LLM-as-judge pipeline.

---

## Pipeline

### V1 — Dense Embedding Search
```
PDF → Loader → Cleaner → Chunker → Embedder → ChromaDB
                                                    ↓
Query ──────────────────────────── Embed → Top-5 Chunks → Groq LLM → Answer
```

### V2 — Hybrid Search + Query Expansion (current)
```
PDF → Loader → OCR Cleaner → Chunker (150w) → Embedder (passage:) → ChromaDB ──┐
                                                                                  ↓
                                                                      Hybrid Search (RRF)
                                                                                  ↑
                                                              BM25 Index ─────────┘

Query → Query Expansion (LLM, ×3) → Hybrid Search → Top-20 Chunks → Groq LLM → Answer + Source
```

![V2 Pipeline](assets/pipeline_v2.png)

---

## Embedding Space

Chunks visualized with t-SNE — 2041 OCR-cleaned chunks, `intfloat/multilingual-e5-base`.

| V1 | V2 |
|---|---|
| ![t-SNE V1](assets/tsne_v1.png) | ![t-SNE V2](assets/tsne_v2.png) |

---

## Benchmark Results

18 questions × 3 models, scored by an LLM judge on three criteria (1–5 each):
- **Faithfulness** — does the answer stay within the retrieved context?
- **Accuracy** — are the facts correct?
- **Completeness** — does the answer fully address the question?

| Model | Overall | Easy | Medium | Hard |
|---|---|---|---|---|
| **Llama 3.1 8B Instant** | **3.80** | 4.33 | 3.72 | 3.33 |
| Llama 4 Scout 17B | 3.74 | **4.39** | 3.67 | 3.17 |
| Qwen3 32B | 2.28 | 1.00 | **3.45** | 2.39 |

> Qwen3-32B scored low on easy questions because it consistently added information from its training data rather than restricting itself to the retrieved context. Llama models were more faithful to the source material.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | `intfloat/multilingual-e5-base` (768-dim, multilingual) |
| Vector store | ChromaDB (cosine similarity, HNSW index) |
| Keyword search | BM25 via `rank-bm25` |
| Result fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Query expansion | Groq LLM (3 additional queries per question) |
| LLM inference | Groq API — Llama 3.3 70B / Qwen3 32B |
| Frontend | Streamlit (5 tabs) |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
├── app.py                    # Streamlit dashboard (entry point)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── src/
│   ├── config.py             # Directory path constants
│   ├── loader.py             # PDF text extraction (pypdf)
│   ├── cleaner.py            # OCR artifact removal (regex)
│   ├── chunking.py           # Word-based chunking — 150 words, 32 overlap
│   ├── embedder.py           # ChromaDB embedding with passage: prefix
│   ├── build_bm25.py         # BM25 index builder
│   ├── hybrid_search.py      # BM25 + embedding + RRF fusion
│   ├── query_expansion.py    # LLM query expansion + multi-query merge
│   └── visualize.py          # Pipeline diagrams and t-SNE plots
│
├── evaluation/
│   ├── benchmark.py          # Collect model answers (18 questions × 3 models)
│   ├── judge.py              # LLM-as-judge scoring
│   ├── eval.py               # Retrieval quality test (4 known-answer queries)
│   └── questions.json        # Benchmark questions with difficulty labels
│
├── results/                  # Generated output — gitignored
│   ├── benchmark_results.json
│   └── benchmark_scores.json
│
├── assets/                   # Visualization images — tracked in git
│   ├── pipeline_v1.png / pipeline_v2.png
│   ├── tsne_v1.png / tsne_v2.png
│   └── retrieval.png
│
└── data/                     # Gitignored — raw PDFs + generated indexes
    ├── raw/                  # Source PDF books (not included)
    ├── processed/            # dataset.json (chunked text)
    └── chroma/               # ChromaDB vector store + bm25.pkl
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier available)
- Source PDF books placed in `data/raw/`

### Install

```bash
git clone https://github.com/your-username/siyrat-rag.git
cd siyrat-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### Build the pipeline

Run these steps once to prepare the data before launching the app:

```bash
# 1. Extract text from PDFs, clean OCR artifacts, and chunk into 150-word segments
python src/chunking.py

# 2. Generate embeddings and store in ChromaDB
python src/embedder.py

# 3. Build the BM25 keyword index
python src/build_bm25.py
```

### Run

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| 💬 Chat | Ask questions — hybrid search + query expansion + LLM answer with source |
| 📊 Pipeline | V1 vs V2 pipeline diagrams side by side |
| 🔵 Embedding Space | t-SNE visualization of the chunk embedding space |
| 🔍 Retrieval | Interactive bar chart — enter any query to see top-5 RRF scores |
| 🏆 Benchmark | LLM-as-judge results with per-model and per-difficulty breakdowns |

---

## Deployment (Railway)

The vector index (`data/chroma/`) and processed dataset (`data/processed/`) are tracked in git — Railway reads the repo directly and builds the Docker image on their servers. **No local Docker installation needed.**

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/siyrat-rag.git
git push -u origin main
```

### Step 2 — Deploy on Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**
2. Select your repository — Railway auto-detects the `Dockerfile`
3. Go to **Variables** tab and add:
   ```
   GROQ_API_KEY = your_key_here
   ```
4. Click **Deploy** — Railway builds the image and starts the app
5. Go to **Settings → Networking → Generate Domain** to get your public URL

### Updating the deployment

```bash
git add .
git commit -m "your change"
git push
# Railway auto-redeploys on every push to main
```

---

## Running locally (without Docker)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env       # add your GROQ_API_KEY

streamlit run app.py
# → http://localhost:8501
```

If you need to rebuild the index from new PDFs:

```bash
python src/chunking.py     # chunk PDFs → data/processed/dataset.json
python src/embedder.py     # embed → data/chroma/
python src/build_bm25.py   # keyword index → data/chroma/bm25.pkl
```

---

## Evaluation

### Retrieval quality check

Tests whether 4 known-answer questions retrieve the correct chunk in the top-20 results:

```bash
python evaluation/eval.py "experiment note"
# Results appended to experiment_log.md
```

### Run the full benchmark

```bash
# Step 1: Collect answers from all 3 models (saves to results/benchmark_results.json)
python evaluation/benchmark.py

# Step 2: Score answers with LLM-as-judge (saves to results/benchmark_scores.json)
python evaluation/judge.py
```

### Regenerate visualizations

```bash
python src/visualize.py
# Saves 5 images to assets/
```

---

## Design Decisions

**Why `passage:` / `query:` prefixes?**
`intfloat/multilingual-e5-base` uses asymmetric encoding — documents are encoded with `passage:` and queries with `query:`. Without the prefix, the model is used out of spec and retrieval quality degrades.

**Why 150-word chunks with 32-word overlap?**
Tested chunk sizes from 50 to 300 words. 150 words balances context richness (enough for an LLM to generate a meaningful answer) with retrieval precision (small enough to rank the right chunk highly).

**Why BM25 + dense search instead of dense only?**
Dense embeddings miss exact keyword matches for proper nouns (names, dates, places) common in biographical text. BM25 handles these well. RRF fusion captures the best of both without requiring score normalization.

**Why query expansion?**
Uzbek Islamic terminology has many synonyms and transliteration variants (e.g., "Rasululloh", "Payg'ambar", "Muhammad alayhissalom"). A single query may miss relevant chunks that use a different variant. Expansion generates 3 alternative phrasings, improving recall without hurting precision.

---

## License

MIT
