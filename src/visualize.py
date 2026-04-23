# src/visualize.py
# Generate pipeline diagrams and t-SNE embedding visualizations.
# Run this script to regenerate all images in assets/.
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(__file__))
from config import CHROMA_DIR
import chromadb
from sentence_transformers import SentenceTransformer

# All generated images are saved to assets/ at the repository root
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PIPELINE DIAGRAM — V1 (embedding only)
# ─────────────────────────────────────────────────────────────────────────────
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    steps = [
        ("PDF\nFiles",        "#4e79a7", 1.0),
        ("Loader\n(pypdf)",   "#f28e2b", 2.8),
        ("Cleaner\n(regex)",  "#e15759", 4.6),
        ("Chunker\n(tokens)", "#76b7b2", 6.4),
        ("Embedder\n(e5)",    "#59a14f", 8.2),
        ("ChromaDB\n(vector)","#edc948", 10.0),
        ("Groq LLM\n(answer)","#b07aa1", 11.8),
    ]
    for label, color, x in steps:
        box = FancyBboxPatch((x - 0.75, 1.8), 1.5, 1.4,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white",
                             linewidth=2, zorder=3)
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)

    for i in range(len(steps) - 1):
        x1 = steps[i][2] + 0.75
        x2 = steps[i + 1][2] - 0.75
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=2), zorder=2)

    # Query path (bottom row)
    query_steps = [("Query", 1.0), ("Embed", 8.2), ("Top-5\nChunks", 10.0),
                   ("Groq\nLLM", 11.8), ("Answer", 13.6)]
    colors_q = ["#9c755f", "#59a14f", "#edc948", "#b07aa1", "#76b7b2"]
    for (label, x), color in zip(query_steps, colors_q):
        box = FancyBboxPatch((x - 0.65, 0.2), 1.3, 0.9,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="white",
                             linewidth=1.5, zorder=3, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 0.65, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    for i in range(len(query_steps) - 1):
        x1 = query_steps[i][1] + 0.65
        x2 = query_steps[i + 1][1] - 0.65
        ax.annotate("", xy=(x2, 0.65), xytext=(x1, 0.65),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5), zorder=2)

    ax.text(0.3, 2.5, "INDEXING", ha="center", va="center",
            fontsize=7, color="#777777", rotation=90)
    ax.text(0.3, 0.65, "QUERY", ha="center", va="center",
            fontsize=7, color="#777777", rotation=90)
    ax.set_title("RAG Pipeline V1 — Embedding Search", fontsize=14,
                 fontweight="bold", pad=15, color="#333333")

    path = os.path.join(ASSETS_DIR, "pipeline_v1.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. t-SNE EMBEDDING SPACE — V1
# ─────────────────────────────────────────────────────────────────────────────
def draw_tsne():
    print("Loading embeddings from ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("siyrat")
    total      = collection.count()
    print(f"   {total} chunks found")

    result     = collection.get(include=["embeddings", "documents"])
    embeddings = np.array(result["embeddings"])
    documents  = result["documents"]

    print("Computing t-SNE (this may take a moment)...")
    perplexity = min(30, len(embeddings) - 1)
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42, max_iter=1000, verbose=0)
    coords = tsne.fit_transform(embeddings)

    doc_lengths = [len(d.split()) for d in documents]
    colors = np.array(doc_lengths)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=colors, cmap="plasma",
                         s=40, alpha=0.75, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Chunk length (words)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(f"Embedding Space V1 — t-SNE\n({total} chunks, multilingual-e5-base)",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.set_xlabel("t-SNE 1", color="white", fontsize=10)
    ax.set_ylabel("t-SNE 2", color="white", fontsize=10)

    path = os.path.join(ASSETS_DIR, "tsne_v1.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")
    return coords, embeddings, documents


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRIEVAL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def draw_retrieval(test_questions=None):
    if test_questions is None:
        test_questions = [
            "Rasululloh qaysi yili tug'ilgan?",
            "Muhammad alayhissalom qayerda tug'ilgan?",
            "Payg'ambarning onasi kim edi?",
        ]

    print("Generating retrieval visualization...")
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("siyrat")

    n_q  = len(test_questions)
    fig, axes = plt.subplots(n_q, 1, figsize=(12, 4 * n_q))
    if n_q == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f5f5f5")
    palette = sns.color_palette("muted", 5)

    for ax, question in zip(axes, test_questions):
        vector = embedding_model.encode(question).tolist()
        result = collection.query(
            query_embeddings=[vector],
            n_results=5,
            include=["documents", "distances"]
        )
        docs         = result["documents"][0]
        distances    = result["distances"][0]
        similarities = [1 - d for d in distances]

        labels = [f"Chunk {i+1}:\n{d[:55]}..." if len(d) > 55 else f"Chunk {i+1}:\n{d}"
                  for i, d in enumerate(docs)]

        bars = ax.barh(range(len(similarities)), similarities,
                       color=palette, edgecolor="white", linewidth=0.8)
        for bar, sim in zip(bars, similarities):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{sim:.3f}", va="center", fontsize=9, color="#333333")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Cosine Similarity", fontsize=9)
        ax.set_title(f'Query: "{question}"', fontsize=11,
                     fontweight="bold", color="#222222", pad=8)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    fig.suptitle("Retrieval Results — Top-5 Matching Chunks",
                 fontsize=14, fontweight="bold", y=1.01, color="#111111")

    path = os.path.join(ASSETS_DIR, "retrieval.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE DIAGRAM — V2 (Hybrid Search + Query Expansion)
# ─────────────────────────────────────────────────────────────────────────────
def draw_pipeline_v2():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#f0f4f8")

    index_steps = [
        ("PDF\nFiles",          "#4e79a7", 1.0),
        ("OCR\nCleaner",        "#e15759", 2.8),
        ("Chunker\n(150 words)","#76b7b2", 4.6),
        ("Embedder\n(passage:)","#59a14f", 6.4),
        ("ChromaDB\n(cosine)",  "#edc948", 8.2),
        ("BM25\nIndex",         "#f28e2b", 10.0),
    ]
    for label, color, x in index_steps:
        box = FancyBboxPatch((x - 0.75, 3.8), 1.5, 1.4,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="white", linewidth=2, zorder=3)
        ax.add_patch(box)
        ax.text(x, 4.5, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    for i in range(len(index_steps) - 1):
        x1 = index_steps[i][2] + 0.75
        x2 = index_steps[i + 1][2] - 0.75
        ax.annotate("", xy=(x2, 4.5), xytext=(x1, 4.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2), zorder=2)

    query_steps = [
        ("Query",               "#9c755f", 1.0),
        ("Query\nExpansion",    "#b07aa1", 3.0),
        ("Hybrid\nSearch(RRF)", "#4e79a7", 5.5),
        ("Top-20\nChunks",      "#edc948", 8.0),
        ("Groq\nLLM",           "#59a14f", 10.5),
        ("Answer +\nSource",    "#e15759", 12.5),
    ]
    for label, color, x in query_steps:
        box = FancyBboxPatch((x - 0.85, 1.0), 1.7, 1.4,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="white", linewidth=1.8,
                             zorder=3, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 1.7, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    for i in range(len(query_steps) - 1):
        x1 = query_steps[i][2] + 0.85
        x2 = query_steps[i + 1][2] - 0.85
        ax.annotate("", xy=(x2, 1.7), xytext=(x1, 1.7),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.8), zorder=2)

    # Arrows from ChromaDB and BM25 down into Hybrid Search
    ax.annotate("", xy=(5.5, 2.4), xytext=(8.2, 3.8),
                arrowprops=dict(arrowstyle="->", color="#4e79a7", lw=1.5, linestyle="dashed"))
    ax.annotate("", xy=(5.5, 2.4), xytext=(10.0, 3.8),
                arrowprops=dict(arrowstyle="->", color="#f28e2b", lw=1.5, linestyle="dashed"))

    ax.text(0.3, 4.5, "INDEXING", ha="center", va="center",
            fontsize=7, color="#555", rotation=90)
    ax.text(0.3, 1.7, "QUERY", ha="center", va="center",
            fontsize=7, color="#555", rotation=90)
    ax.set_title("RAG Pipeline V2 — Hybrid Search + Query Expansion",
                 fontsize=13, fontweight="bold", pad=15, color="#222")

    path = os.path.join(ASSETS_DIR, "pipeline_v2.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. t-SNE EMBEDDING SPACE — V2 (OCR-cleaned, passage: prefix)
# ─────────────────────────────────────────────────────────────────────────────
def draw_tsne_v2():
    print("Loading embeddings from ChromaDB (V2)...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("siyrat")
    total      = collection.count()

    result     = collection.get(include=["embeddings", "documents"])
    embeddings = np.array(result["embeddings"])
    documents  = result["documents"]

    print("Computing t-SNE...")
    perplexity = min(30, len(embeddings) - 1)
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42, max_iter=1000, verbose=0)
    coords = tsne.fit_transform(embeddings)

    doc_lengths = [len(d.split()) for d in documents]
    colors = np.array(doc_lengths)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=colors, cmap="viridis",
                         s=35, alpha=0.8, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Chunk length (words)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(f"Embedding Space V2 — t-SNE\n"
                 f"(OCR-cleaned, {total} chunks, multilingual-e5-base + passage: prefix)",
                 fontsize=12, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_xlabel("t-SNE 1", color="white", fontsize=10)
    ax.set_ylabel("t-SNE 2", color="white", fontsize=10)

    path = os.path.join(ASSETS_DIR, "tsne_v2.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  RAG Visualization Generator")
    print("=" * 50 + "\n")

    print("1/5 — Pipeline diagram (V1)...")
    draw_pipeline()

    print("\n2/5 — Embedding space t-SNE (V1)...")
    draw_tsne()

    print("\n3/5 — Retrieval results...")
    draw_retrieval()

    print("\n4/5 — Pipeline diagram (V2)...")
    draw_pipeline_v2()

    print("\n5/5 — Embedding space t-SNE (V2)...")
    draw_tsne_v2()

    print(f"\nAll images saved to: assets/")
