# app.py
# Streamlit dashboard — entry point for the Siyrat RAG application.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import json
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from config import CHROMA_DIR
from hybrid_search import hybrid_search, load_bm25
from query_expansion import expand_query, multi_query_search

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_DIR    = os.path.dirname(__file__)
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
SCORES_PATH = os.path.join(BASE_DIR, "results", "benchmark_scores.json")

st.set_page_config(
    page_title="Siyrat RAG",
    page_icon="📖",
    layout="wide",
)


@st.cache_resource
def load_models():
    embed      = SentenceTransformer("intfloat/multilingual-e5-base")
    client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("siyrat")
    bm25_data  = load_bm25()
    return embed, client, collection, bm25_data


embed_model, groq_client, collection, bm25_data = load_models()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat",
    "📊 Pipeline",
    "🔵 Embedding Space",
    "🔍 Retrieval",
    "🏆 Benchmark",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Chat (RAG question answering)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("💬 Savol-javob (RAG)")

    model_choice = st.selectbox("Model tanlang:", [
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    ])

    query = st.text_input(
        "Savolingizni yozing:",
        placeholder="Masalan: Rasululloh qaysi yili tug'ilgan?",
    )

    if st.button("Javob olish") and query:
        with st.spinner("Qidirilmoqda..."):
            queries = expand_query(query, groq_client, model=model_choice)
            results = multi_query_search(
                queries=queries,
                embed_model=embed_model,
                chroma_collection=collection,
                bm25_data=bm25_data,
                top_k=20,
            )
            chunks  = [r["text"]   for r in results]
            sources = [r["source"] for r in results]
            context = "\n\n".join(
                f"[{m.replace('.pdf', '').replace('_', ' ')}]:\n{c[:600]}"
                for c, m in zip(chunks, sources)
            )

            prompt = f"""Sen Muhammad alayhissalomning hayoti haqida ma'lumot beruvchi yordamchisan.

QOIDALAR:
1. Faqat xotirada mavjud MANBA asosida javob ber.
2. Savolni diqqat bilan o'qi — aynan so'ralgan narsaga javob ber.
3. Savoldagi har bir so'zni e'tiborga ol: "tug'ilgan" va "o'sgan" — ikki xil narsa.
4. MANBA dagi BARCHA tegishli ma'lumotlarni birlashtirib, to'liq javob ber. Agar bir nechta joy, ism yoki sana bo'lsa — hammasini ayt.
5. Masalan, "qayerda o'sgan?" savoliga: emizgan ayol (sut ona), qabila, shahar — hammasi muhim.
6. Agar MANBA da aniq javob bo'lmasa — "Bu haqida manbada ma'lumot yo'q" de.
7. O'zingdan hech qanday qo'shimcha ma'lumot kiritma, taxmin ham qilma.
8. Faqat MANBAdagi so'zlarni ishlatib javob ber — o'z bilimingdan foydalanma.
9. Javobda qaysi kitobdan olganingni ko'rsat. Masalan: "Tarixi Muhammadiy ga ko'ra..."

MANBA:
{context}

SAVOL: {query}

JAVOB:"""

            response = groq_client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = response.choices[0].message.content
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        st.markdown("### Javob:")
        st.write(answer)

        with st.expander("🔎 Kengaytirilgan so'rovlar"):
            for i, q in enumerate(queries):
                label = "📌 Asl savol" if i == 0 else f"🔁 Kengaytirilgan {i}"
                st.markdown(f"**{label}:** {q}")

        books = sorted(set(
            m.replace(".pdf", "").replace("_", " ").replace("-", " ")
            for m in sources if m
        ))
        st.caption("📚 Manba: " + " · ".join(books))

        with st.expander("Ishlatilgan chunklarni ko'rish"):
            for i, (chunk, source) in enumerate(zip(chunks, sources)):
                st.markdown(f"**Chunk {i+1}** — `{source}`")
                st.text(chunk)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Pipeline diagrams
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📊 Pipeline diagrammasi")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("V1 — Oddiy RAG")
        st.caption("Embedding only, top-5 chunks")
        img = os.path.join(ASSETS_DIR, "pipeline_v1.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.warning("Image not found. Run `src/visualize.py` to generate it.")
    with col2:
        st.subheader("V2 — Hybrid + Query Expansion")
        st.caption("BM25 + Embedding + RRF + Query Expansion, top-20 chunks")
        img2 = os.path.join(ASSETS_DIR, "pipeline_v2.png")
        if os.path.exists(img2):
            st.image(img2, use_container_width=True)
        else:
            st.warning("Image not found. Run `src/visualize.py` to generate it.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Embedding space (t-SNE)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🔵 Embedding Space (t-SNE)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("V1 — 2243 chunks")
        st.caption("Basic chunking, no prefix")
        img = os.path.join(ASSETS_DIR, "tsne_v1.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.warning("Image not found. Run `src/visualize.py` to generate it.")
    with col2:
        st.subheader("V2 — 2041 chunks")
        st.caption("OCR-cleaned, passage: prefix")
        img2 = os.path.join(ASSETS_DIR, "tsne_v2.png")
        if os.path.exists(img2):
            st.image(img2, use_container_width=True)
        else:
            st.warning("Image not found. Run `src/visualize.py` to generate it.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Interactive retrieval explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🔍 Retrieval natijalari")
    st.caption("Berilgan savolga mos top-5 chunk va ularning RRF scorelari.")

    retrieval_query = st.text_input(
        "Savol kiriting:",
        placeholder="Masalan: Rasulullohning onasi kim edi?",
        key="retrieval_input",
    )

    if st.button("Qidirish", key="retrieval_btn") and retrieval_query:
        with st.spinner("Qidirilmoqda..."):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            results = hybrid_search(
                query=retrieval_query,
                embed_model=embed_model,
                chroma_collection=collection,
                bm25_data=bm25_data,
                top_k=5,
            )
            docs    = [r["text"]      for r in results]
            sources = [r["source"]    for r in results]
            scores  = [r["rrf_score"] for r in results]

            labels = [
                f"Chunk {i+1} ({src.replace('.pdf', '')}): {d[:45]}..."
                if len(d) > 45 else f"Chunk {i+1}: {d}"
                for i, (d, src) in enumerate(zip(docs, sources))
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            palette = sns.color_palette("muted", len(scores))
            bars = ax.barh(range(len(scores)), scores,
                           color=palette, edgecolor="white", linewidth=0.8)
            for bar, s in zip(bars, scores):
                ax.text(bar.get_width() + 0.0002,
                        bar.get_y() + bar.get_height() / 2,
                        f"{s:.4f}", va="center", fontsize=9)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("RRF Score")
            ax.set_title(f'Top-5 chunks: "{retrieval_query}"', fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("Chunklar matnini ko'rish"):
            for i, (doc, src) in enumerate(zip(docs, sources)):
                st.markdown(f"**Chunk {i+1}** — `{src}`")
                st.text(doc[:500])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Benchmark results (LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("🏆 Benchmark natijalari")
    st.caption("LLM-as-judge: 18 savol · 3 model · 3 mezon (faithfulness, accuracy, completeness)")

    if not os.path.exists(SCORES_PATH):
        st.warning("results/benchmark_scores.json not found. Run `evaluation/judge.py` first.")
    else:
        with open(SCORES_PATH, encoding="utf-8") as _f:
            _scores = json.load(_f)

        _models  = _scores["models"]
        _summary = _scores["summary"]
        _details = _scores["details"]

        _name_map = {
            "qwen/qwen3-32b":                            "Qwen3-32B",
            "llama-3.1-8b-instant":                      "Llama3.1-8B",
            "meta-llama/llama-4-scout-17b-16e-instruct": "Llama4-Scout",
        }
        _names = [_name_map.get(m, m) for m in _models]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # ── Overall average bar chart ─────────────────────────────────────────
        st.subheader("Umumiy o'rtacha ball (1–5)")
        _overall = [_summary[m]["overall_avg"] for m in _models]
        _colors  = ["#4C72B0", "#55A868", "#C44E52"]

        _fig, _ax = plt.subplots(figsize=(7, 3))
        _bars = _ax.bar(_names, _overall, color=_colors, edgecolor="white", width=0.5)
        for _bar, _val in zip(_bars, _overall):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.04,
                     str(_val), ha="center", va="bottom", fontweight="bold", fontsize=11)
        _ax.set_ylim(0, 5.5)
        _ax.set_ylabel("Avg Score")
        _ax.set_title("Overall Average Score per Model", fontweight="bold")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _fig.patch.set_facecolor("white")
        _ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(_fig)
        plt.close(_fig)

        # ── By-difficulty grouped bar chart ──────────────────────────────────
        st.subheader("Qiyinlik darajasi bo'yicha")
        _diffs = ["easy", "medium", "hard"]
        _x     = np.arange(len(_diffs))
        _width = 0.25

        _fig2, _ax2 = plt.subplots(figsize=(8, 3.5))
        for _i, (_m, _nm, _c) in enumerate(zip(_models, _names, _colors)):
            _vals  = [_summary[_m]["by_difficulty"][d] for d in _diffs]
            _rects = _ax2.bar(_x + _i * _width, _vals, _width,
                              label=_nm, color=_c, edgecolor="white")
            for _r, _v in zip(_rects, _vals):
                _ax2.text(_r.get_x() + _r.get_width() / 2, _r.get_height() + 0.04,
                          str(_v), ha="center", va="bottom", fontsize=8)
        _ax2.set_xticks(_x + _width)
        _ax2.set_xticklabels(["Easy", "Medium", "Hard"])
        _ax2.set_ylim(0, 5.8)
        _ax2.set_ylabel("Avg Score")
        _ax2.set_title("Score by Difficulty Level", fontweight="bold")
        _ax2.legend(loc="upper right")
        _ax2.spines["top"].set_visible(False)
        _ax2.spines["right"].set_visible(False)
        _fig2.patch.set_facecolor("white")
        _ax2.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(_fig2)
        plt.close(_fig2)

        # ── Per-question detail table ─────────────────────────────────────────
        st.subheader("Savollar bo'yicha batafsil")
        _diff_filter = st.selectbox(
            "Qiyinlik:", ["Hammasi", "easy", "medium", "hard"], key="bench_diff"
        )

        for _row in _details:
            if _diff_filter != "Hammasi" and _row["difficulty"] != _diff_filter:
                continue

            _badge = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(
                _row["difficulty"], ""
            )
            st.markdown(f"**Q{_row['id']}.** {_row['question']}  {_badge}")

            _cols = st.columns(len(_models))
            for _col, _m, _nm in zip(_cols, _models, _names):
                if _m not in _row["scores"]:
                    continue
                _ms  = _row["scores"][_m]
                _avg = _ms["avg_score"]
                _j   = _ms["judgment"]
                _f   = _j.get("faithfulness", {}).get("score", "?")
                _a   = _j.get("accuracy",     {}).get("score", "?")
                _c_s = _j.get("completeness", {}).get("score", "?")
                with _col:
                    st.metric(label=_nm, value=f"{_avg}/5")
                    st.caption(f"F:{_f}  A:{_a}  C:{_c_s}")

            st.divider()
