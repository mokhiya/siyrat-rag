# src/query_expansion.py
# Query expansion: use an LLM to generate additional search queries from the original question
import sys
import os

sys.path.append(os.path.dirname(__file__))


def expand_query(query, groq_client, model="llama-3.3-70b-versatile"):
    """
    Generate 3 additional search queries related to the given question.
    Returns the original query plus up to 3 expansions (4 queries total).
    """
    prompt = f"""Quyidagi savolga oid 3 ta qo'shimcha qidiruv so'rovini yoz.
Maqsad: Muhammad alayhissalom hayoti haqidagi o'zbek tilidagi kitoblardan ma'lumot topish.
Qidiruv so'rovlari xilma-xil bo'lsin — sinonimlar, boshqa ifodalar ishlatilsin.
Faqat qidiruv so'rovlarini yoz, har birini yangi qatorda. Boshqa hech narsa yozma.

Savol: {query}

Qidiruv so'rovlari:"""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150,
        )
        lines = response.choices[0].message.content.strip().split('\n')
        expansions = [l.strip().lstrip("123456789.-) ") for l in lines if l.strip()]
        expansions = [e for e in expansions if len(e) > 5][:3]
    except Exception:
        expansions = []

    return [query] + expansions  # original query first, then expanded variants


def multi_query_search(queries, embed_model, chroma_collection, bm25_data, top_k=20):
    """
    Run hybrid search for each query and merge the results.

    Chunks found across multiple queries accumulate a higher score.
    The original query is weighted 1.0; expansions are weighted 0.7.
    """
    from hybrid_search import hybrid_search

    all_results = {}

    for i, query in enumerate(queries):
        results = hybrid_search(
            query=query,
            embed_model=embed_model,
            chroma_collection=chroma_collection,
            bm25_data=bm25_data,
            top_k=15,
        )
        weight = 1.0 if i == 0 else 0.7

        for r in results:
            cid = r["id"]
            if cid not in all_results:
                all_results[cid] = dict(r)
                all_results[cid]["rrf_score"] *= weight
            else:
                # Boost score when a chunk appears in multiple query results
                all_results[cid]["rrf_score"] += r["rrf_score"] * weight

    combined = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)
    return combined[:top_k]
