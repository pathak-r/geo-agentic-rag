"""
Vector Store Management
FAISS index + optional BM25 hybrid; retrieval tuned for stable chat Q&A.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    EMBED_BATCH_SIZE,
    FAISS_INDEX_PATH,
    RAG_MODE,
    RAG_MULTI_QUERY,
    RAG_MULTI_QUERY_N,
    RAG_RERANK,
    RAG_RERANK_POOL,
    TOP_K_RESULTS,
)

_cache: Optional[Dict] = None


def _load_cache() -> Dict:
    global _cache
    if _cache is not None:
        return _cache

    import faiss
    from rank_bm25 import BM25Okapi

    index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    data_path = os.path.join(FAISS_INDEX_PATH, "store_data.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run ingest.py first."
        )

    faiss_index = faiss.read_index(index_path)

    with open(data_path, "r") as f:
        store_data = json.load(f)

    texts: List[str] = store_data["texts"]
    metadatas: List[Dict] = store_data["metadatas"]

    tokenized = [_tokenize(t) for t in texts]
    bm25_index = BM25Okapi(tokenized)

    _cache = {
        "faiss_index": faiss_index,
        "texts": texts,
        "metadatas": metadatas,
        "bm25_index": bm25_index,
    }
    print(f"[vector_store] Loaded {len(texts)} chunks (RAG_MODE={RAG_MODE}).")
    return _cache


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _dedupe_key(metadata: Dict, text: str) -> str:
    sf = metadata.get("source_file") or ""
    ci = metadata.get("chunk_index")
    if sf and ci is not None:
        return f"{sf}::{ci}"
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"hash::{h}"


def _dedupe_candidates(results: List[Dict]) -> List[Dict]:
    best: Dict[str, Dict] = {}
    best_score: Dict[str, float] = {}
    for r in results:
        key = _dedupe_key(r["metadata"], r["text"])
        s = float(r.get("score", 0.0))
        if key not in best_score or s > best_score[key]:
            best_score[key] = s
            best[key] = {**r, "score": s}
    return list(best.values())


def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_faiss_index(documents: List[Dict], embeddings_model) -> None:
    global _cache
    import faiss

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print(f"Generating embeddings for {len(texts)} chunks (batch_size={EMBED_BATCH_SIZE})...")
    parts: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vec = embeddings_model.embed_documents(batch)
        parts.append(np.array(vec, dtype="float32"))
    vectors_np = np.vstack(parts)

    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))

    store_data = {"texts": texts, "metadatas": metadatas}
    with open(os.path.join(FAISS_INDEX_PATH, "store_data.json"), "w") as f:
        json.dump(store_data, f)

    _cache = None
    print(f"FAISS index built with {index.ntotal} vectors (dim={dimension})")
    print(f"Saved to: {FAISS_INDEX_PATH}")


def load_faiss_index():
    c = _load_cache()
    return c["faiss_index"], c["texts"], c["metadatas"]


def generate_query_variants(query: str, n_variants: int = RAG_MULTI_QUERY_N) -> List[str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    prompt = (
        f"Generate {n_variants} alternative phrasings of the following question about "
        f"oil well operations and drilling data. Use different but equivalent terminology. "
        f"Output only the questions, one per line, no numbering.\n\nQuestion: {query}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        variants = lines[:n_variants]
    except Exception as e:
        print(f"[vector_store] Query variant generation failed ({e}); using original query only.")
        variants = []

    return [query] + variants


def search_documents(
    query: str,
    embeddings_model,
    top_k: int = TOP_K_RESULTS,
    mode: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve chunks. mode 'semantic' = FAISS only; 'hybrid' = FAISS + BM25 + RRF.
    """
    import faiss

    mode = (mode or RAG_MODE).strip().lower()
    if mode not in ("semantic", "hybrid"):
        mode = "semantic"

    cache = _load_cache()
    faiss_index = cache["faiss_index"]
    texts: List[str] = cache["texts"]
    metadatas: List[Dict] = cache["metadatas"]
    bm25_index = cache["bm25_index"]

    fetch_k = min(max(top_k * 2, top_k), len(texts))

    query_vector = np.array(
        embeddings_model.embed_query(query), dtype="float32"
    ).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    sims, faiss_indices = faiss_index.search(query_vector, fetch_k)
    faiss_ranked = [int(i) for i in faiss_indices[0] if i >= 0]

    if mode == "semantic":
        results = []
        for rank in range(len(faiss_indices[0])):
            idx = int(faiss_indices[0][rank])
            if idx < 0:
                continue
            sc = float(sims[0][rank])
            results.append({
                "text": texts[idx],
                "metadata": metadatas[idx],
                "score": sc,
            })
            if len(results) >= top_k:
                break
        return results

    tokenized_query = _tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked = np.argsort(bm25_scores)[::-1][:fetch_k].tolist()

    fused = _reciprocal_rank_fusion([faiss_ranked, bm25_ranked])[:top_k]
    results = []
    for idx, rrf_score in fused:
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "score": rrf_score,
        })
    return results


def _embedding_rerank(
    query: str,
    candidates: List[Dict],
    embeddings_model,
    top_k: int,
) -> List[Dict]:
    if not candidates:
        return []
    import faiss

    texts = [c["text"] for c in candidates]
    q_emb = np.array(embeddings_model.embed_query(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    parts: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs = embeddings_model.embed_documents(batch)
        parts.append(np.array(vecs, dtype="float32"))
    doc_mat = np.vstack(parts)
    faiss.normalize_L2(doc_mat)

    sims = (doc_mat @ q_emb.T).flatten()
    order = np.argsort(sims)[::-1][:top_k]
    out: List[Dict] = []
    for j in order:
        out.append({**candidates[int(j)], "score": float(sims[int(j)])})
    return out


def _merge_multi_query_rrf(
    query: str,
    embeddings_model,
    pool_k: int,
    mode: str,
) -> List[Dict]:
    queries = generate_query_variants(query, n_variants=RAG_MULTI_QUERY_N)
    rrf_k = 60
    scores: Dict[str, float] = {}
    store: Dict[str, Dict] = {}

    for q in queries:
        chunk_results = search_documents(q, embeddings_model, top_k=pool_k, mode=mode)
        for rank, r in enumerate(chunk_results):
            key = _dedupe_key(r["metadata"], r["text"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            store[key] = r

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    out: List[Dict] = []
    for k in sorted_keys[:pool_k]:
        out.append({**store[k], "score": scores[k]})
    return out


def retrieve_for_chat(
    query: str,
    embeddings_model,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Main entry for the document tool: optional multi-query, dedupe, embedding re-rank.
    """
    top_k = top_k or TOP_K_RESULTS
    cache = _load_cache()
    n_texts = len(cache["texts"])
    pool_k = min(RAG_RERANK_POOL, n_texts)
    pool_k = max(pool_k, top_k)

    mode = RAG_MODE if RAG_MODE in ("semantic", "hybrid") else "semantic"

    if RAG_MULTI_QUERY:
        candidates = _merge_multi_query_rrf(query, embeddings_model, pool_k, mode)
    else:
        candidates = search_documents(query, embeddings_model, top_k=pool_k, mode=mode)

    candidates = _dedupe_candidates(candidates)

    if RAG_RERANK and len(candidates) > top_k:
        candidates = _embedding_rerank(query, candidates, embeddings_model, top_k)
    else:
        candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]

    return candidates


def search_documents_multi_query(
    query: str, embeddings_model, top_k: int = TOP_K_RESULTS
) -> List[Dict]:
    """Backward-compatible alias — uses retrieve_for_chat (respects env flags)."""
    return retrieve_for_chat(query, embeddings_model, top_k=top_k)
