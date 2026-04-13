"""
Vector Store Management
Build and query FAISS index for document retrieval.
Uses hybrid retrieval: FAISS (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion.
"""
import os
import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.config import FAISS_INDEX_PATH, TOP_K_RESULTS

# ── Module-level cache (populated on first query, reused for the process lifetime) ──
_cache: Optional[Dict] = None


def _load_cache() -> Dict:
    """Load FAISS index, texts, metadatas, and BM25 index once and cache them."""
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
    print(f"[vector_store] Loaded {len(texts)} chunks into FAISS + BM25 hybrid index.")
    return _cache


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace for BM25."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]], k: int = 60
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked lists of document indices using RRF.
    score(d) = sum( 1 / (k + rank_i(d)) ) for each list i that contains d.
    Returns list of (idx, rrf_score) sorted descending.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_faiss_index(documents: List[Dict], embeddings_model) -> None:
    """
    Build FAISS index from processed documents.
    Saves index and metadata to disk.
    Call this from ingest.py — invalidates the in-memory cache afterwards.
    """
    global _cache
    import faiss

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = embeddings_model.embed_documents(texts)
    vectors_np = np.array(vectors, dtype="float32")

    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))

    store_data = {"texts": texts, "metadatas": metadatas}
    with open(os.path.join(FAISS_INDEX_PATH, "store_data.json"), "w") as f:
        json.dump(store_data, f)

    _cache = None  # invalidate so next query rebuilds BM25 from fresh data
    print(f"FAISS index built with {index.ntotal} vectors (dim={dimension})")
    print(f"Saved to: {FAISS_INDEX_PATH}")


def load_faiss_index():
    """Legacy helper — kept for backward compatibility."""
    c = _load_cache()
    return c["faiss_index"], c["texts"], c["metadatas"]


def search_documents(query: str, embeddings_model, top_k: int = TOP_K_RESULTS) -> List[Dict]:
    """
    Hybrid search: FAISS (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion.
    Returns list of {text, metadata, score} dicts, deduplicated and ranked.
    """
    import faiss

    cache = _load_cache()
    faiss_index: "faiss.Index" = cache["faiss_index"]
    texts: List[str] = cache["texts"]
    metadatas: List[Dict] = cache["metadatas"]
    bm25_index = cache["bm25_index"]

    fetch_k = min(top_k * 2, len(texts))  # fetch more candidates before RRF

    # ── FAISS retrieval ──────────────────────────────────────────────────────────
    query_vector = np.array(
        embeddings_model.embed_query(query), dtype="float32"
    ).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    _, faiss_indices = faiss_index.search(query_vector, fetch_k)
    faiss_ranked = [int(i) for i in faiss_indices[0] if i >= 0]

    # ── BM25 retrieval ───────────────────────────────────────────────────────────
    tokenized_query = _tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked = np.argsort(bm25_scores)[::-1][:fetch_k].tolist()

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────────────
    fused = _reciprocal_rank_fusion([faiss_ranked, bm25_ranked])[:top_k]

    results = []
    for idx, rrf_score in fused:
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "score": rrf_score,
        })

    return results
