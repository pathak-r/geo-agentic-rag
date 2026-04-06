"""
Vector Store Management
Build and query FAISS index for document retrieval.
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from src.config import FAISS_INDEX_PATH, TOP_K_RESULTS


def build_faiss_index(documents: List[Dict], embeddings_model) -> None:
    """
    Build FAISS index from processed documents.
    Saves index and metadata to disk.
    """
    import faiss

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    print(f"Generating embeddings for {len(texts)} chunks...")
    vectors = embeddings_model.embed_documents(texts)
    vectors_np = np.array(vectors, dtype="float32")

    # Build FAISS index
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    # Save index
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))

    # Save texts and metadata alongside
    store_data = {
        "texts": texts,
        "metadatas": metadatas,
    }
    with open(os.path.join(FAISS_INDEX_PATH, "store_data.json"), "w") as f:
        json.dump(store_data, f)

    print(f"FAISS index built with {index.ntotal} vectors (dim={dimension})")
    print(f"Saved to: {FAISS_INDEX_PATH}")


def load_faiss_index():
    """Load FAISS index and associated data from disk."""
    import faiss

    index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    data_path = os.path.join(FAISS_INDEX_PATH, "store_data.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run ingest.py first."
        )

    index = faiss.read_index(index_path)

    with open(data_path, "r") as f:
        store_data = json.load(f)

    return index, store_data["texts"], store_data["metadatas"]


def search_documents(query: str, embeddings_model, top_k: int = TOP_K_RESULTS) -> List[Dict]:
    """
    Search the FAISS index for documents relevant to the query.
    Returns list of {text, metadata, score} dicts.
    """
    import faiss

    index, texts, metadatas = load_faiss_index()

    # Embed query
    query_vector = np.array(
        embeddings_model.embed_query(query), dtype="float32"
    ).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Search
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for missing results
            continue
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "score": float(score),
        })

    return results
