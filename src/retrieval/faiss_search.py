"""FAISS-based nearest neighbor retrieval for speaker embeddings."""

import numpy as np


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + eps)


def build_index(embeddings: np.ndarray, use_gpu: bool = True):
    """Build a FAISS FlatIP index from (optionally GPU-resident) embeddings.

    Args:
        embeddings: float32 array of shape (N, D). Will be L2-normalized internally.
        use_gpu:    Move index to GPU if available.

    Returns:
        (index, normalized_embeddings)
    """
    import faiss

    normalized = l2_normalize(embeddings).astype(np.float32)
    d = normalized.shape[1]
    index = faiss.IndexFlatIP(d)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[faiss_search] Using GPU index")
        except Exception as e:
            print(f"[faiss_search] GPU unavailable ({e}), using CPU index")

    index.add(normalized)
    return index, normalized


def find_neighbors(
    index,
    query_embeddings: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Find k nearest neighbors for each query, excluding self-match.

    Searches for k+1 neighbors, then removes the query's own index.

    Args:
        index:            FAISS index built with build_index()
        query_embeddings: L2-normalized float32 array of shape (N, D)
        k:                Number of neighbors to return

    Returns:
        neighbors: int64 array of shape (N, k)
    """
    n = len(query_embeddings)
    _, I = index.search(query_embeddings.astype(np.float32), k + 1)

    result = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        row = I[i]
        # Remove self (exact match at distance 1.0 for normalized vectors)
        filtered = row[row != i]
        if len(filtered) < k:
            # Fallback: fill missing with arbitrary valid indices
            missing = k - len(filtered)
            extra = np.array([j for j in range(n) if j != i and j not in filtered[:k]])[:missing]
            filtered = np.concatenate([filtered, extra])
        result[i] = filtered[:k]

    return result
