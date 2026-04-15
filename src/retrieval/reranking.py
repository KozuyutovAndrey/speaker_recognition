"""K-reciprocal re-ranking for speaker retrieval (Zhong et al., 2017).

Key idea: if A is in top-k of B AND B is in top-k of A → stronger evidence.
Reweights distances using Jaccard similarity of reciprocal neighbor sets.

Reference: "Re-ranking Person Re-identification with k-reciprocal Encoding"
           Zhong et al., CVPR 2017. Adapted for speaker retrieval.
"""

import numpy as np
import faiss


def _get_topk_faiss(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return top-k neighbor indices for each query using FAISS GPU. [N, k]"""
    N, D = embeddings.shape
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, D)
    index.add(embeddings)
    # k+1 to include self, then strip it
    _, indices = index.search(embeddings, k + 1)
    # Remove self (first column is always the query itself)
    result = np.zeros((N, k), dtype=np.int64)
    for i in range(N):
        row = [j for j in indices[i] if j != i][:k]
        result[i] = row
    return result


def rerank_and_retrieve(
    embeddings: np.ndarray,
    k: int = 10,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    K-reciprocal re-ranking — memory-efficient version using sparse V matrix.

    Instead of computing N×N full distance matrix (67 GB for 134k vectors),
    works only with top-(k1*2) FAISS neighbors per query.

    Args:
        embeddings:   L2-normalized embeddings [N, D]
        k1:           k-reciprocal neighborhood size
        k2:           query expansion size
        lambda_value: weight for original cosine distance (0=pure rerank, 1=original)

    Returns:
        neighbors: [N, k] reranked neighbor indices
    """
    N = len(embeddings)
    search_k = min(k1 * 2 + 1, N)
    retrieve_k = min(k + k1, N)  # retrieve extra candidates for re-ranking

    print(f"[reranking] FAISS top-{search_k} search for {N} queries...")
    topk_indices = _get_topk_faiss(embeddings, search_k)   # [N, search_k]

    # Build neighbor sets and cosine distances for top candidates only
    print(f"[reranking] Building reciprocal features (k1={k1}, k2={k2})...")

    # V stored as dict of {j: weight} per query i — sparse representation
    V = [dict() for _ in range(N)]

    # Precompute cosine distances for topk neighbors
    topk_sims = np.einsum('nd,kd->nk', embeddings,
                          embeddings[topk_indices.reshape(-1)].reshape(N, search_k, -1).reshape(N * search_k, -1)
                          ).reshape(N, search_k) if False else None

    # Simpler: compute per-row
    for i in range(N):
        neighbors_i = set(topk_indices[i, :k1])

        # k-reciprocal: keep j if i is in top-k1 of j
        R = set()
        for j in neighbors_i:
            if i in set(topk_indices[j, :k1]):
                R.add(j)

        # Expand R
        R_expanded = set(R)
        for j in list(R):
            neighbors_j = set(topk_indices[j, :max(k1 // 2, 1)])
            if len(R & neighbors_j) >= (2 / 3) * k1:
                R_expanded |= neighbors_j

        # Gaussian weights using cosine similarity
        for j in R_expanded:
            sim_ij = float(embeddings[i] @ embeddings[j])
            dist_ij = max(1.0 - sim_ij, 0.0)
            V[i][j] = np.exp(-dist_ij)

        # Normalize
        total = sum(V[i].values()) + 1e-8
        for j in V[i]:
            V[i][j] /= total

    # k2 query expansion
    if k2 > 1:
        print(f"[reranking] k2={k2} expansion...")
        V_new = [dict() for _ in range(N)]
        for i in range(N):
            # Average V over top-k2 neighbors
            top_k2 = list(topk_indices[i, :k2])
            all_keys = set(V[i].keys())
            for nb in top_k2:
                all_keys |= V[nb].keys()
            for key in all_keys:
                vals = [V[idx].get(key, 0.0) for idx in [i] + top_k2]
                avg = sum(vals) / (len(top_k2) + 1)
                if avg > 0:
                    V_new[i][key] = avg
        V = V_new

    # Re-rank: for each query, score candidates by Jaccard + original distance
    print(f"[reranking] Re-scoring top-{retrieve_k} candidates...")
    topk_cands = _get_topk_faiss(embeddings, retrieve_k)  # [N, retrieve_k]

    final_neighbors = np.zeros((N, k), dtype=np.int64)

    for i in range(N):
        cands = list(topk_cands[i])
        scores = []
        V_i_sum = sum(V[i].values()) + 1e-8

        for j in cands:
            # Original cosine distance
            orig_dist = max(1.0 - float(embeddings[i] @ embeddings[j]), 0.0)

            # Jaccard distance
            common_keys = set(V[i].keys()) & set(V[j].keys())
            intersect = sum(min(V[i][k_], V[j][k_]) for k_ in common_keys)
            V_j_sum = sum(V[j].values()) + 1e-8
            union = V_i_sum + V_j_sum - intersect + 1e-8
            jaccard_dist = max(1.0 - intersect / union, 0.0)

            final_score = (1 - lambda_value) * jaccard_dist + lambda_value * orig_dist
            scores.append((final_score, j))

        scores.sort()
        final_neighbors[i] = [j for _, j in scores[:k]]

    print("[reranking] Done.")
    return final_neighbors
