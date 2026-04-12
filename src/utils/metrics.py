"""Precision@K for speaker retrieval evaluation."""

import numpy as np


def precision_at_k(
    neighbors: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Compute mean Precision@K over all queries.

    Args:
        neighbors: int64 array of shape (N, K_max), neighbor indices
        labels:    int or str array of shape (N,), speaker labels
        k:         cutoff

    Returns:
        Mean P@K across all queries
    """
    assert k <= neighbors.shape[1], f"k={k} > available neighbors={neighbors.shape[1]}"
    labels = np.asarray(labels)
    top_k = neighbors[:, :k]
    hits = (labels[top_k] == labels[:, None]).sum(axis=1)
    # Each query: how many of top-k share the same speaker?
    # Exclude self is already done in find_neighbors
    return float(hits.mean() / k)


def precision_at_k_report(
    neighbors: np.ndarray,
    labels: np.ndarray,
    ks: list[int] = (1, 5, 10),
) -> dict[str, float]:
    """Return P@K for multiple K values."""
    return {f"P@{k}": precision_at_k(neighbors, labels, k) for k in ks}
