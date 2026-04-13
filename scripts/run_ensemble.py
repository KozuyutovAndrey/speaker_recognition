#!/usr/bin/env python3
"""Ensemble embeddings from two models by weighted average.

Loads precomputed .npy embeddings, L2-normalizes each, blends with weight alpha,
L2-normalizes result, then builds FAISS index and saves submission.

Usage:
    bash scripts/run.sh scripts/run_ensemble.py \
        --emb-a  embeddings/campplus_test.npy \
        --emb-b  embeddings/ecapa_tdnn_v2_realaugment_best_test.npy \
        --alpha  0.7 \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_ensemble.csv

    # Grid search over alpha:
    bash scripts/run.sh scripts/run_ensemble.py --grid-search
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.faiss_search import build_index, find_neighbors
from src.utils.submission import save_submission, validate_submission


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def blend(emb_a: np.ndarray, emb_b: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted concatenation of L2-normalized embeddings, then re-normalize.

    Works regardless of embedding dimensions (512 vs 192).
    Cosine similarity of concat ≈ alpha²·sim_a + (1-alpha)²·sim_b.
    alpha=1.0 → pure CAM++, alpha=0.0 → pure ECAPA.
    """
    a = l2_normalize(emb_a.astype(np.float32))
    b = l2_normalize(emb_b.astype(np.float32))
    mixed = np.concatenate([alpha * a, (1.0 - alpha) * b], axis=1)
    return l2_normalize(mixed)


def make_submission(embeddings, filepaths, output_path, test_csv, k=10):
    index, normalized = build_index(embeddings, use_gpu=True)
    neighbors = find_neighbors(index, normalized, k=k)
    save_submission(filepaths, neighbors, output_path, k=k)
    validate_submission(output_path, test_csv, k=k)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-a",    default="embeddings/campplus_test.npy")
    p.add_argument("--emb-b",    default="embeddings/ecapa_tdnn_v2_realaugment_best_test.npy")
    p.add_argument("--alpha",    type=float, default=0.7,
                   help="Weight for emb-a (CAM++). emb-b gets (1-alpha).")
    p.add_argument("--test-csv", default="extracted_data/Для участников/test_public.csv")
    p.add_argument("--output",   default="submission_ensemble.csv")
    p.add_argument("--grid-search", action="store_true",
                   help="Try alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] and save all")
    p.add_argument("--k",        type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[ensemble] Loading embeddings...")
    emb_a = np.load(args.emb_a)
    emb_b = np.load(args.emb_b)
    print(f"[ensemble] CAM++:    {emb_a.shape}")
    print(f"[ensemble] ECAPA v2: {emb_b.shape}")
    assert emb_a.shape[0] == emb_b.shape[0], "Embedding count mismatch!"

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()

    if args.grid_search:
        alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for alpha in alphas:
            mixed = blend(emb_a, emb_b, alpha)
            out = Path(f"submission_ensemble_a{int(alpha*10)}.csv")
            print(f"\n[ensemble] alpha={alpha:.1f} (CAM++={alpha:.1f}, ECAPA={1-alpha:.1f}) → {out}")
            make_submission(mixed, filepaths, out, args.test_csv, k=args.k)
        print("\n[ensemble] Grid search done. Submit each to compare.")
    else:
        mixed = blend(emb_a, emb_b, args.alpha)
        print(f"[ensemble] alpha={args.alpha} (CAM++={args.alpha}, ECAPA v2={1-args.alpha:.1f})")
        make_submission(mixed, filepaths, Path(args.output), args.test_csv, k=args.k)
        print(f"[ensemble] → {args.output}")


if __name__ == "__main__":
    main()
