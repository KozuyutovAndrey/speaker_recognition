#!/usr/bin/env python3
"""Apply k-reciprocal re-ranking to precomputed embeddings.

Loads one or two .npy embedding files, optionally blends them,
then applies re-ranking and saves submission.

Usage:
    # Re-rank single model
    python scripts/run_rerank.py \
        --emb-a embeddings/campplus_ft_tta3.npy \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_rerank_campplus.csv

    # Re-rank ensemble
    python scripts/run_rerank.py \
        --emb-a embeddings/campplus_ft_tta3.npy \
        --emb-b embeddings/eres2net_ft_tta3.npy \
        --alpha 0.6 \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_rerank_ensemble.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.reranking import rerank_and_retrieve
from src.utils.submission import save_submission, validate_submission


def l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-a",       required=True, help="Primary embeddings .npy")
    p.add_argument("--emb-b",       default=None,  help="Secondary embeddings .npy (optional)")
    p.add_argument("--emb-c",       default=None,  help="Tertiary embeddings .npy (optional)")
    p.add_argument("--alpha",       type=float, default=0.6,
                   help="Weight for emb-a when blending (1-alpha for emb-b)")
    p.add_argument("--gamma",       type=float, default=0.0,
                   help="Weight for emb-c; alpha and (1-alpha-gamma) used for emb-a/emb-b")
    p.add_argument("--test-csv",    required=True)
    p.add_argument("--output",      default="submission_reranked.csv")
    p.add_argument("--k",           type=int, default=10)
    p.add_argument("--k1",          type=int, default=20,
                   help="k-reciprocal neighborhood size")
    p.add_argument("--k2",          type=int, default=6,
                   help="Query expansion neighborhood size")
    p.add_argument("--lambda-val",  type=float, default=0.3,
                   help="Weight for original distance (0=pure rerank, 1=original)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print(f"[rerank] Loading embeddings...")
    emb_a = l2(np.load(args.emb_a).astype(np.float32))
    print(f"[rerank] emb-a: {emb_a.shape}")

    if args.emb_b and args.emb_c:
        emb_b = l2(np.load(args.emb_b).astype(np.float32))
        emb_c = l2(np.load(args.emb_c).astype(np.float32))
        print(f"[rerank] emb-b: {emb_b.shape}, emb-c: {emb_c.shape}")
        beta = 1 - args.alpha - args.gamma
        embeddings = l2(np.concatenate([args.alpha * emb_a, beta * emb_b, args.gamma * emb_c], axis=1))
        print(f"[rerank] blended: {embeddings.shape} (alpha={args.alpha}, beta={beta:.2f}, gamma={args.gamma})")
    elif args.emb_b:
        emb_b = l2(np.load(args.emb_b).astype(np.float32))
        print(f"[rerank] emb-b: {emb_b.shape}")
        embeddings = l2(np.concatenate([args.alpha * emb_a, (1 - args.alpha) * emb_b], axis=1))
        print(f"[rerank] blended: {embeddings.shape} (alpha={args.alpha})")
    else:
        embeddings = emb_a

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()

    print(f"[rerank] k1={args.k1}, k2={args.k2}, lambda={args.lambda_val}")
    neighbors = rerank_and_retrieve(
        embeddings,
        k=args.k,
        k1=args.k1,
        k2=args.k2,
        lambda_value=args.lambda_val,
    )

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    validate_submission(output_path, args.test_csv, k=args.k)
    print(f"[rerank] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
