#!/usr/bin/env python3
"""End-to-end inference script for speaker retrieval.

Downloads model weights from HuggingFace (if not present), extracts embeddings
with TTA-10 for both CAM++ and ERes2Net, blends them, applies k-reciprocal
re-ranking, and saves submission.csv.

Usage:
    python infer.py \
        --data-root data/ \
        --test-csv extracted_data/test_public.csv \
        --output submission.csv

    # With custom weight paths:
    python infer.py \
        --data-root data/ \
        --test-csv extracted_data/test_public.csv \
        --campplus-ckpt weights/campplus_finetune_stage3_best.pt \
        --eres2net-ckpt weights/eres2net_finetune_stage3_best.pt \
        --output submission.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


HF_REPO = "s0ft44/kryptonit-tembr-weights"
CAMPPLUS_FILENAME = "campplus_finetune_stage3_best.pt"
ERES2NET_FILENAME = "eres2net_finetune_stage3_best.pt"
ALPHA = 0.55   # CAM++ weight
K1    = 70
K2    = 6
LAMBDA_VAL = 0.1
N_CROPS = 10
BATCH_SIZE = 64


def download_weights(weights_dir: Path):
    from huggingface_hub import hf_hub_download
    weights_dir.mkdir(parents=True, exist_ok=True)

    for filename in [CAMPPLUS_FILENAME, ERES2NET_FILENAME]:
        dest = weights_dir / filename
        if dest.exists():
            print(f"[weights] Found {dest}")
        else:
            print(f"[weights] Downloading {filename} from {HF_REPO}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=str(weights_dir),
            )
            print(f"[weights] Saved → {dest}")


def l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def extract_embeddings(model_type: str, checkpoint: Path, test_csv: str,
                       data_root: str, n_crops: int, batch_size: int) -> np.ndarray:
    from scripts.run_tta_infer import run_tta_infer
    emb = run_tta_infer(
        model_type=model_type,
        checkpoint=str(checkpoint),
        test_csv=test_csv,
        data_root=data_root,
        n_crops=n_crops,
        batch_size=batch_size,
        cache_name=None,   # no disk cache during inference
    )
    return emb


def parse_args():
    p = argparse.ArgumentParser(description="Speaker retrieval inference")
    p.add_argument("--data-root",      required=True, help="Root folder with test_public/")
    p.add_argument("--test-csv",       required=True, help="CSV with filepath column")
    p.add_argument("--output",         default="submission.csv")
    p.add_argument("--campplus-ckpt",  default=None, help="CAM++ checkpoint (auto-download if absent)")
    p.add_argument("--eres2net-ckpt",  default=None, help="ERes2Net checkpoint (auto-download if absent)")
    p.add_argument("--weights-dir",    default="weights", help="Where to store/find weights")
    p.add_argument("--n-crops",        type=int, default=N_CROPS)
    p.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    p.add_argument("--alpha",          type=float, default=ALPHA)
    p.add_argument("--k1",             type=int, default=K1)
    p.add_argument("--k2",             type=int, default=K2)
    p.add_argument("--lambda-val",     type=float, default=LAMBDA_VAL)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    weights_dir = Path(args.weights_dir)

    campplus_ckpt = Path(args.campplus_ckpt) if args.campplus_ckpt else weights_dir / CAMPPLUS_FILENAME
    eres2net_ckpt = Path(args.eres2net_ckpt) if args.eres2net_ckpt else weights_dir / ERES2NET_FILENAME

    # Check weights exist — do NOT download during inference (no internet)
    missing = [str(f) for f in [campplus_ckpt, eres2net_ckpt] if not f.exists()]
    if missing:
        print(f"[infer] ERROR: weights not found: {missing}")
        print(f"[infer] Download them first (with internet):")
        print(f"  python infer.py --download-weights --weights-dir {args.weights_dir}")
        sys.exit(1)

    # Extract embeddings
    print(f"\n[infer] Step 1/3 — CAM++ TTA×{args.n_crops} embeddings...")
    from scripts.run_tta_infer import main as tta_main
    import sys as _sys

    def run_tta(model_type, ckpt, cache_name):
        _sys.argv = [
            "run_tta_infer.py",
            "--checkpoint", str(ckpt),
            "--model-type", model_type,
            "--n-crops", str(args.n_crops),
            "--data-root", args.data_root,
            "--test-csv", args.test_csv,
            "--cache-name", cache_name,
            "--batch-size", str(args.batch_size),
        ]
        tta_main()

    run_tta("campplus", campplus_ckpt, "_infer_campplus_tmp")
    emb_a = l2(np.load("embeddings/_infer_campplus_tmp.npy").astype(np.float32))
    print(f"[infer] CAM++ embeddings: {emb_a.shape}")

    print(f"\n[infer] Step 1/3 — ERes2Net TTA×{args.n_crops} embeddings...")
    run_tta("eres2net", eres2net_ckpt, "_infer_eres2net_tmp")
    emb_b = l2(np.load("embeddings/_infer_eres2net_tmp.npy").astype(np.float32))
    print(f"[infer] ERes2Net embeddings: {emb_b.shape}")

    # Blend
    print(f"\n[infer] Step 2/3 — Blending (α={args.alpha:.2f} CAM++ + {1-args.alpha:.2f} ERes2Net)...")
    embeddings = l2(np.concatenate([args.alpha * emb_a, (1 - args.alpha) * emb_b], axis=1))
    print(f"[infer] Blended embeddings: {embeddings.shape}")

    # Rerank
    print(f"\n[infer] Step 3/3 — K-reciprocal re-ranking (k1={args.k1}, k2={args.k2}, λ={args.lambda_val})...")
    import pandas as pd
    from src.retrieval.reranking import rerank_and_retrieve
    from src.utils.submission import save_submission, validate_submission

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()

    neighbors = rerank_and_retrieve(
        embeddings,
        k=10,
        k1=args.k1,
        k2=args.k2,
        lambda_value=args.lambda_val,
    )

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=10)
    validate_submission(output_path, args.test_csv, k=10)

    # Cleanup tmp embeddings
    for tmp in ["embeddings/_infer_campplus_tmp.npy", "embeddings/_infer_eres2net_tmp.npy"]:
        Path(tmp).unlink(missing_ok=True)

    print(f"\n[infer] Done in {(time.time()-t0)/60:.1f} min → {output_path}")


if __name__ == "__main__":
    main()
