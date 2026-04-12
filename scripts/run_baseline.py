#!/usr/bin/env python3
"""End-to-end baseline pipeline: extract embeddings → retrieve neighbors → write submission.

Usage:
    python scripts/run_baseline.py \\
        --data-root "extracted_data/Для участников" \\
        --test-csv  "extracted_data/Для участников/test_public.csv" \\
        --model     "extracted_data/Для участников/baseline.onnx" \\
        --output    submission.csv \\
        [--batch-size 64] \\
        [--device cuda|cpu] \\
        [--cache-dir embeddings/] \\
        [--limit N]   # process only first N files (for quick testing)
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.inference.extract_embeddings import extract_embeddings
from src.models.onnx_wrapper import OnnxEmbedder
from src.retrieval.faiss_search import build_index, find_neighbors, l2_normalize
from src.utils.submission import save_submission, validate_submission


def parse_args():
    p = argparse.ArgumentParser(description="Baseline speaker retrieval pipeline")
    p.add_argument("--data-root", required=True, help="Root dir where audio files live")
    p.add_argument("--test-csv",  required=True, help="Path to test_public.csv")
    p.add_argument("--model",     required=True, help="Path to baseline.onnx")
    p.add_argument("--output",    default="submission.csv", help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--cache-dir", default=None, help="Directory for cached .npy embeddings")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--k",         type=int, default=10)
    p.add_argument("--limit",     type=int, default=None, help="Process only first N files")
    p.add_argument("--no-gpu-index", action="store_true", help="Use CPU FAISS index")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    data_root = Path(args.data_root)
    test_csv  = Path(args.test_csv)
    model_path = Path(args.model)
    output_path = Path(args.output)

    # Load file list
    df = pd.read_csv(test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]
        print(f"[run_baseline] Limiting to {args.limit} files")

    print(f"[run_baseline] Files to process: {len(filepaths)}")

    # Cache path
    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_stem = model_path.stem
        suffix = f"_limit{args.limit}" if args.limit else ""
        cache_path = cache_dir / f"{model_stem}_test{suffix}.npy"

    # Load model
    print(f"[run_baseline] Loading model: {model_path}")
    model = OnnxEmbedder(model_path=model_path, device=args.device)

    # Extract embeddings
    embeddings = extract_embeddings(
        model=model,
        filepaths=filepaths,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=cache_path,
    )
    print(f"[run_baseline] Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    print("[run_baseline] Building FAISS index ...")
    use_gpu = not args.no_gpu_index
    index, normalized = build_index(embeddings, use_gpu=use_gpu)

    # Find neighbors
    print(f"[run_baseline] Searching {args.k} nearest neighbors ...")
    neighbors = find_neighbors(index, normalized, k=args.k)

    # Save submission
    save_submission(filepaths, neighbors, output_path, k=args.k)

    # Validate (only when processing the full test set)
    if args.limit is None:
        validate_submission(output_path, test_csv, k=args.k)

    elapsed = time.time() - t0
    print(f"[run_baseline] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
