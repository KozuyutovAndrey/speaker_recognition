#!/usr/bin/env python3
"""Local Precision@K evaluation on a speaker-level train/val split.

Usage:
    python scripts/run_eval.py \\
        --data-root data/ \\
        --train-csv "extracted_data/Для участников/train.csv" \\
        --model     "extracted_data/Для участников/baseline.onnx" \\
        [--n-val-speakers 500] \\
        [--seed 42] \\
        [--batch-size 64] \\
        [--device cuda] \\
        [--cache-dir embeddings/] \\
        [--experiment-name baseline_onnx]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.extract_embeddings import extract_embeddings
from src.models.onnx_wrapper import OnnxEmbedder
from src.retrieval.faiss_search import build_index, find_neighbors, l2_normalize
from src.utils.experiment_logger import ExperimentResult, log_experiment
from src.utils.metrics import precision_at_k_report


def make_val_split(
    train_csv: Path,
    n_val_speakers: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Speaker-level split. Returns (train_df, val_df)."""
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)
    speakers = sorted(df["speaker_id"].unique())
    val_speakers = set(rng.choice(speakers, size=n_val_speakers, replace=False))
    val_df = df[df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
    train_df = df[~df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
    return train_df, val_df


def parse_args():
    p = argparse.ArgumentParser(description="Local Precision@K evaluation")
    p.add_argument("--data-root",       required=True)
    p.add_argument("--train-csv",       required=True)
    p.add_argument("--model",           required=True)
    p.add_argument("--n-val-speakers",  type=int, default=500)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--batch-size",      type=int, default=64)
    p.add_argument("--device",          default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-workers",     type=int, default=4)
    p.add_argument("--cache-dir",       default=None)
    p.add_argument("--experiment-name", default="baseline_onnx")
    p.add_argument("--no-gpu-index",    action="store_true")
    p.add_argument("--limit",           type=int, default=None,
                   help="Limit val files (for quick smoke-test)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    data_root = Path(args.data_root)
    train_csv = Path(args.train_csv)
    model_path = Path(args.model)

    # Build val split
    print(f"[run_eval] Building val split ({args.n_val_speakers} speakers, seed={args.seed})")
    _, val_df = make_val_split(train_csv, n_val_speakers=args.n_val_speakers, seed=args.seed)

    if args.limit:
        val_df = val_df.iloc[: args.limit].reset_index(drop=True)
        print(f"[run_eval] Limiting to {args.limit} val files")

    filepaths = val_df["filepath"].tolist()
    speaker_ids = val_df["speaker_id"].values
    print(f"[run_eval] Val files: {len(filepaths)}, unique speakers: {val_df['speaker_id'].nunique()}")

    # Cache
    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_limit{args.limit}" if args.limit else ""
        cache_path = cache_dir / f"{model_path.stem}_val{args.n_val_speakers}{suffix}.npy"

    # Load model
    print(f"[run_eval] Loading model: {model_path}")
    model = OnnxEmbedder(model_path=model_path, device=args.device)

    # Extract embeddings
    t_infer_start = time.time()
    embeddings = extract_embeddings(
        model=model,
        filepaths=filepaths,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=cache_path,
    )
    inference_time = time.time() - t_infer_start
    print(f"[run_eval] Embeddings shape: {embeddings.shape}, inference: {inference_time:.1f}s")

    # Build index and search
    use_gpu = not args.no_gpu_index
    index, normalized = build_index(embeddings, use_gpu=use_gpu)
    neighbors = find_neighbors(index, normalized, k=10)

    # Compute metrics
    metrics = precision_at_k_report(neighbors, speaker_ids, ks=[1, 5, 10])
    print(f"\n[run_eval] Results for '{args.experiment_name}':")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    # Log experiment
    result = ExperimentResult(
        name=args.experiment_name,
        model=str(model_path),
        config={
            "n_val_speakers": args.n_val_speakers,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        precision_at_1=metrics["P@1"],
        precision_at_5=metrics["P@5"],
        precision_at_10=metrics["P@10"],
        inference_time_sec=round(inference_time, 2),
    )
    log_experiment(result)

    elapsed = time.time() - t0
    print(f"\n[run_eval] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
