#!/usr/bin/env python3
"""Zero-shot inference with ECAPA2 (Jenthe/ECAPA2 on HuggingFace).

ECAPA2 is a significantly improved ECAPA-TDNN trained on 1600h+ data.
Uses SpeechBrain interface.

Usage:
    bash scripts/run.sh scripts/run_ecapa2.py \
        --data-root data/ \
        --test-csv  "extracted_data/Для участников/test_public.csv" \
        --output    submission_ecapa2.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.extract_embeddings import extract_embeddings
from src.retrieval.faiss_search import build_index, find_neighbors
from src.utils.submission import save_submission, validate_submission


class ECAPA2Embedder:
    """ECAPA2 embedder via SpeechBrain (Jenthe/ECAPA2)."""

    def __init__(self, device: str = "cuda"):
        from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="Jenthe/ECAPA2",
            savedir="weights/pretrained/ECAPA2",
            run_opts={"device": device},
        )
        self.device = device

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(waveforms).to(self.device)
            embs = self.model.encode_batch(t)  # (B, 1, D)
        return embs.squeeze(1).cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   required=True)
    p.add_argument("--test-csv",    required=True)
    p.add_argument("--output",      default="submission_ecapa2.csv")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dir",   default="embeddings/")
    p.add_argument("--k",           type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",       type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[run_ecapa2] Loading ECAPA2 model...")
    model = ECAPA2Embedder(device=device)

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root)
    print(f"[run_ecapa2] Files: {len(filepaths)}, device: {device}")

    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_limit{args.limit}" if args.limit else ""
        cache_path = cache_dir / f"ecapa2_test{suffix}.npy"

    embeddings = extract_embeddings(
        model=model,
        filepaths=filepaths,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        cache_path=cache_path,
    )
    print(f"[run_ecapa2] Embeddings: {embeddings.shape}")

    index, normalized = build_index(embeddings, use_gpu=not args.no_gpu_index)
    neighbors = find_neighbors(index, normalized, k=args.k)

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[run_ecapa2] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
