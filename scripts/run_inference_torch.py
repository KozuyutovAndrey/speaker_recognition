#!/usr/bin/env python3
"""Inference with a trained PyTorch ECAPA-TDNN checkpoint.

Usage:
    python scripts/run_inference_torch.py \\
        --checkpoint weights/ecapa_tdnn_aamsoft_best.pt \\
        --config     configs/train_ecapa.yaml \\
        --data-root  data/ \\
        --test-csv   "extracted_data/Для участников/test_public.csv" \\
        --output     submission.csv \\
        [--batch-size 128] \\
        [--cache-dir  embeddings/]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ecapa_tdnn import ECAPA_TDNN
from src.data.dataset import FlacDataset, collate_pad
from src.retrieval.faiss_search import build_index, find_neighbors, l2_normalize
from src.utils.submission import save_submission, validate_submission
from concurrent.futures import ThreadPoolExecutor


class TorchEmbedder:
    """Adapter to match the extract_embeddings interface."""

    def __init__(self, model: ECAPA_TDNN, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(waveforms).to(self.device)
            embs = self.model(t)
        return embs.cpu().numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--data-root",  required=True)
    p.add_argument("--test-csv",   required=True)
    p.add_argument("--output",     default="submission.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--cache-dir",  default=None)
    p.add_argument("--k",          type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",      type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    encoder = ECAPA_TDNN(
        n_mels=cfg.model.n_mels,
        channels=cfg.model.channels,
        emb_dim=cfg.model.emb_dim,
        scale=cfg.model.scale,
        sr=cfg.model.sr,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state"])
    print(f"[run_inference_torch] Loaded checkpoint (epoch={ckpt['epoch']}, "
          f"metrics={ckpt['metrics']})")

    model = TorchEmbedder(encoder, device=device)

    # File list
    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root)
    print(f"[run_inference_torch] Processing {len(filepaths)} files on {device}")

    # Cache
    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_stem = Path(args.checkpoint).stem
        suffix = f"_limit{args.limit}" if args.limit else ""
        cache_path = cache_dir / f"{ckpt_stem}_test{suffix}.npy"

    from src.inference.extract_embeddings import extract_embeddings
    embeddings = extract_embeddings(
        model=model,
        filepaths=filepaths,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_path=cache_path,
    )
    print(f"[run_inference_torch] Embeddings shape: {embeddings.shape}")

    use_gpu = not args.no_gpu_index
    index, normalized = build_index(embeddings, use_gpu=use_gpu)
    neighbors = find_neighbors(index, normalized, k=args.k)

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)

    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[run_inference_torch] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
