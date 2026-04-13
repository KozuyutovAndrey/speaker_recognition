#!/usr/bin/env python3
"""Inference with fine-tuned CAM++ checkpoint.

Loads CAMPlusWrapper + encoder_state from checkpoint, runs GPU-batched inference.

Usage:
    bash scripts/run.sh scripts/run_campplus_ft_infer.py \
        --checkpoint weights/campplus_finetune_best.pt \
        --data-root data/ \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_campplus_ft.csv
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.campplus_wrapper import CAMPlusWrapper
from src.retrieval.faiss_search import build_index, find_neighbors
from src.utils.audio import load_audio, normalize_amplitude
from src.utils.submission import save_submission, validate_submission


def load_item(args):
    fp, data_root = args
    try:
        wav = load_audio(data_root / fp, target_sr=16000, max_duration_s=10.0)
        wav = normalize_amplitude(wav)
        return wav
    except Exception as e:
        print(f"[WARNING] Failed {fp}: {e}")
        return np.zeros(16000, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="weights/campplus_finetune_best.pt")
    p.add_argument("--data-root",    required=True)
    p.add_argument("--test-csv",     required=True)
    p.add_argument("--output",       default="submission_campplus_ft.csv")
    p.add_argument("--cache-dir",    default="embeddings/")
    p.add_argument("--cache-name",   default="campplus_ft_test")
    p.add_argument("--batch-size",   type=int, default=256)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--k",            type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",        type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print(f"[campplus_ft_infer] Loading model from {args.checkpoint}...")
    model = CAMPlusWrapper(freeze_frontend=False, emb_dim=512)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["encoder_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[campplus_ft_infer] Model on {device}")

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root).resolve()
    n = len(filepaths)
    print(f"[campplus_ft_infer] Files: {n}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    cache_path = cache_dir / f"{args.cache_name}{suffix}.npy"

    if cache_path.exists():
        print(f"[campplus_ft_infer] Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        embeddings = None
        ptr = 0
        batches = [filepaths[i: i + args.batch_size] for i in range(0, n, args.batch_size)]

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            for batch_fps in tqdm(batches, desc="CAM++ FT embeddings"):
                wavs = list(pool.map(load_item, [(fp, data_root) for fp in batch_fps]))

                max_len = max(len(w) for w in wavs)
                padded = np.zeros((len(wavs), max_len), dtype=np.float32)
                for i, w in enumerate(wavs):
                    padded[i, : len(w)] = w

                with torch.no_grad():
                    t = torch.from_numpy(padded).to(device)
                    embs = model(t)  # [B, 512] L2-normalized

                embs = embs.cpu().numpy().astype(np.float32)

                if embeddings is None:
                    embeddings = np.zeros((n, embs.shape[1]), dtype=np.float32)

                embeddings[ptr: ptr + len(embs)] = embs
                ptr += len(embs)

        np.save(cache_path, embeddings)
        print(f"[campplus_ft_infer] Saved embeddings to {cache_path}")

    print(f"[campplus_ft_infer] Embeddings: {embeddings.shape}")

    index, normalized = build_index(embeddings, use_gpu=not args.no_gpu_index)
    neighbors = find_neighbors(index, normalized, k=args.k)

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[campplus_ft_infer] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
