#!/usr/bin/env python3
"""TTA (Test-Time Augmentation) inference with multi-crop averaging.

For each audio file takes N deterministic crops at different positions,
extracts embeddings for each, then averages → more robust embedding.

Crop positions (for audio longer than chunk_s):
  n_crops=1 → center crop only
  n_crops=3 → 20%, 50%, 80% of audio
  n_crops=5 → 10%, 30%, 50%, 70%, 90% of audio

Short audio (< chunk_s): full audio is used for all crops (no padding loss).

Usage:
    # TTA for fine-tuned CAM++
    bash scripts/run.sh scripts/run_tta_infer.py \
        --checkpoint weights/campplus_finetune_best.pt \
        --model-type campplus \
        --data-root data/ \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_campplus_ft_tta3.csv \
        --n-crops 3

    # TTA for fine-tuned ERes2Net
    bash scripts/run.sh scripts/run_tta_infer.py \
        --checkpoint weights/eres2net_finetune_best.pt \
        --model-type eres2net \
        --emb-dim 192 \
        --data-root data/ \
        --test-csv "extracted_data/Для участников/test_public.csv" \
        --output submission_eres2net_ft_tta3.csv \
        --n-crops 3
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


def get_crop(wav: np.ndarray, chunk_samples: int, position: float) -> np.ndarray:
    """Extract a deterministic crop at relative position [0.0, 1.0]."""
    n = len(wav)
    if n <= chunk_samples:
        return wav  # short audio: return as-is
    max_start = n - chunk_samples
    start = int(round(position * max_start))
    return wav[start: start + chunk_samples]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--model-type",   choices=["campplus", "eres2net", "eres2netv2", "ecapa", "wavlm"], default="campplus")
    p.add_argument("--emb-dim",      type=int, default=512,
                   help="512 for CAM++, 192 for ERes2Net")
    p.add_argument("--data-root",    required=True)
    p.add_argument("--test-csv",     required=True)
    p.add_argument("--output",       default="submission_tta.csv")
    p.add_argument("--cache-dir",    default="embeddings/")
    p.add_argument("--cache-name",   default=None)
    p.add_argument("--batch-size",   type=int, default=256)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--n-crops",      type=int, default=3,
                   help="Number of crops per file (1=center, 3=20/50/80%, 5=10/30/50/70/90%)")
    p.add_argument("--chunk-s",      type=float, default=3.0)
    p.add_argument("--k",            type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",        type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Crop positions
    if args.n_crops == 1:
        positions = [0.5]
    else:
        positions = [i / (args.n_crops - 1) for i in range(args.n_crops)]
    print(f"[tta_infer] Crop positions: {[f'{p:.2f}' for p in positions]}")

    chunk_samples = int(args.chunk_s * 16000)

    # Load model
    print(f"[tta_infer] Loading {args.model_type} from {args.checkpoint}...")
    if args.model_type == "campplus":
        from src.models.campplus_wrapper import CAMPlusWrapper
        model = CAMPlusWrapper(freeze_frontend=False, emb_dim=args.emb_dim)
    elif args.model_type == "eres2net":
        from src.models.eres2net_wrapper import ERes2NetWrapper
        model = ERes2NetWrapper(freeze_frontend=False, emb_dim=args.emb_dim)
    elif args.model_type == "eres2netv2":
        from src.models.eres2netv2_wrapper import ERes2NetV2Wrapper
        model = ERes2NetV2Wrapper(freeze_frontend=False, emb_dim=args.emb_dim)
    elif args.model_type == "ecapa":
        from src.models.ecapa_tdnn import ECAPA_TDNN
        model = ECAPA_TDNN(n_mels=80, channels=512, emb_dim=args.emb_dim, scale=8, sr=16000)
    else:  # wavlm
        from src.models.wavlm_wrapper import WavLMWrapper
        model = WavLMWrapper(emb_dim=192, freeze_backbone=False, use_grad_checkpointing=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["encoder_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[tta_infer] Model on {device}")

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root).resolve()
    n = len(filepaths)
    print(f"[tta_infer] Files: {n}, crops per file: {args.n_crops}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    cache_prefix = args.cache_name or f"{args.model_type}_ft_tta{args.n_crops}"
    cache_path = cache_dir / f"{cache_prefix}{suffix}.npy"

    if cache_path.exists():
        print(f"[tta_infer] Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        embeddings = None
        ptr = 0
        batches = [filepaths[i: i + args.batch_size] for i in range(0, n, args.batch_size)]

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            for batch_fps in tqdm(batches, desc=f"TTA {args.n_crops}-crop"):
                # Load full audio in parallel
                wavs = list(pool.map(load_item, [(fp, data_root) for fp in batch_fps]))
                B = len(wavs)

                crop_embs = []
                for pos in positions:
                    # Extract crop at this position for each audio
                    crops = [get_crop(w, chunk_samples, pos) for w in wavs]

                    # Pad to longest in batch → [B, T]
                    max_len = max(len(c) for c in crops)
                    padded = np.zeros((B, max_len), dtype=np.float32)
                    for i, c in enumerate(crops):
                        padded[i, : len(c)] = c

                    with torch.no_grad():
                        t = torch.from_numpy(padded).to(device)
                        embs = model(t)  # [B, D] L2-normalized

                    crop_embs.append(embs.cpu().numpy())

                # Average across crops, then L2-normalize
                avg = np.mean(crop_embs, axis=0).astype(np.float32)  # [B, D]
                norms = np.linalg.norm(avg, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                avg = avg / norms

                if embeddings is None:
                    embeddings = np.zeros((n, avg.shape[1]), dtype=np.float32)

                embeddings[ptr: ptr + B] = avg
                ptr += B

        np.save(cache_path, embeddings)
        print(f"[tta_infer] Saved embeddings to {cache_path}")

    print(f"[tta_infer] Embeddings: {embeddings.shape}")

    index, normalized = build_index(embeddings, use_gpu=not args.no_gpu_index)
    neighbors = find_neighbors(index, normalized, k=args.k)

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[tta_infer] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
