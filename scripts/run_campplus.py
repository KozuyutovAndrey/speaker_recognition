#!/usr/bin/env python3
"""Zero-shot inference with ModelScope speaker verification models.

Supports CAM++, ERes2Net, and other DAMO/3D-Speaker models.
Bypasses slow sequential pipeline loop — direct GPU batched inference.

Usage:
    # CAM++
    bash scripts/run.sh scripts/run_campplus.py \
        --data-root data/ \
        --test-csv  "extracted_data/Для участников/test_public.csv" \
        --output    submission_campplus.csv

    # ERes2Net English
    bash scripts/run.sh scripts/run_campplus.py \
        --model-id  damo/speech_eres2net_sv_en_voxceleb_16k \
        --data-root data/ \
        --test-csv  "extracted_data/Для участников/test_public.csv" \
        --output    submission_eres2net.csv \
        --cache-name eres2net_test
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",     default="damo/speech_campplus_sv_en_voxceleb_16k")
    p.add_argument("--model-revision", default="v1.0.2")
    p.add_argument("--cache-name",   default=None, help="Override cache filename prefix")
    p.add_argument("--data-root",    required=True)
    p.add_argument("--test-csv",     required=True)
    p.add_argument("--output",       default="submission_campplus.csv")
    p.add_argument("--cache-dir",    default="embeddings/")
    p.add_argument("--batch-size",   type=int, default=256)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--k",            type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",        type=int, default=None)
    p.add_argument("--torchaudio-frontend", action="store_true",
                   help="Use GPU-batched torchaudio MelSpectrogram instead of Kaldi.fbank "
                        "(required for ERes2Net which only returns 1 embedding per batch call)")
    p.add_argument("--n-mels",       type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print(f"[run_campplus] Loading {args.model_id}...")
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    pipe = pipeline(
        task=Tasks.speaker_verification,
        model=args.model_id,
        model_revision=args.model_revision,
    )
    model = pipe.model
    model.eval()
    device = next(model.parameters()).device
    print(f"[run_campplus] Model device: {device}")

    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root).resolve()
    n = len(filepaths)
    print(f"[run_campplus] Files: {n}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    cache_prefix = args.cache_name or args.model_id.replace("/", "_")
    cache_path = cache_dir / f"{cache_prefix}_test{suffix}.npy"

    if cache_path.exists():
        print(f"[run_campplus] Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        embeddings = None
        ptr = 0
        batches = [
            filepaths[i: i + args.batch_size]
            for i in range(0, n, args.batch_size)
        ]

        # GPU-batched torchaudio frontend (needed for ERes2Net which returns only
        # 1 embedding per batch call due to sequential Kaldi.fbank inside).
        if args.torchaudio_frontend:
            import torchaudio.transforms as T
            frontend = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,   # 25 ms at 16kHz
                hop_length=160,   # 10 ms at 16kHz
                n_mels=args.n_mels,
                center=False,
            ).to(device)
            print(f"[run_campplus] Using GPU-batched torchaudio frontend (n_mels={args.n_mels})")

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            for batch_fps in tqdm(batches, desc="embeddings"):
                # Load audio in parallel
                wavs = list(pool.map(load_item, [(fp, data_root) for fp in batch_fps]))

                # Pad to longest in batch → [B, T]
                max_len = max(len(w) for w in wavs)
                padded = np.zeros((len(wavs), max_len), dtype=np.float32)
                for i, w in enumerate(wavs):
                    padded[i, : len(w)] = w

                # GPU inference
                with torch.no_grad():
                    if args.torchaudio_frontend:
                        # GPU-batched: bypass broken Kaldi.fbank loop in model.forward()
                        t = torch.from_numpy(padded).to(device)
                        specs = frontend(t)                          # [B, n_mels, T']
                        specs = torch.log(specs + 1e-6)
                        specs = specs.transpose(1, 2)               # [B, T', n_mels]
                        specs = specs - specs.mean(dim=1, keepdim=True)
                        embs = model.embedding_model(specs)         # [B, D]
                        embs = embs.detach().cpu()
                    else:
                        embs = model(padded)  # [B, D] — works for CAM++

                embs = embs.numpy().astype(np.float32)
                # L2 normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embs = embs / norms

                if embeddings is None:
                    embeddings = np.zeros((n, embs.shape[1]), dtype=np.float32)

                embeddings[ptr: ptr + len(embs)] = embs
                ptr += len(embs)

        np.save(cache_path, embeddings)
        print(f"[run_campplus] Saved embeddings to {cache_path}")

    print(f"[run_campplus] Embeddings: {embeddings.shape}")

    index, normalized = build_index(embeddings, use_gpu=not args.no_gpu_index)
    neighbors = find_neighbors(index, normalized, k=args.k)

    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[run_campplus] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
