#!/usr/bin/env python3
"""Inference with a zero-shot pre-trained speaker verification model.

Supports:
  - speechbrain/spkrec-ecapa-voxceleb  (ECAPA-TDNN, VoxCeleb1+2)
  - microsoft/wavlm-base-plus-sv       (WavLM, noisy pre-training)

Usage:
    python scripts/run_pretrained.py \
        --model speechbrain/spkrec-ecapa-voxceleb \
        --data-root data/ \
        --test-csv  "extracted_data/Для участников/test_public.csv" \
        --output    submission_pretrained.csv \
        [--batch-size 64] [--cache-dir embeddings/]
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


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class SpeechBrainEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        from speechbrain.pretrained import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"weights/pretrained/{model_name.split('/')[-1]}",
            run_opts={"device": device},
        )
        self.device = device

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(waveforms).to(self.device)
            # SpeechBrain expects (batch, time)
            embs = self.model.encode_batch(t)  # (B, 1, D)
        return embs.squeeze(1).cpu().numpy()


class WavLMEmbedder:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv", device: str = "cuda"):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(device).eval()
        self.device = device

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        import torch
        # Reconstruct variable-length arrays by trimming trailing zeros added by collate_pad.
        # This restores proper attention_mask so WavLM doesn't treat padding as speech.
        trimmed = []
        for w in waveforms:
            nonzero = np.flatnonzero(w)
            end = int(nonzero[-1]) + 1 if len(nonzero) > 0 else 1
            trimmed.append(w[:end])
        inputs = self.extractor(
            trimmed,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.embeddings.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--data-root",  required=True)
    p.add_argument("--test-csv",   required=True)
    p.add_argument("--output",     default="submission_pretrained.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--cache-dir",  default=None)
    p.add_argument("--k",          type=int, default=10)
    p.add_argument("--no-gpu-index", action="store_true")
    p.add_argument("--limit",      type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"[run_pretrained] Loading: {args.model}")
    if "wavlm" in args.model.lower():
        model = WavLMEmbedder(args.model, device=device)
    else:
        model = SpeechBrainEmbedder(args.model, device=device)

    # File list
    df = pd.read_csv(args.test_csv)
    filepaths = df["filepath"].tolist()
    if args.limit:
        filepaths = filepaths[: args.limit]

    data_root = Path(args.data_root)
    print(f"[run_pretrained] Files: {len(filepaths)}, device: {device}")

    # Cache
    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = args.model.replace("/", "_")
        suffix = f"_limit{args.limit}" if args.limit else ""
        cache_path = cache_dir / f"{stem}_test{suffix}.npy"

    # WavLM has its own normalization inside the feature extractor
    normalize_audio = "wavlm" not in args.model.lower()

    # Extract
    embeddings = extract_embeddings(
        model=model,
        filepaths=filepaths,
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=normalize_audio,
        cache_path=cache_path,
    )
    print(f"[run_pretrained] Embeddings: {embeddings.shape}")

    # Retrieve
    index, normalized = build_index(embeddings, use_gpu=not args.no_gpu_index)
    neighbors = find_neighbors(index, normalized, k=args.k)

    # Save
    output_path = Path(args.output)
    save_submission(filepaths, neighbors, output_path, k=args.k)
    if args.limit is None:
        validate_submission(output_path, args.test_csv, k=args.k)

    print(f"[run_pretrained] Done in {time.time()-t0:.1f}s → {output_path}")


if __name__ == "__main__":
    main()
