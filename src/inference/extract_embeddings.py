"""Batched embedding extraction with caching and I/O prefetching."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.dataset import FlacDataset, collate_pad


def extract_embeddings(
    model,
    filepaths: list[str],
    data_root: Path,
    target_sr: int = 16000,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    max_duration_s: float | None = 10.0,
    vad_fn=None,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Extract speaker embeddings for a list of audio files.

    Args:
        model:       Any object with .embed_batch(waveforms: ndarray) -> ndarray
        filepaths:   List of relative file paths (as in CSV)
        data_root:   Root directory; full path = data_root / filepath
        batch_size:  Number of files per GPU batch
        num_workers: Threads for parallel audio loading
        cache_path:  If provided, load from / save to this .npy file

    Returns:
        embeddings: float32 array of shape (N, D)
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"[extract_embeddings] Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    dataset = FlacDataset(
        filepaths=filepaths,
        data_root=data_root,
        target_sr=target_sr,
        normalize=normalize,
        max_duration_s=max_duration_s,
        vad_fn=vad_fn,
    )
    n = len(dataset)

    # Pre-allocate output array (filled after first batch to know dim D)
    all_embeddings: np.ndarray | None = None
    ptr = 0

    def load_item(idx: int) -> tuple[np.ndarray, str]:
        try:
            return dataset[idx]
        except Exception as e:
            print(f"[WARNING] Failed to load {filepaths[idx]}: {e}")
            return np.zeros(target_sr, dtype=np.float32), filepaths[idx]

    indices = list(range(n))
    batches = [indices[i : i + batch_size] for i in range(0, n, batch_size)]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for batch_idx in tqdm(batches, desc="Extracting embeddings", unit="batch"):
            # Load audio in parallel threads
            items = list(pool.map(load_item, batch_idx))
            waveforms, _ = collate_pad(items)

            embs = model.embed_batch(waveforms)  # (B, D)

            if all_embeddings is None:
                D = embs.shape[1]
                all_embeddings = np.zeros((n, D), dtype=np.float32)

            all_embeddings[ptr : ptr + len(embs)] = embs
            ptr += len(embs)

    if all_embeddings is None:
        raise RuntimeError("No files were processed.")

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, all_embeddings)
        print(f"[extract_embeddings] Saved embeddings to {cache_path}")

    return all_embeddings
