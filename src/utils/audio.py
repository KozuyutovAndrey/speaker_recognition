"""Audio loading and preprocessing utilities."""

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    max_duration_s: float | None = 10.0,
) -> np.ndarray:
    """Load a FLAC/WAV file, convert to float32 mono, resample if needed.

    Args:
        max_duration_s: Crop to this many seconds from the middle of the file.
                        None = no limit.

    Returns:
        waveform: float32 numpy array of shape (T,)
    """
    path = str(path)
    waveform, sr = sf.read(path, dtype="float32", always_2d=False)

    # Convert stereo to mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")

    # Crop to max duration (take center segment)
    if max_duration_s is not None:
        max_samples = int(max_duration_s * target_sr)
        if len(waveform) > max_samples:
            start = (len(waveform) - max_samples) // 2
            waveform = waveform[start : start + max_samples]

    return waveform


def normalize_amplitude(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak normalization to [-1, 1]."""
    peak = np.abs(waveform).max()
    return waveform / (peak + eps)


def apply_vad_energy(
    waveform: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
    min_duration_s: float = 0.5,
) -> np.ndarray:
    """Remove silent frames using energy-based VAD.

    If the result is shorter than min_duration_s, returns the original waveform.
    """
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(waveform) // frame_len
    if n_frames == 0:
        return waveform

    frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    rms_db = 20 * np.log10(rms + 1e-9)

    mask = rms_db > threshold_db
    if mask.sum() == 0:
        return waveform

    speech_frames = frames[mask]
    result = speech_frames.flatten()

    if len(result) < int(sr * min_duration_s):
        return waveform

    return result
