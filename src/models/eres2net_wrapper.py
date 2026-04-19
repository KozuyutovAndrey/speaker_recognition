"""ERes2Net wrapper for fine-tuning via our standard training pipeline.

Wraps damo/speech_eres2net_sv_en_voxceleb_16k (ModelScope) to match
the same interface as CAMPlusWrapper: forward(waveforms [B, T]) → embeddings [B, 192].

Uses GPU-batched torchaudio MelSpectrogram instead of sequential Kaldi.fbank
(which returns only 1 embedding per batch call regardless of batch size).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class ERes2NetWrapper(nn.Module):
    """Fine-tunable ERes2Net encoder with GPU-batched frontend.

    Args:
        freeze_frontend: If True, gradients don't flow through MelSpectrogram.
        emb_dim:         Output embedding dim (192 for ERes2Net voxceleb).
        sample_rate:     Audio sample rate (default 16000).
    """

    def __init__(
        self,
        freeze_frontend: bool = True,
        emb_dim: int = 192,
        sample_rate: int = 16000,
    ):
        super().__init__()

        import os
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model_id = "damo/speech_eres2net_sv_en_voxceleb_16k"
        cache_root = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
        local_path = os.path.join(cache_root, "models", model_id)
        if os.path.exists(local_path):
            pipe = pipeline(task=Tasks.speaker_verification, model=local_path)
        else:
            pipe = pipeline(task=Tasks.speaker_verification, model=model_id, model_revision="master")
        eres2net_model = pipe.model

        self.embedding_model = eres2net_model.embedding_model
        self.n_mels = eres2net_model.feature_dim  # 80
        self.emb_dim = emb_dim
        self.freeze_frontend = freeze_frontend

        # GPU-batched MelSpectrogram frontend.
        # Matches Kaldi defaults: 25ms window, 10ms hop, 80 mels.
        self.frontend = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,   # 25 ms at 16kHz
            hop_length=160,   # 10 ms at 16kHz
            n_mels=self.n_mels,
            center=False,
        )

        self.embedding_model.train()

    def _compute_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """GPU batched log-mel features: [B, T] → [B, T', n_mels]."""
        specs = self.frontend(waveforms)          # [B, n_mels, T']
        specs = torch.log(specs + 1e-6)          # log scale
        specs = specs.transpose(1, 2)            # [B, T', n_mels]
        specs = specs - specs.mean(dim=1, keepdim=True)  # per-sample mean norm
        return specs

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [B, T] float32 raw audio at 16kHz
        Returns:
            embeddings: [B, emb_dim] L2-normalized
        """
        if self.freeze_frontend:
            with torch.no_grad():
                features = self._compute_features(waveforms)
        else:
            features = self._compute_features(waveforms)

        embeddings = self.embedding_model(features)  # [B, emb_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Inference interface matching CAMPlusWrapper.embed_batch."""
        self.eval()
        with torch.no_grad():
            device = next(self.embedding_model.parameters()).device
            t = torch.from_numpy(waveforms).to(device)
            embs = self.forward(t)
        return embs.cpu().numpy()
