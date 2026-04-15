"""CAM++ wrapper for fine-tuning via our standard training pipeline.

Wraps damo/speech_campplus_sv_en_voxceleb_16k (ModelScope) to match
the same interface as ECAPA-TDNN: forward(waveforms [B, T]) → embeddings [B, 512].

Uses GPU-batched torchaudio MelSpectrogram instead of sequential Kaldi.fbank
for 10-30x faster feature extraction during training. The embedding_model
adapts to the new frontend during fine-tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class CAMPlusWrapper(nn.Module):
    """Fine-tunable CAM++ encoder with GPU-batched frontend.

    Args:
        freeze_frontend: If True, gradients don't flow through MelSpectrogram.
        emb_dim:         Output embedding dim (512 for CAM++ voxceleb).
        sample_rate:     Audio sample rate (default 16000).
    """

    def __init__(
        self,
        freeze_frontend: bool = True,
        emb_dim: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__()

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        pipe = pipeline(
            task=Tasks.speaker_verification,
            model="damo/speech_campplus_sv_en_voxceleb_16k",
            model_revision="v1.0.2",
        )
        campplus_model = pipe.model  # SpeakerVerificationCAMPPlus

        self.embedding_model = campplus_model.embedding_model
        self.n_mels = campplus_model.feature_dim  # 80
        self.emb_dim = emb_dim
        self.freeze_frontend = freeze_frontend

        # GPU-batched MelSpectrogram frontend.
        # Matches Kaldi defaults: 25ms window, 10ms hop, 80 mels.
        # Will be moved to GPU when .to(device) is called on this module.
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
        """Inference interface matching ECAPA-TDNN.embed_batch."""
        self.eval()
        device = next(self.embedding_model.parameters()).device
        with torch.no_grad():
            t = torch.from_numpy(waveforms).to(device)
            embs = self.forward(t)
        return embs.cpu().numpy()

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Inference interface matching ECAPA-TDNN.embed_batch."""
        self.eval()
        with torch.no_grad():
            device = next(self.embedding_model.parameters()).device
            t = torch.from_numpy(waveforms).to(device)
            embs = self.forward(t)
        return embs.cpu().numpy()
