"""WavLM Large wrapper for speaker verification fine-tuning.

Uses microsoft/wavlm-large (316M params, pretrained on 94k hours).
Forward: waveform [B, T] → L2-normalized embedding [B, emb_dim].

Architecture:
  WavLM Large → weighted sum of 25 transformer layers → AttentiveStatsPool → Linear → L2-norm

Supports:
  - freeze_backbone: train only head (stage1, fast convergence)
  - gradient checkpointing: reduces VRAM ~40% for full fine-tune (stage2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling: [B, T, D] → [B, 2*D]."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        w = self.attn(x).squeeze(-1)          # [B, T]
        w = torch.softmax(w, dim=1).unsqueeze(-1)  # [B, T, 1]
        mean = (w * x).sum(dim=1)             # [B, D]
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)  # [B, 2*D]


class WavLMWrapper(nn.Module):
    """Fine-tunable WavLM Large encoder for speaker verification.

    Args:
        emb_dim:              Output embedding dimension (default 192).
        freeze_backbone:      If True, WavLM weights are frozen (stage1).
        use_grad_checkpointing: Enable gradient checkpointing to save VRAM (stage2).
    """

    WAVLM_DIM = 1024  # hidden size of WavLM Large

    def __init__(
        self,
        emb_dim: int = 192,
        freeze_backbone: bool = False,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")

        if use_grad_checkpointing:
            self.wavlm.gradient_checkpointing_enable()

        if freeze_backbone:
            for p in self.wavlm.parameters():
                p.requires_grad = False

        # Learnable weighted sum over 25 transformer layers
        n_layers = self.wavlm.config.num_hidden_layers + 1  # +1 for CNN features
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

        self.pool = AttentiveStatsPool(self.WAVLM_DIM)
        self.bn = nn.BatchNorm1d(self.WAVLM_DIM * 2)
        self.proj = nn.Linear(self.WAVLM_DIM * 2, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [B, T] float32 raw audio at 16kHz, NOT normalized
        Returns:
            embeddings: [B, emb_dim] L2-normalized
        """
        # WavLM expects normalized input
        # Normalize per sample to zero mean unit variance
        mean = waveforms.mean(dim=1, keepdim=True)
        std = waveforms.std(dim=1, keepdim=True).clamp(min=1e-8)
        waveforms = (waveforms - mean) / std

        out = self.wavlm(
            waveforms,
            output_hidden_states=True,
            return_dict=True,
        )

        # Weighted sum of all hidden states: list of [B, T', 1024]
        hidden_states = torch.stack(out.hidden_states, dim=1)  # [B, n_layers, T', 1024]
        weights = torch.softmax(self.layer_weights, dim=0)      # [n_layers]
        x = (hidden_states * weights.view(1, -1, 1, 1)).sum(dim=1)  # [B, T', 1024]

        x = self.pool(x)    # [B, 2048]
        x = self.bn(x)
        x = self.proj(x)    # [B, emb_dim]
        x = F.normalize(x, p=2, dim=1)
        return x

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        """Inference interface matching other wrappers."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.from_numpy(waveforms).to(device)
            embs = self.forward(t)
        return embs.cpu().numpy()
