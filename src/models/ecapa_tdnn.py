"""ECAPA-TDNN speaker encoder.

Architecture from:
  "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
   in TDNN Based Speaker Verification" (Desplanques et al., 2020)

This implementation is self-contained (no SpeechBrain dependency) and
designed for training from scratch or fine-tuning.

Input : raw waveform float32 (B, T) at 16 kHz
Output: L2-normalized speaker embedding (B, emb_dim)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# Frontend: log-mel spectrogram
# ---------------------------------------------------------------------------

class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,   # 25 ms
        hop_length: int = 160,   # 10 ms
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.log_offset = 1e-6

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) → mel: (B, n_mels, T')"""
        mel = self.mel(waveform)
        return torch.log(mel + self.log_offset)


# ---------------------------------------------------------------------------
# SE (Squeeze-Excitation) block
# ---------------------------------------------------------------------------

class SE(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        scale = self.se(x).unsqueeze(-1)
        return x * scale


# ---------------------------------------------------------------------------
# Res2 dilated convolution block (TDNN with skip connections)
# ---------------------------------------------------------------------------

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, scale: int = 8):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(
                width, width,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
            )
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spx = torch.split(x, self.width, dim=1)
        out = []
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0:
                sp = spx[0]
            else:
                sp = sp + spx[i]
            sp = F.relu(bn(conv(sp)))
            out.append(sp)
        out.append(spx[-1])
        return torch.cat(out, dim=1)


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.res2conv = Res2Conv1dReluBn(channels, kernel_size, dilation, scale)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res2conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x + residual


# ---------------------------------------------------------------------------
# Attentive Statistics Pooling
# ---------------------------------------------------------------------------

class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling: computes attended mean + std."""

    def __init__(self, in_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim * 3, attention_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, in_dim, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = x.std(dim=-1, keepdim=True).expand_as(x)
        attn_input = torch.cat([x, mean, std], dim=1)  # (B, 3C, T)
        w = self.attention(attn_input)  # (B, C, T)
        mu = (w * x).sum(dim=-1)
        sigma = (w * (x - mu.unsqueeze(-1)) ** 2).sum(dim=-1).clamp(min=1e-9).sqrt()
        return torch.cat([mu, sigma], dim=1)  # (B, 2C)


# ---------------------------------------------------------------------------
# ECAPA-TDNN
# ---------------------------------------------------------------------------

class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN speaker encoder.

    Args:
        n_mels:   Number of mel filterbanks
        channels: Number of channels in TDNN blocks (512 or 1024)
        emb_dim:  Speaker embedding dimension
        scale:    Res2 scale factor
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 512,
        emb_dim: int = 192,
        scale: int = 8,
        sr: int = 16000,
    ):
        super().__init__()
        self.frontend = LogMelFrontend(sr=sr, n_mels=n_mels)

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.layer1 = SERes2Block(channels, kernel_size=3, dilation=2, scale=scale)
        self.layer2 = SERes2Block(channels, kernel_size=3, dilation=3, scale=scale)
        self.layer3 = SERes2Block(channels, kernel_size=3, dilation=4, scale=scale)

        # Multi-scale feature aggregation
        self.cat_conv = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, 1),
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(),
        )

        self.pool = AttentiveStatsPool(channels * 3)

        self.bn_after_pool = nn.BatchNorm1d(channels * 6)
        self.fc = nn.Linear(channels * 6, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) → embedding: (B, emb_dim), L2-normalized"""
        x = self.frontend(waveform)  # (B, n_mels, T')
        x = self.conv1(x)            # (B, C, T')
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.cat_conv(torch.cat([x1, x2, x3], dim=1))  # (B, 3C, T')
        x = self.pool(x)             # (B, 6C)
        x = self.bn_after_pool(x)
        x = self.fc(x)               # (B, emb_dim)
        x = self.bn_emb(x)
        return F.normalize(x, p=2, dim=1)

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """Alias for forward; used during inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(waveform)
