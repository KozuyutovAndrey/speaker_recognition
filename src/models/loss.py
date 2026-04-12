"""Metric learning losses for speaker verification.

AAM-Softmax (Additive Angular Margin Softmax / ArcFace):
  - Standard loss for speaker verification training
  - Adds angular margin m to the correct class angle
  - Makes embeddings of the same speaker cluster tightly
  - SubcenterArcFace: K sub-centers per class for handling noise/variation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax (ArcFace).

    Args:
        emb_dim:    Speaker embedding dimension
        n_classes:  Number of training speakers
        margin:     Angular margin in radians (default 0.2 ≈ 11.5°)
        scale:      Logit scale (default 30)
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # threshold to avoid acos instability
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized (B, emb_dim)
            labels:     (B,) long tensor with class indices

        Returns:
            Scalar cross-entropy loss
        """
        # Normalize weight vectors
        w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, w)  # (B, n_classes)
        sine = (1.0 - cosine.pow(2)).clamp(min=0).sqrt()

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Avoid gradient explosion when θ + m > π
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only on the correct class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


class SubcenterArcFace(nn.Module):
    """Subcenter ArcFace: K sub-centers per class.

    Better handles intra-class variance (e.g., clean vs noisy speech
    from the same speaker).

    Args:
        emb_dim:    Embedding dimension
        n_classes:  Number of speakers
        K:          Number of sub-centers per speaker (default 3)
        margin:     Angular margin
        scale:      Logit scale
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        K: int = 3,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super().__init__()
        self.K = K
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(n_classes * K, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)  # (n_classes*K, D)

        # Cosine to all sub-centers
        cosine_all = F.linear(embeddings, w)  # (B, n_classes*K)
        cosine_all = cosine_all.view(-1, cosine_all.shape[1] // self.K, self.K)
        # Take max sub-center per class
        cosine, _ = cosine_all.max(dim=-1)  # (B, n_classes)

        sine = (1.0 - cosine.pow(2)).clamp(min=0).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)
