#!/usr/bin/env python3
"""Train ECAPA-TDNN speaker encoder with AAM-Softmax.

Usage:
    python scripts/run_train.py --config configs/train_ecapa.yaml [overrides...]

    # Example overrides:
    python scripts/run_train.py --config configs/train_ecapa.yaml \\
        training.epochs=50 training.batch_size=128 model.channels=1024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import AugmentationPipeline
from src.models.ecapa_tdnn import ECAPA_TDNN
from src.models.loss import AAMSoftmax, SubcenterArcFace
from src.training.trainer import Trainer


def make_val_split(train_csv: Path, n_val_speakers: int, seed: int):
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)
    speakers = sorted(df["speaker_id"].unique())
    val_speakers = set(rng.choice(speakers, size=n_val_speakers, replace=False))
    val_df = df[df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
    train_df = df[~df["speaker_id"].isin(val_speakers)].reset_index(drop=True)
    return train_df, val_df


def build_augmentation(cfg) -> AugmentationPipeline | None:
    aug_cfg = cfg.augmentation
    if not aug_cfg.enabled:
        return None

    noise_files = []
    rir_files = []

    if aug_cfg.noise_dir:
        noise_dir = Path(aug_cfg.noise_dir)
        noise_files = list(noise_dir.rglob("*.flac")) + list(noise_dir.rglob("*.wav"))
        print(f"[run_train] Noise files: {len(noise_files)}")

    if aug_cfg.rir_dir:
        rir_dir = Path(aug_cfg.rir_dir)
        rir_files = list(rir_dir.rglob("*.wav")) + list(rir_dir.rglob("*.flac"))
        print(f"[run_train] RIR files: {len(rir_files)}")

    return AugmentationPipeline(
        sr=cfg.model.sr,
        p_white_noise=aug_cfg.p_white_noise,
        p_pink_noise=aug_cfg.p_pink_noise,
        p_file_noise=aug_cfg.p_file_noise,
        p_rir=aug_cfg.p_rir,
        p_telephone=aug_cfg.p_telephone,
        p_codec=aug_cfg.p_codec,
        p_speed=aug_cfg.p_speed,
        p_volume=aug_cfg.p_volume,
        noise_snr_range=(aug_cfg.noise_snr_min, aug_cfg.noise_snr_max),
        noise_files=noise_files,
        rir_files=rir_files,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--reset-best", action="store_true",
                        help="Reset best P@10 to 0 after resume (use when loss/augmentation changed)")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-notation overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print(OmegaConf.to_yaml(cfg))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_train] Device: {device}")

    # Data split
    train_df, val_df = make_val_split(
        Path(cfg.data.train_csv),
        n_val_speakers=cfg.data.n_val_speakers,
        seed=cfg.data.seed,
    )
    n_train_speakers = train_df["speaker_id"].nunique()
    print(f"[run_train] Train speakers: {n_train_speakers}, "
          f"train files: {len(train_df)}, val files: {len(val_df)}")

    # Model
    encoder = ECAPA_TDNN(
        n_mels=cfg.model.n_mels,
        channels=cfg.model.channels,
        emb_dim=cfg.model.emb_dim,
        scale=cfg.model.scale,
        sr=cfg.model.sr,
    )
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"[run_train] Encoder params: {n_params:.1f}M")

    # Loss
    if cfg.loss.type == "aam_softmax":
        loss_fn = AAMSoftmax(
            emb_dim=cfg.model.emb_dim,
            n_classes=n_train_speakers,
            margin=cfg.loss.margin,
            scale=cfg.loss.scale,
        )
    elif cfg.loss.type == "subcenter_arcface":
        loss_fn = SubcenterArcFace(
            emb_dim=cfg.model.emb_dim,
            n_classes=n_train_speakers,
            K=cfg.loss.subcenter_k,
            margin=cfg.loss.margin,
            scale=cfg.loss.scale,
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss.type}")

    # Augmentation
    augment = build_augmentation(cfg)
    if augment:
        print("[run_train] Augmentation enabled")
    else:
        print("[run_train] Augmentation DISABLED")

    # Train
    trainer = Trainer(
        encoder=encoder,
        loss_fn=loss_fn,
        train_df=train_df,
        val_df=val_df,
        data_root=Path(cfg.data.data_root),
        augment=augment,
        config={
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "batch_size": cfg.training.batch_size,
            "num_workers": cfg.training.num_workers,
            "epochs": cfg.training.epochs,
            "chunk_s": cfg.training.chunk_s,
            "warmup_epochs": cfg.training.warmup_epochs,
            "grad_clip": cfg.training.grad_clip,
            "target_sr": cfg.training.target_sr,
            "checkpoint_dir": cfg.checkpoint.dir,
            "log_file": cfg.checkpoint.log_file,
            "bf16": cfg.training.get("bf16", True),
        },
        device=device,
        experiment_name=cfg.experiment_name,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
        if args.reset_best:
            trainer.best_p10 = 0.0
            print("[run_train] best_p10 reset to 0.0")

    trainer.fit()

    print(f"\n[run_train] Best checkpoint saved to: {cfg.checkpoint.dir}")
    print(f"[run_train] To run inference with trained model, use scripts/run_inference_torch.py")


if __name__ == "__main__":
    main()
