#!/usr/bin/env python3
"""Fine-tune WavLM Large for speaker verification.

Stage1 (frozen backbone, train head only):
    bash scripts/run.sh scripts/run_train_wavlm.py \
        --config configs/train_wavlm_stage1.yaml

Stage2 (full fine-tune, gradient checkpointing):
    bash scripts/run.sh scripts/run_train_wavlm.py \
        --config configs/train_wavlm_stage2.yaml \
        --resume weights/wavlm_stage1_best.pt --reset-best
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import AugmentationPipeline
from src.models.wavlm_wrapper import WavLMWrapper
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

    noise_files, rir_files = [], []
    if aug_cfg.noise_dir:
        noise_dir = Path(aug_cfg.noise_dir)
        noise_files = list(noise_dir.rglob("*.flac")) + list(noise_dir.rglob("*.wav"))
        print(f"[run_train_wavlm] Noise files: {len(noise_files)}")
    if aug_cfg.rir_dir:
        rir_dir = Path(aug_cfg.rir_dir)
        rir_files = list(rir_dir.rglob("*.wav")) + list(rir_dir.rglob("*.flac"))
        print(f"[run_train_wavlm] RIR files: {len(rir_files)}")

    return AugmentationPipeline(
        sr=16000,
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--reset-best", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print(OmegaConf.to_yaml(cfg))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_train_wavlm] Device: {device}")

    train_df, val_df = make_val_split(
        Path(cfg.data.train_csv),
        n_val_speakers=cfg.data.n_val_speakers,
        seed=cfg.data.seed,
    )
    n_train_speakers = train_df["speaker_id"].nunique()
    print(f"[run_train_wavlm] Train speakers: {n_train_speakers}, "
          f"train files: {len(train_df)}, val files: {len(val_df)}")

    print("[run_train_wavlm] Loading WavLM Large...")
    encoder = WavLMWrapper(
        emb_dim=cfg.model.emb_dim,
        freeze_backbone=cfg.model.freeze_backbone,
        use_grad_checkpointing=cfg.model.use_grad_checkpointing,
    ).to(device)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
    total = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"[run_train_wavlm] Params: {total:.1f}M total, {trainable:.1f}M trainable")

    if cfg.loss.type == "subcenter_arcface":
        loss_fn = SubcenterArcFace(
            emb_dim=cfg.model.emb_dim,
            n_classes=n_train_speakers,
            K=cfg.loss.subcenter_k,
            margin=cfg.loss.margin,
            scale=cfg.loss.scale,
        )
    else:
        loss_fn = AAMSoftmax(
            emb_dim=cfg.model.emb_dim,
            n_classes=n_train_speakers,
            margin=cfg.loss.margin,
            scale=cfg.loss.scale,
        )

    augment = build_augmentation(cfg)
    print(f"[run_train_wavlm] Augmentation: {'enabled' if augment else 'DISABLED'}")

    save_name = cfg.checkpoint.get("save_name", "wavlm_best.pt")
    grad_accum = cfg.training.get("grad_accumulation_steps", 1)

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
            "checkpoint_name": save_name,
            "grad_accumulation_steps": grad_accum,
        },
        device=device,
        experiment_name=cfg.experiment_name,
    )

    if args.resume:
        if not args.resume_from:
            parser.error("--resume requires --resume-from <path>")
        trainer.resume_from_checkpoint(args.resume_from)
        print(f"[run_train_wavlm] Resumed from: {args.resume_from}")

    if args.reset_best:
        trainer.best_p10 = 0.0
        print("[run_train_wavlm] best_p10 reset to 0.0")

    trainer.fit()
    print(f"\n[run_train_wavlm] Best checkpoint: {cfg.checkpoint.dir}{save_name}")


if __name__ == "__main__":
    main()
