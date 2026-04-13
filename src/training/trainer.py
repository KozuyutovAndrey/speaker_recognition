"""Training loop for speaker encoder.

Trains ECAPA-TDNN (or any encoder) with AAM-Softmax on the competition
train data. Evaluates on val split via Precision@K after each epoch.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.train_dataset import SpeakerTrainDataset, collate_train
from src.data.augmentation import AugmentationPipeline
from src.retrieval.faiss_search import build_index, find_neighbors, l2_normalize
from src.utils.metrics import precision_at_k_report
from src.utils.experiment_logger import ExperimentResult, log_experiment


class Trainer:
    """Speaker encoder trainer.

    Args:
        encoder:      The ECAPA-TDNN (or other) model
        loss_fn:      AAMSoftmax or SubcenterArcFace
        train_df:     DataFrame [speaker_id, filepath] for training
        val_df:       DataFrame [speaker_id, filepath] for validation
        data_root:    Root dir for audio files
        config:       Dict with training hyperparameters (see default below)
        device:       'cuda' or 'cpu'
        experiment_name: Name for logging
    """

    DEFAULT_CONFIG = {
        "lr": 1e-3,
        "weight_decay": 2e-5,
        "batch_size": 256,
        "num_workers": 8,
        "epochs": 20,
        "chunk_s": 3.0,
        "target_sr": 16000,
        "warmup_epochs": 2,
        "checkpoint_dir": "weights/",
        "log_file": "results/experiments.jsonl",
        "val_batch_size": 128,
        "grad_clip": 5.0,
    }

    def __init__(
        self,
        encoder: nn.Module,
        loss_fn: nn.Module,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        data_root: Path,
        augment: AugmentationPipeline | None = None,
        config: dict | None = None,
        device: str = "cuda",
        experiment_name: str = "ecapa_tdnn",
    ):
        self.encoder = encoder.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_df = train_df
        self.val_df = val_df
        self.data_root = Path(data_root)
        self.augment = augment
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = device
        self.experiment_name = experiment_name

        self._setup_dataloaders()
        self._setup_optimizer()

        self.best_p10 = 0.0
        self.start_epoch = 1
        self.checkpoint_dir = Path(self.cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load encoder + optimizer + scheduler state to continue training."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])

        # Try loading loss weights (fails if loss type or n_classes changed)
        loss_ok = True
        try:
            self.loss_fn.load_state_dict(ckpt["loss_state"])
        except RuntimeError as e:
            print(f"[Trainer] WARNING: loss state mismatch ({e}). Reinitializing loss + optimizer.")
            loss_ok = False

        # Load optimizer only if loss is compatible; otherwise reinitialize
        if loss_ok:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
                if "scheduler_state" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except RuntimeError as e:
                print(f"[Trainer] WARNING: optimizer state mismatch ({e}). Reinitializing optimizer.")
        else:
            print("[Trainer] Optimizer reinitialized (new loss function).")
        self.best_p10 = ckpt["metrics"].get("P@10", 0.0)
        resumed_epoch = ckpt.get("epoch", 0)
        self.start_epoch = resumed_epoch + 1
        # Extend total epochs: run cfg["epochs"] MORE epochs from checkpoint
        self.cfg["epochs"] = resumed_epoch + self.cfg["epochs"]
        print(f"[Trainer] Resumed from epoch {resumed_epoch}, "
              f"best P@10={self.best_p10:.4f}, "
              f"will train epochs {self.start_epoch}→{self.cfg['epochs']}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_dataloaders(self):
        train_dataset = SpeakerTrainDataset(
            df=self.train_df,
            data_root=self.data_root,
            target_sr=self.cfg["target_sr"],
            chunk_s=self.cfg["chunk_s"],
            augment=self.augment,
            normalize=True,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            collate_fn=collate_train,
            drop_last=True,
            persistent_workers=True,
        )

        val_dataset = SpeakerTrainDataset(
            df=self.val_df,
            data_root=self.data_root,
            target_sr=self.cfg["target_sr"],
            chunk_s=self.cfg["chunk_s"],
            augment=None,  # No augmentation at eval time
            normalize=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg["val_batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            collate_fn=collate_train,
        )
        self.val_labels = self.val_df["speaker_id"].values

    def _setup_optimizer(self):
        params = list(self.encoder.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        # Cosine annealing with linear warmup
        total_steps = self.cfg["epochs"] * len(self.train_loader)
        warmup_steps = self.cfg["warmup_epochs"] * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # BF16 GradScaler — используем только для FP16, BF16 не требует скейлера
        # но GradScaler с enabled=False оставляем для унификации кода
        use_bf16 = self.cfg.get("bf16", True) and self.device == "cuda"
        self.autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.use_autocast = use_bf16
        if use_bf16:
            print("[Trainer] BF16 autocast enabled (Ampere+)")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.encoder.train()
        self.loss_fn.train()
        total_loss = 0.0

        bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for waveforms, labels in bar:
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_autocast):
                embeddings = self.encoder(waveforms)
                loss = self.loss_fn(embeddings, labels)
            loss.backward()

            if self.cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.loss_fn.parameters()),
                    self.cfg["grad_clip"],
                )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Extract val embeddings and compute Precision@K."""
        self.encoder.eval()
        all_embs = []

        for waveforms, _ in tqdm(self.val_loader, desc="Val embeddings", leave=False):
            waveforms = waveforms.to(self.device)
            embs = self.encoder(waveforms).cpu().numpy()
            all_embs.append(embs)

        embeddings = np.concatenate(all_embs, axis=0)
        index, normalized = build_index(embeddings, use_gpu=(self.device == "cuda"))
        neighbors = find_neighbors(index, normalized, k=10)
        return precision_at_k_report(neighbors, self.val_labels, ks=[1, 5, 10])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        print(f"[Trainer] Starting '{self.experiment_name}': "
              f"{self.cfg['epochs']} epochs, device={self.device}")
        t0 = time.time()

        for epoch in range(self.start_epoch, self.cfg["epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()

            p10 = metrics["P@10"]
            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"P@1={metrics['P@1']:.4f} P@5={metrics['P@5']:.4f} P@10={p10:.4f}"
            )

            # Save best checkpoint
            if p10 > self.best_p10:
                self.best_p10 = p10
                self._save_checkpoint(epoch, metrics)
                print(f"  → New best P@10={p10:.4f}, checkpoint saved")

        elapsed = time.time() - t0
        print(f"[Trainer] Done in {elapsed/60:.1f}min, best P@10={self.best_p10:.4f}")

        log_experiment(ExperimentResult(
            name=self.experiment_name,
            model="ECAPA-TDNN",
            config=self.cfg,
            precision_at_1=self.best_p10,   # placeholder, real values at best epoch
            precision_at_5=self.best_p10,
            precision_at_10=self.best_p10,
            inference_time_sec=elapsed,
        ))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        torch.save({
            "epoch": epoch,
            "encoder_state": self.encoder.state_dict(),
            "loss_state": self.loss_fn.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
        }, path)

    def load_best_checkpoint(self) -> dict:
        path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        print(f"[Trainer] Loaded checkpoint: epoch={ckpt['epoch']}, metrics={ckpt['metrics']}")
        return ckpt
