"""Training loop with aggressive checkpointing for encoder training.

Saves intermediate checkpoints:
- Every N steps (configurable)
- Every epoch
- Best validation metric (early stopping)
- On interruption (KeyboardInterrupt / SIGTERM)

Checkpoint format:
    {stage}_{backbone}_{epoch}_{step}.pt
    {stage}_{backbone}_best.pt
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from hipert.training.dataset import CurriculumScheduler, ScoringDataset
from hipert.training.encoder import SymptomConditionedEncoder
from hipert.training.losses import CompositeLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training stage."""
    stage_name: str = "stage_a"
    backbone_name: str = "mpnet"

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Scheduler
    cosine_t0: int = 5  # restart period for cosine annealing

    # Checkpointing
    checkpoint_dir: Path = Path("output/training_checkpoints")
    checkpoint_every_steps: int = 500
    checkpoint_every_epochs: int = 1
    keep_last_n_checkpoints: int = 5

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Curriculum (Stage B only)
    use_curriculum: bool = False
    curriculum_p: float = 2.0
    curriculum_beta_scale: float = 5.0

    # Gradual unfreezing
    freeze_backbone_layers: int = 0
    unfreeze_at_epoch: int = 0

    # Loss config
    lambda_1: float = 1.0
    lambda_2: float = 0.5
    lambda_3: float = 0.05
    use_symmetric_ce: bool = False

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    # Logging
    log_every_steps: int = 50


@dataclass
class TrainingState:
    """Mutable training state for resumption."""
    epoch: int = 0
    global_step: int = 0
    best_val_metric: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    train_losses: list[dict] = field(default_factory=list)
    val_metrics: list[dict] = field(default_factory=list)


class Trainer:
    """Encoder trainer with intermediate checkpointing."""

    def __init__(
        self,
        model: SymptomConditionedEncoder,
        config: TrainingConfig,
        train_dataset: ScoringDataset,
        val_dataset: ScoringDataset | None = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.state = TrainingState()

        # Loss
        self.criterion = CompositeLoss(
            lambda_1=config.lambda_1,
            lambda_2=config.lambda_2,
            lambda_3=config.lambda_3,
            use_symmetric_ce=config.use_symmetric_ce,
        )

        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: warmup then cosine
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.cosine_t0,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps],
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if config.mixed_precision and config.device == "cuda" else None

        # Curriculum (optional)
        self.curriculum: CurriculumScheduler | None = None

        # Checkpoint management
        self._checkpoint_dir = config.checkpoint_dir / config.stage_name / config.backbone_name
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._saved_checkpoints: list[Path] = []

        # Interruption handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Save checkpoint on interruption."""
        logger.warning("Interrupt received! Saving emergency checkpoint...")
        self._interrupted = True
        self._save_checkpoint("interrupted")

    def _save_checkpoint(self, tag: str) -> Path:
        """Save a training checkpoint."""
        filename = f"{self.config.stage_name}_{self.config.backbone_name}_{tag}.pt"
        path = self._checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_metric": self.state.best_val_metric,
                "best_epoch": self.state.best_epoch,
                "patience_counter": self.state.patience_counter,
            },
            "config": {
                "stage_name": self.config.stage_name,
                "backbone_name": self.config.backbone_name,
                "learning_rate": self.config.learning_rate,
                "max_epochs": self.config.max_epochs,
            },
            "timestamp": time.time(),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (step=%d)", path.name, self.state.global_step)

        # Track and prune old checkpoints (keep best + last N)
        if tag not in ("best", "interrupted", "final"):
            self._saved_checkpoints.append(path)
            while len(self._saved_checkpoints) > self.config.keep_last_n_checkpoints:
                old = self._saved_checkpoints.pop(0)
                if old.exists():
                    old.unlink()
                    logger.debug("Pruned old checkpoint: %s", old.name)

        return path

    def resume_from_checkpoint(self, path: Path) -> None:
        """Resume training from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        state = checkpoint.get("state", {})
        self.state.epoch = state.get("epoch", 0)
        self.state.global_step = state.get("global_step", 0)
        self.state.best_val_metric = state.get("best_val_metric", float("inf"))
        self.state.best_epoch = state.get("best_epoch", 0)
        self.state.patience_counter = state.get("patience_counter", 0)

        logger.info(
            "Resumed from %s (epoch=%d, step=%d, best=%.4f)",
            path.name, self.state.epoch, self.state.global_step,
            self.state.best_val_metric,
        )

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Training summary with loss history and best metrics.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=ScoringDataset.collate_fn,
            num_workers=0,  # Windows compatibility
            pin_memory=self.config.device == "cuda",
        )

        # Setup curriculum if enabled
        if self.config.use_curriculum:
            total_steps = len(train_loader) * self.config.max_epochs
            self.curriculum = CurriculumScheduler(
                total_steps=total_steps,
                p=self.config.curriculum_p,
                beta_scale=self.config.curriculum_beta_scale,
            )
            logger.info("Curriculum enabled: %d total steps", total_steps)

        logger.info(
            "Training %s/%s: %d examples, %d epochs, lr=%.1e, batch=%d",
            self.config.stage_name, self.config.backbone_name,
            len(self.train_dataset), self.config.max_epochs,
            self.config.learning_rate, self.config.batch_size,
        )

        for epoch in range(self.state.epoch, self.config.max_epochs):
            if self._interrupted:
                break

            self.state.epoch = epoch

            # Gradual unfreezing
            if epoch == self.config.unfreeze_at_epoch and self.config.freeze_backbone_layers > 0:
                logger.info("Epoch %d: unfreezing all backbone layers", epoch)
                self.model.unfreeze_all()
                # Re-init optimizer with all params
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate * 0.1,
                    weight_decay=self.config.weight_decay,
                )

            # Train one epoch
            epoch_loss = self._train_epoch(train_loader, epoch)
            self.state.train_losses.append({
                "epoch": epoch, "loss": epoch_loss,
            })

            # Epoch checkpoint
            if (epoch + 1) % self.config.checkpoint_every_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}")

            # Validation
            if self.val_dataset is not None:
                val_metric = self._validate()
                self.state.val_metrics.append({
                    "epoch": epoch, "metric": val_metric,
                })
                logger.info(
                    "Epoch %d: train_loss=%.4f, val_metric=%.4f (best=%.4f)",
                    epoch, epoch_loss, val_metric, self.state.best_val_metric,
                )

                # Early stopping check
                if val_metric < self.state.best_val_metric - self.config.min_delta:
                    self.state.best_val_metric = val_metric
                    self.state.best_epoch = epoch
                    self.state.patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.state.patience_counter += 1
                    if self.state.patience_counter >= self.config.patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch, self.config.patience,
                        )
                        break
            else:
                logger.info("Epoch %d: train_loss=%.4f", epoch, epoch_loss)
                # Without validation, save best on lowest train loss
                if epoch_loss < self.state.best_val_metric:
                    self.state.best_val_metric = epoch_loss
                    self.state.best_epoch = epoch
                    self._save_checkpoint("best")

        # Final checkpoint
        self._save_checkpoint("final")

        # Save training log
        log_path = self._checkpoint_dir / "training_log.json"
        summary = {
            "stage": self.config.stage_name,
            "backbone": self.config.backbone_name,
            "total_epochs": self.state.epoch + 1,
            "total_steps": self.state.global_step,
            "best_epoch": self.state.best_epoch,
            "best_val_metric": self.state.best_val_metric,
            "train_losses": self.state.train_losses,
            "val_metrics": self.state.val_metrics,
        }
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Training complete: %d epochs, best at epoch %d (%.4f)",
            self.state.epoch + 1, self.state.best_epoch,
            self.state.best_val_metric,
        )
        return summary

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} [{self.config.stage_name}/{self.config.backbone_name}]",
            unit="batch",
        )

        for batch_idx, batch in enumerate(progress):
            if self._interrupted:
                break

            # Move to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.scaler is not None):
                logits = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    symptom_ids=batch["symptom_id"],
                    pre_input_ids=batch.get("pre_input_ids"),
                    pre_attention_mask=batch.get("pre_attention_mask"),
                    post_input_ids=batch.get("post_input_ids"),
                    post_attention_mask=batch.get("post_attention_mask"),
                )

                loss, components = self.criterion(
                    logits,
                    batch["label"],
                    self.model.symptom_embeddings,
                    weights=batch.get("weight"),
                )

                loss = loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1

                # Step checkpoint
                if self.state.global_step % self.config.checkpoint_every_steps == 0:
                    self._save_checkpoint(f"step_{self.state.global_step:06d}")

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Progress bar update
            if batch_idx % self.config.log_every_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    loss=f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    lr=f"{lr:.1e}",
                    step=self.state.global_step,
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation. Returns average loss (lower is better)."""
        if self.val_dataset is None:
            return float("inf")

        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=ScoringDataset.collate_fn,
            num_workers=0,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in val_loader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            logits = self.model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                symptom_ids=batch["symptom_id"],
                pre_input_ids=batch.get("pre_input_ids"),
                pre_attention_mask=batch.get("pre_attention_mask"),
                post_input_ids=batch.get("post_input_ids"),
                post_attention_mask=batch.get("post_attention_mask"),
            )

            loss, _ = self.criterion(
                logits, batch["label"],
                self.model.symptom_embeddings,
            )
            total_loss += loss.item() * batch["label"].size(0)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch["label"]).sum().item()
            total_samples += batch["label"].size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        logger.info("Validation: loss=%.4f, accuracy=%.4f", avg_loss, accuracy)
        return avg_loss
