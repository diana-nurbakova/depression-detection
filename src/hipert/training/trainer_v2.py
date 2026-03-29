"""Cross-encoder trainer for HiPerT v2.

Single-stage distillation from LLM silver labels with:
- CORAL ordinal regression or ListMLE listwise ranking
- Leave-symptom-out 5-fold cross-validation
- Score spread diagnostic to detect representation collapse
- NDCG@10 and P@10 validation metrics

Spec reference: hipert_v2_spec.md Sections 6-7
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hipert.training.cross_encoder import BACKBONES, CrossEncoderReranker
from hipert.training.cross_encoder_dataset import (
    CrossEncoderDataset,
    SublistSampler,
    create_cv_splits,
)
from hipert.training.losses import CORALLoss, ListMLELoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerV2Config:
    """Configuration for cross-encoder v2 training."""

    # Model
    backbone_name: str = "mpnet"
    head_type: str = "coral"  # "coral" or "listmle"
    num_unfrozen: int = 4
    hidden_dim: int = 256
    dropout: float = 0.3

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_epochs: int = 20
    batch_size: int = 64
    max_grad_norm: float = 1.0

    # CORAL-specific
    threshold_weights: list[float] | None = None  # e.g., [1.0, 1.5, 2.0]
    use_confidence_weighting: bool = True

    # ListMLE-specific
    listmle_temperature: float = 1.0
    sublist_size: int = 64
    num_sublists_per_batch: int = 8
    steps_per_epoch: int = 200

    # Scheduler
    lr_min_ratio: float = 0.1  # eta_min = lr * ratio

    # Tokenizer
    max_length: int = 256

    # Checkpointing
    checkpoint_dir: Path = Path("output/training_v2")
    patience: int = 3

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    # Cross-validation
    num_folds: int = 5


@dataclass
class TrainingMetrics:
    """Collected metrics from training."""

    fold: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    ndcg_at_10: float = 0.0
    p_at_10: float = 0.0
    mean_cv: float = 0.0
    mean_gap: float = 0.0


def compute_ndcg(ranked_labels: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k from ranked label array."""
    ranked_labels = ranked_labels[:k]
    if len(ranked_labels) == 0:
        return 0.0

    # DCG
    gains = 2.0 ** ranked_labels - 1.0
    discounts = np.log2(np.arange(len(ranked_labels)) + 2.0)
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal = np.sort(ranked_labels)[::-1]
    ideal_gains = 2.0 ** ideal - 1.0
    ideal_dcg = np.sum(ideal_gains / discounts)

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


class TrainerV2:
    """Cross-encoder trainer with leave-symptom-out CV."""

    def __init__(self, config: TrainerV2Config):
        self.config = config
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logger.warning("Interrupt received — finishing current epoch...")
        self._interrupted = True

    def train_fold(
        self,
        train_data: list[dict],
        val_data: list[dict],
        fold: int,
    ) -> tuple[CrossEncoderReranker, list[TrainingMetrics]]:
        """Train one fold of cross-validation.

        Returns:
            (best_model, metrics_history)
        """
        config = self.config

        # Create model
        model = CrossEncoderReranker(
            backbone_name=config.backbone_name,
            head_type=config.head_type,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            num_unfrozen=config.num_unfrozen,
        )
        model = model.to(config.device)

        tokenizer = model.tokenizer

        # Create loss
        if config.head_type == "coral":
            criterion = CORALLoss(
                num_classes=4,
                threshold_weights=config.threshold_weights,
            )
        else:
            criterion = ListMLELoss(temperature=config.listmle_temperature)

        # Optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * config.lr_min_ratio,
        )

        # Mixed precision
        scaler = None
        if config.mixed_precision and config.device == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        # Checkpoint dir for this fold
        fold_dir = (
            config.checkpoint_dir / config.head_type
            / config.backbone_name / f"fold_{fold}"
        )
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Training data
        if config.head_type == "coral":
            train_dataset = CrossEncoderDataset(train_data, tokenizer, config.max_length)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=CrossEncoderDataset.collate_fn,
                num_workers=0,
                pin_memory=config.device == "cuda",
            )

        best_ndcg = -1.0
        patience_counter = 0
        best_state = None
        metrics_history = []

        for epoch in range(config.max_epochs):
            if self._interrupted:
                break

            model.train()

            if config.head_type == "coral":
                epoch_loss = self._train_epoch_coral(
                    model, train_loader, criterion, optimizer, scaler, epoch,
                )
            else:
                epoch_loss = self._train_epoch_listmle(
                    model, train_data, tokenizer, criterion, optimizer, scaler, epoch,
                )

            scheduler.step()

            # Validation
            val_metrics = self._evaluate(model, val_data, tokenizer)
            ndcg = val_metrics["ndcg@10"]
            p10 = val_metrics["p@10"]

            metrics = TrainingMetrics(
                fold=fold,
                epoch=epoch,
                train_loss=epoch_loss,
                ndcg_at_10=ndcg,
                p_at_10=p10,
                mean_cv=val_metrics["mean_cv"],
                mean_gap=val_metrics["mean_gap"],
            )
            metrics_history.append(metrics)

            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Fold %d Epoch %d: loss=%.4f NDCG@10=%.4f P@10=%.4f CV=%.4f lr=%.1e",
                fold, epoch, epoch_loss, ndcg, p10, val_metrics["mean_cv"], lr,
            )

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                # Save checkpoint
                ckpt_path = fold_dir / "best.pt"
                model.save_checkpoint(ckpt_path, extra={
                    "fold": fold,
                    "epoch": epoch,
                    "ndcg@10": ndcg,
                    "p@10": p10,
                })
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
                    break

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(config.device)

        # Save metrics
        metrics_path = fold_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                [{"fold": m.fold, "epoch": m.epoch, "train_loss": m.train_loss,
                  "ndcg@10": m.ndcg_at_10, "p@10": m.p_at_10,
                  "mean_cv": m.mean_cv, "mean_gap": m.mean_gap}
                 for m in metrics_history],
                f, indent=2,
            )

        return model, metrics_history

    def _train_epoch_coral(
        self,
        model: CrossEncoderReranker,
        loader: DataLoader,
        criterion: CORALLoss,
        optimizer: AdamW,
        scaler: torch.amp.GradScaler | None,
        epoch: int,
    ) -> float:
        """Train one epoch with CORAL loss."""
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"CORAL epoch {epoch}", unit="batch"):
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            confidence = batch["confidence"].to(self.config.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.config.device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits = model(input_ids, attention_mask, token_type_ids)
                conf_w = confidence if self.config.use_confidence_weighting else None
                loss = criterion(logits, labels, conf_w)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _train_epoch_listmle(
        self,
        model: CrossEncoderReranker,
        train_data: list[dict],
        tokenizer: AutoTokenizer,
        criterion: ListMLELoss,
        optimizer: AdamW,
        scaler: torch.amp.GradScaler | None,
        epoch: int,
    ) -> float:
        """Train one epoch with ListMLE loss using sublist sampling."""
        sampler = SublistSampler(
            train_data,
            sublist_size=self.config.sublist_size,
            num_sublists_per_batch=self.config.num_sublists_per_batch,
        )

        total_loss = 0.0
        n_steps = 0

        for step in tqdm(range(self.config.steps_per_epoch), desc=f"ListMLE epoch {epoch}"):
            if self._interrupted:
                break

            batch_data = sampler.sample_batch(epoch * self.config.steps_per_epoch + step)
            if not batch_data:
                continue

            dataset = CrossEncoderDataset(batch_data, tokenizer, self.config.max_length)
            batch = CrossEncoderDataset.collate_fn([dataset[i] for i in range(len(dataset))])

            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            symptom_ids = batch["symptom_id"].to(self.config.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.config.device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                scores = model.predict_score(input_ids, attention_mask, token_type_ids)
                loss = criterion(scores, labels, symptom_ids)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: CrossEncoderReranker,
        val_data: list[dict],
        tokenizer: AutoTokenizer,
    ) -> dict[str, float]:
        """Evaluate on validation data.

        Returns: dict with ndcg@10, p@10, mean_cv, mean_gap.
        """
        model.eval()

        # Group by symptom
        by_symptom: dict[int, list[dict]] = defaultdict(list)
        for d in val_data:
            by_symptom[d["symptom_id"]].append(d)

        all_ndcg = []
        all_p10 = []
        all_cv = []
        all_gap = []

        for sid, entries in sorted(by_symptom.items()):
            if len(entries) < 10:
                continue

            dataset = CrossEncoderDataset(entries, tokenizer, self.config.max_length)
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                collate_fn=CrossEncoderDataset.collate_fn,
            )

            scores_list = []
            for batch in loader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.config.device)

                batch_scores = model.predict_score(input_ids, attention_mask, token_type_ids)
                scores_list.append(batch_scores.cpu().numpy())

            scores = np.concatenate(scores_list)
            labels = np.array([e["score"] for e in entries])

            # Rank by predicted score descending
            ranking = np.argsort(-scores)
            ranked_labels = labels[ranking]

            # NDCG@10
            ndcg = compute_ndcg(ranked_labels, k=10)
            all_ndcg.append(ndcg)

            # P@10 (score >= 2 is relevant)
            p10 = float((ranked_labels[:10] >= 2).mean())
            all_p10.append(p10)

            # Score spread
            cv = float(np.std(scores) / (np.abs(np.mean(scores)) + 1e-8))
            all_cv.append(cv)

            gap = float(np.mean(scores[ranking[:10]]) - np.mean(scores[ranking[10:]]))
            all_gap.append(gap)

        model.train()

        return {
            "ndcg@10": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
            "p@10": float(np.mean(all_p10)) if all_p10 else 0.0,
            "mean_cv": float(np.mean(all_cv)) if all_cv else 0.0,
            "mean_gap": float(np.mean(all_gap)) if all_gap else 0.0,
        }

    def train_all_folds(
        self,
        all_data: list[dict],
    ) -> dict:
        """Train across all CV folds.

        Returns:
            Summary dict with per-fold metrics and aggregate stats.
        """
        config = self.config
        fold_results = []

        for fold in range(1, config.num_folds + 1):
            if self._interrupted:
                break

            logger.info(
                "=" * 60 + "\n"
                "Training fold %d/%d [%s / %s]\n" + "=" * 60,
                fold, config.num_folds, config.head_type, config.backbone_name,
            )

            train_data, val_data = create_cv_splits(all_data, fold)
            model, metrics = self.train_fold(train_data, val_data, fold)

            best = max(metrics, key=lambda m: m.ndcg_at_10)
            fold_results.append({
                "fold": fold,
                "best_epoch": best.epoch,
                "best_ndcg@10": best.ndcg_at_10,
                "best_p@10": best.p_at_10,
                "final_cv": best.mean_cv,
                "final_gap": best.mean_gap,
            })

            logger.info(
                "Fold %d best: NDCG@10=%.4f P@10=%.4f (epoch %d)",
                fold, best.ndcg_at_10, best.p_at_10, best.epoch,
            )

        # Aggregate
        if fold_results:
            mean_ndcg = np.mean([r["best_ndcg@10"] for r in fold_results])
            std_ndcg = np.std([r["best_ndcg@10"] for r in fold_results])
            mean_p10 = np.mean([r["best_p@10"] for r in fold_results])
            mean_cv = np.mean([r["final_cv"] for r in fold_results])
        else:
            mean_ndcg = std_ndcg = mean_p10 = mean_cv = 0.0

        summary = {
            "head_type": config.head_type,
            "backbone": config.backbone_name,
            "folds": fold_results,
            "mean_ndcg@10": float(mean_ndcg),
            "std_ndcg@10": float(std_ndcg),
            "mean_p@10": float(mean_p10),
            "mean_cv": float(mean_cv),
        }

        # Save summary
        summary_dir = config.checkpoint_dir / config.head_type / config.backbone_name
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "CV complete [%s/%s]: NDCG@10=%.4f±%.4f P@10=%.4f CV=%.4f",
            config.head_type, config.backbone_name,
            mean_ndcg, std_ndcg, mean_p10, mean_cv,
        )

        return summary


def diagnose_score_spread(
    model: CrossEncoderReranker,
    data: list[dict],
    tokenizer: AutoTokenizer,
    batch_size: int = 128,
    device: str = "cpu",
    threshold_cv: float = 0.05,
) -> dict:
    """Check whether the trained model has collapsed.

    FAIL if:
    - Mean CV across symptoms < threshold_cv
    - Comparison against BiEnc baseline (CV ≈ 0.10)

    Spec reference: hipert_v2_spec.md Section 6.5

    Returns:
        Per-symptom diagnostic report.
    """
    model.eval()
    model.to(device)

    by_symptom: dict[int, list[dict]] = defaultdict(list)
    for d in data:
        by_symptom[d["symptom_id"]].append(d)

    report: dict[int, dict] = {}

    with torch.no_grad():
        for sid, entries in sorted(by_symptom.items()):
            dataset = CrossEncoderDataset(entries, tokenizer, max_length=256)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=CrossEncoderDataset.collate_fn,
            )

            scores_list = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                batch_scores = model.predict_score(input_ids, attention_mask, token_type_ids)
                scores_list.append(batch_scores.cpu().numpy())

            scores = np.concatenate(scores_list)
            sorted_scores = np.sort(scores)[::-1]

            top10_mean = float(sorted_scores[:10].mean())
            rest_mean = float(sorted_scores[10:].mean())
            std = float(scores.std())
            cv = float(std / (np.abs(scores.mean()) + 1e-8))

            report[sid] = {
                "top10_mean": top10_mean,
                "rest_mean": rest_mean,
                "gap": top10_mean - rest_mean,
                "std": std,
                "cv": cv,
                "healthy": cv > threshold_cv,
            }

    mean_cv = np.mean([r["cv"] for r in report.values()])
    n_collapsed = sum(1 for r in report.values() if not r["healthy"])

    logger.info("Score spread diagnostic:")
    logger.info("  Mean CV: %.4f (v1 was 0.01; BiEnc baseline ~0.10)", mean_cv)
    logger.info("  Collapsed symptoms: %d/%d", n_collapsed, len(report))

    if mean_cv < threshold_cv:
        logger.warning("*** WARNING: Model has collapsed. Do not submit. ***")

    for sid, r in sorted(report.items()):
        status = "OK" if r["healthy"] else "COLLAPSED"
        logger.info(
            "  Symptom %2d: top10=%.4f rest=%.4f gap=%.4f CV=%.4f [%s]",
            sid, r["top10_mean"], r["rest_mean"], r["gap"], r["cv"], status,
        )

    return {
        "per_symptom": report,
        "mean_cv": float(mean_cv),
        "n_collapsed": n_collapsed,
        "healthy": mean_cv > threshold_cv,
    }


def select_best_variant(
    coral_summary: dict,
    listmle_summary: dict,
) -> str:
    """Select between CORAL and ListMLE based on CV results.

    Spec reference: hipert_v2_spec.md Section 7.2

    Returns:
        'coral' or 'listmle'
    """
    coral_cv = coral_summary.get("mean_cv", 0.0)
    listmle_cv = listmle_summary.get("mean_cv", 0.0)
    coral_ndcg = coral_summary.get("mean_ndcg@10", 0.0)
    listmle_ndcg = listmle_summary.get("mean_ndcg@10", 0.0)

    coral_healthy = coral_cv > 0.05
    listmle_healthy = listmle_cv > 0.05

    if coral_healthy and not listmle_healthy:
        logger.info("Selected CORAL (ListMLE collapsed)")
        return "coral"
    if listmle_healthy and not coral_healthy:
        logger.info("Selected ListMLE (CORAL collapsed)")
        return "listmle"
    if not coral_healthy and not listmle_healthy:
        logger.error("Both variants collapsed!")
        # Return whichever has higher CV
        return "coral" if coral_cv >= listmle_cv else "listmle"

    # Both healthy: pick by NDCG@10
    if coral_ndcg > listmle_ndcg + 0.01:
        logger.info("Selected CORAL (NDCG@10: %.4f vs %.4f)", coral_ndcg, listmle_ndcg)
        return "coral"
    elif listmle_ndcg > coral_ndcg + 0.01:
        logger.info("Selected ListMLE (NDCG@10: %.4f vs %.4f)", listmle_ndcg, coral_ndcg)
        return "listmle"
    else:
        logger.info("Within 0.01 — defaulting to CORAL (calibrated scores)")
        return "coral"
