"""Stage B: ADHD silver-label fine-tuning with curriculum learning.

Takes the Stage A depression-pretrained encoder and fine-tunes on
LLM-generated silver labels with confidence weighting and curriculum scheduling.

Curriculum phases:
    Phase 1 (c < 0.35): Motor H/I + Verbal H/I (easiest)
    Phase 2 (0.35-0.65): Organization/Memory
    Phase 3 (c >= 0.65): Sustained Attention (hardest, most confounders)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hipert.training.calibration import CalibrationPipeline
from hipert.training.dataset import ScoringDataset, load_silver_labels
from hipert.training.encoder import BACKBONES, SymptomConditionedEncoder
from hipert.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def train_stage_b(
    silver_labels_dir: Path,
    stage_a_checkpoint: Path,
    backbone_name: str = "mpnet",
    checkpoint_dir: Path = Path("output/training_checkpoints"),
    max_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    device: str | None = None,
    resume_from: Path | None = None,
    calibration_every: int = 3,
    min_confidence: float = 0.3,
) -> Path:
    """Run Stage B ADHD fine-tuning with curriculum.

    Args:
        silver_labels_dir: Directory with symptom_{id}.jsonl files.
        stage_a_checkpoint: Path to Stage A best checkpoint.
        backbone_name: Which backbone to fine-tune.
        checkpoint_dir: Where to save checkpoints.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Lower than Stage A (fine-tuning).
        device: cuda or cpu.
        resume_from: Checkpoint to resume from.
        calibration_every: Re-calibrate every N epochs.
        min_confidence: Minimum confidence_weight for including examples.

    Returns:
        Path to best checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=== Stage B: ADHD fine-tuning (%s) ===", backbone_name)

    model_id = BACKBONES.get(backbone_name, backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load silver labels
    all_examples = load_silver_labels(
        silver_labels_dir, min_confidence=min_confidence,
    )
    if not all_examples:
        raise RuntimeError(f"No silver labels found in {silver_labels_dir}")

    logger.info("Silver labels: %d examples", len(all_examples))

    # Train/val split (90/10, stratified by symptom)
    from collections import defaultdict
    by_symptom = defaultdict(list)
    for ex in all_examples:
        by_symptom[ex.symptom_id].append(ex)

    train_examples = []
    val_examples = []
    for sid, exs in by_symptom.items():
        split = int(len(exs) * 0.9)
        train_examples.extend(exs[:split])
        val_examples.extend(exs[split:])

    logger.info(
        "Split: %d train, %d val across %d symptoms",
        len(train_examples), len(val_examples), len(by_symptom),
    )

    # Load Stage A checkpoint
    model = SymptomConditionedEncoder.load_checkpoint(
        stage_a_checkpoint,
        backbone_name=backbone_name,
        num_symptoms=18,  # Now ASRS symptoms
    )

    train_ds = ScoringDataset(train_examples, tokenizer, include_context=True)
    val_ds = ScoringDataset(val_examples, tokenizer, include_context=True)

    config = TrainingConfig(
        stage_name="stage_b",
        backbone_name=backbone_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=300,
        checkpoint_every_epochs=1,
        keep_last_n_checkpoints=5,
        patience=5,
        use_curriculum=True,
        curriculum_p=2.0,
        curriculum_beta_scale=5.0,
        use_symmetric_ce=True,  # noise-robust loss for silver labels
        device=device,
    )

    trainer = Trainer(model, config, train_ds, val_ds)
    if resume_from is not None:
        trainer.resume_from_checkpoint(resume_from)

    summary = trainer.train()
    best_path = checkpoint_dir / "stage_b" / backbone_name / f"stage_b_{backbone_name}_best.pt"

    # Post-training calibration
    logger.info("Fitting calibration on validation set...")
    calibration = CalibrationPipeline(num_symptoms=18)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=ScoringDataset.collate_fn,
    )
    calibration.fit(model, val_loader, device=device)
    cal_dir = checkpoint_dir / "stage_b" / backbone_name / "calibration"
    calibration.save(cal_dir)
    logger.info("Calibration saved to %s", cal_dir)

    logger.info(
        "Stage B complete: best at epoch %d (%.4f)",
        summary["best_epoch"], summary["best_val_metric"],
    )
    return best_path
