"""Stage A: Depression pre-training on BDI-Sen + eRisk 2025 T1.

Two sub-stages:
- A1: BDI-Sen graded pre-training (2,529 sentences, 5,003 pairs, graded 0-3)
- A2: eRisk 2025 T1 binary pre-training (11,042 judgments, binary 0/1)

Uses BDI-II symptom IDs (1-21) during training, mapped to ASRS later.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hipert.data.bdisen_loader import load_annotations as load_bdisen
from hipert.data.cross_dataset_mappings import BDISEN_TO_ASRS
from hipert.data.erisk2025_loader import load_qrels as load_erisk2025_qrels
from hipert.training.calibration import CalibrationPipeline
from hipert.training.dataset import ScoringDataset, ScoringExample
from hipert.training.encoder import BACKBONES, SymptomConditionedEncoder
from hipert.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger(__name__)


def _load_bdisen_examples(bdisen_dir: Path) -> list[ScoringExample]:
    """Load BDI-Sen 2.0 as ScoringExamples (graded 0-3, BDI-II symptom IDs)."""
    bdisen_file = Path(bdisen_dir) / "bdi_majority_vote.jsonl"
    if not bdisen_file.exists():
        # Caller passed the file directly instead of the directory
        bdisen_file = Path(bdisen_dir)
    annotations = load_bdisen(bdisen_file)
    examples = []

    for ann in annotations:
        # BDI-Sen uses 0-3 graded labels and BDI-II symptom IDs
        examples.append(ScoringExample(
            text=ann.sentence,
            pre="",
            post="",
            symptom_id=ann.symptom_id,
            label=ann.relevance,
            weight=1.0,  # gold annotations
            difficulty=0.3,  # medium-easy (well-curated)
        ))

    logger.info("Loaded %d BDI-Sen examples", len(examples))
    return examples


def _load_erisk2025_examples(
    erisk2025_dir: Path,
    trec_dir: Path,
) -> list[ScoringExample]:
    """Load eRisk 2025 T1 as ScoringExamples (binary, BDI-II IDs)."""
    qrels = load_erisk2025_qrels(erisk2025_dir)
    examples = []

    # eRisk 2025 uses majority/consensus binary labels
    for qrel in qrels:
        label = min(qrel.relevance, 1)  # clamp to binary 0/1
        examples.append(ScoringExample(
            text=qrel.sentence_text or "",
            pre=qrel.pre_text or "",
            post=qrel.post_text or "",
            symptom_id=qrel.query_id,
            label=label,
            weight=1.0 if qrel.consensus else 0.8,
            difficulty=0.4,
        ))

    logger.info("Loaded %d eRisk 2025 T1 examples", len(examples))
    return examples


def train_stage_a(
    bdisen_dir: Path,
    erisk2025_dir: Path | None = None,
    erisk2025_trec_dir: Path | None = None,
    backbone_name: str = "mpnet",
    checkpoint_dir: Path = Path("output/training_checkpoints"),
    max_epochs_a1: int = 10,
    max_epochs_a2: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    device: str | None = None,
    resume_from: Path | None = None,
) -> Path:
    """Run Stage A depression pre-training.

    Returns:
        Path to the best checkpoint for use in Stage B or Run 4.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = BACKBONES.get(backbone_name, backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # ---- Stage A1: BDI-Sen ----
    logger.info("=== Stage A1: BDI-Sen pre-training (%s) ===", backbone_name)

    bdisen_examples = _load_bdisen_examples(bdisen_dir)
    if not bdisen_examples:
        raise RuntimeError(f"No BDI-Sen examples found in {bdisen_dir}")

    # Train/val split (90/10)
    split_idx = int(len(bdisen_examples) * 0.9)
    train_examples = bdisen_examples[:split_idx]
    val_examples = bdisen_examples[split_idx:]

    # BDI-Sen uses symptom IDs 1-21, need num_symptoms=21
    model = SymptomConditionedEncoder(
        backbone_name=backbone_name,
        num_symptoms=21,  # BDI-II has 21 symptoms
        freeze_backbone_layers=6,  # freeze early layers initially
    )

    train_ds = ScoringDataset(train_examples, tokenizer, include_context=False)
    val_ds = ScoringDataset(val_examples, tokenizer, include_context=False)

    config_a1 = TrainingConfig(
        stage_name="stage_a1",
        backbone_name=backbone_name,
        learning_rate=learning_rate,
        max_epochs=max_epochs_a1,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=200,
        checkpoint_every_epochs=1,
        freeze_backbone_layers=6,
        unfreeze_at_epoch=3,
        patience=3,
        device=device,
    )

    trainer = Trainer(model, config_a1, train_ds, val_ds)
    if resume_from is not None:
        trainer.resume_from_checkpoint(resume_from)

    summary_a1 = trainer.train()
    best_a1 = checkpoint_dir / "stage_a1" / backbone_name / "stage_a1_best.pt"
    logger.info("Stage A1 complete: best at epoch %d", summary_a1["best_epoch"])

    # ---- Stage A2: eRisk 2025 T1 (optional) ----
    if erisk2025_dir is not None:
        logger.info("=== Stage A2: eRisk 2025 T1 pre-training (%s) ===", backbone_name)

        erisk_examples = _load_erisk2025_examples(
            erisk2025_dir, erisk2025_trec_dir or erisk2025_dir,
        )

        if erisk_examples:
            split_idx = int(len(erisk_examples) * 0.9)
            train_erisk = erisk_examples[:split_idx]
            val_erisk = erisk_examples[split_idx:]

            # Load from Stage A1 best checkpoint
            model_a2 = SymptomConditionedEncoder(
                backbone_name=backbone_name,
                num_symptoms=21,
            )
            model_a2 = SymptomConditionedEncoder.load_checkpoint(
                best_a1, backbone_name=backbone_name, num_symptoms=21,
            )

            train_ds_a2 = ScoringDataset(
                train_erisk, tokenizer, include_context=True,
            )
            val_ds_a2 = ScoringDataset(
                val_erisk, tokenizer, include_context=True,
            )

            config_a2 = TrainingConfig(
                stage_name="stage_a2",
                backbone_name=backbone_name,
                learning_rate=learning_rate * 0.5,
                max_epochs=max_epochs_a2,
                batch_size=batch_size,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every_steps=500,
                checkpoint_every_epochs=1,
                patience=2,
                device=device,
            )

            trainer_a2 = Trainer(model_a2, config_a2, train_ds_a2, val_ds_a2)
            summary_a2 = trainer_a2.train()
            best_path = checkpoint_dir / "stage_a2" / backbone_name / "stage_a2_best.pt"
            logger.info("Stage A2 complete: best at epoch %d", summary_a2["best_epoch"])
            return best_path

    return best_a1
