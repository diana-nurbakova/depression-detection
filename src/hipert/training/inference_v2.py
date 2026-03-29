"""Cross-encoder v2 inference: ensemble of backbones × folds → TREC output.

Scoring function:
    CORAL:   φ(s,q) = (1/B)(1/F) Σ_backbone Σ_fold [σ(f₁)+σ(f₂)+σ(f₃)]
    ListMLE: φ(s,q) = (1/B)(1/F) Σ_backbone Σ_fold f(enc([q;s]))

Spec reference: hipert_v2_spec.md Section 7
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hipert.training.cross_encoder import BACKBONES, CrossEncoderReranker
from hipert.training.cross_encoder_dataset import CrossEncoderDataset
from hipert.training.extract_training_data import ASRS_ITEMS

logger = logging.getLogger(__name__)

ENSEMBLE_BACKBONES = ["mental-roberta", "clinical-bert", "mpnet"]


def score_with_model(
    model: CrossEncoderReranker,
    candidates: list[dict],
    symptom_id: int,
    tokenizer: AutoTokenizer,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Score candidates for one symptom using a single model.

    Returns:
        numpy array of scores, one per candidate.
    """
    symptom_text = ASRS_ITEMS[symptom_id]

    # Build data for dataset
    data_items = [
        {
            "symptom_id": symptom_id,
            "symptom_text": symptom_text,
            "sentence_text": c["text"],
            "score": 0,
            "confidence": 1.0,
        }
        for c in candidates
    ]

    dataset = CrossEncoderDataset(data_items, tokenizer, max_length=256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CrossEncoderDataset.collate_fn,
    )

    model.eval()
    model.to(device)
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            scores = model.predict_score(input_ids, attention_mask, token_type_ids)
            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


def run_v2_inference(
    checkpoint_dir: Path,
    candidates_dir: Path,
    output_dir: Path,
    head_type: str = "coral",
    backbone_name: str = "mpnet",
    num_folds: int = 5,
    batch_size: int = 128,
    device: str | None = None,
    top_n: int = 1000,
) -> None:
    """Run inference for a single backbone, averaging across folds.

    Reads candidates from output/candidates/symptom_{id}.json,
    scores them, and writes to output_dir/symptom_{id}.json.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = BACKBONES.get(backbone_name, backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load fold models
    fold_models = []
    for fold in range(1, num_folds + 1):
        ckpt_path = (
            checkpoint_dir / head_type / backbone_name
            / f"fold_{fold}" / "best.pt"
        )
        if not ckpt_path.exists():
            logger.warning("Missing checkpoint for fold %d: %s", fold, ckpt_path)
            continue

        model = CrossEncoderReranker.load_checkpoint(
            ckpt_path, backbone_name=backbone_name, head_type=head_type,
        )
        fold_models.append(model)

    if not fold_models:
        raise FileNotFoundError(
            f"No fold checkpoints found at {checkpoint_dir / head_type / backbone_name}"
        )

    logger.info(
        "Loaded %d fold models for %s/%s", len(fold_models), head_type, backbone_name,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for symptom_id in range(1, 19):
        cand_path = candidates_dir / f"symptom_{symptom_id}.json"
        if not cand_path.exists():
            logger.warning("No candidates for symptom %d", symptom_id)
            continue

        with open(cand_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Score with each fold and average
        fold_scores = []
        for model in fold_models:
            scores = score_with_model(
                model, candidates, symptom_id, tokenizer, batch_size, device,
            )
            fold_scores.append(scores)

        mean_scores = np.mean(fold_scores, axis=0)

        # Build ranking
        scored = [
            {"docno": candidates[i]["docno"], "score": float(mean_scores[i])}
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = scored[:top_n]

        out_path = output_dir / f"symptom_{symptom_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)

        logger.info(
            "Symptom %d: scored %d candidates, top=%.4f",
            symptom_id, len(scored), scored[0]["score"] if scored else 0.0,
        )


def run_v2_ensemble_inference(
    checkpoint_dir: Path,
    candidates_dir: Path,
    output_dir: Path,
    head_type: str = "coral",
    backbones: list[str] | None = None,
    num_folds: int = 5,
    batch_size: int = 128,
    device: str | None = None,
    top_n: int = 1000,
) -> None:
    """Run ensemble inference across multiple backbones × folds.

    φ_ensemble(s,q) = (1/B) Σ_backbone φ_backbone(s,q)
    where φ_backbone = (1/F) Σ_fold φ_fold(s,q)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if backbones is None:
        backbones = ENSEMBLE_BACKBONES

    logger.info("Ensemble v2 inference: %s with %d backbones", head_type, len(backbones))

    # Score per backbone into separate dirs
    backbone_dirs = []
    for bb in backbones:
        bb_dir = output_dir / f"_single_{bb}"
        try:
            run_v2_inference(
                checkpoint_dir=checkpoint_dir,
                candidates_dir=candidates_dir,
                output_dir=bb_dir,
                head_type=head_type,
                backbone_name=bb,
                num_folds=num_folds,
                batch_size=batch_size,
                device=device,
                top_n=top_n * 2,  # wider pool for ensemble
            )
            backbone_dirs.append(bb_dir)
        except FileNotFoundError as e:
            logger.warning("Skipping backbone %s: %s", bb, e)

    if not backbone_dirs:
        raise RuntimeError("No backbone models available for ensemble")

    # Average scores across backbones
    output_dir.mkdir(parents=True, exist_ok=True)

    for symptom_id in range(1, 19):
        score_accumulator: dict[str, list[float]] = {}

        for bb_dir in backbone_dirs:
            score_path = bb_dir / f"symptom_{symptom_id}.json"
            if not score_path.exists():
                continue

            with open(score_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                docno = item["docno"]
                if docno not in score_accumulator:
                    score_accumulator[docno] = []
                score_accumulator[docno].append(item["score"])

        averaged = [
            {"docno": docno, "score": sum(scores) / len(scores)}
            for docno, scores in score_accumulator.items()
        ]
        averaged.sort(key=lambda x: x["score"], reverse=True)
        averaged = averaged[:top_n]

        out_path = output_dir / f"symptom_{symptom_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(averaged, f, indent=2)

        if averaged:
            logger.info(
                "Symptom %d ensemble: %d sentences (%d backbones), top=%.4f",
                symptom_id, len(averaged), len(backbone_dirs),
                averaged[0]["score"],
            )


def write_trec_run(
    scores_dir: Path,
    system_name: str,
    output_path: Path,
    top_n: int = 1000,
) -> None:
    """Write TREC-formatted run file from scored JSON files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for symptom_id in range(1, 19):
            scores_path = scores_dir / f"symptom_{symptom_id}.json"
            if not scores_path.exists():
                logger.warning("No scores for symptom %d", symptom_id)
                continue

            with open(scores_path, "r", encoding="utf-8") as sf:
                data = json.load(sf)

            for rank_idx, item in enumerate(data[:top_n]):
                position = f"{rank_idx + 1:04d}"
                f.write(
                    f"{symptom_id}\tQ0\t{item['docno']}\t{position}"
                    f"\t{item['score']:.6f}\t{system_name}\n"
                )

    logger.info("TREC run written: %s", output_path)
