"""Calibrated inference + ensemble scoring for Run 1 and Run 4.

Produces per-symptom ranked sentence lists using:
    phi(s, q) = sum(r * p_cal(r | s, q)) for r in {0,1,2,3}
    phi_ensemble = (1/3) * [phi_model1 + phi_model2 + phi_model3]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from hipert.training.calibration import CalibrationPipeline
from hipert.training.dataset import ScoringDataset, ScoringExample
from hipert.training.encoder import BACKBONES, SymptomConditionedEncoder

logger = logging.getLogger(__name__)

# Default ensemble members
ENSEMBLE_BACKBONES = ["mental-roberta", "clinical-bert", "mpnet"]


def score_sentences(
    model: SymptomConditionedEncoder,
    calibration: CalibrationPipeline,
    examples: list[ScoringExample],
    tokenizer: AutoTokenizer,
    batch_size: int = 64,
    device: str = "cpu",
) -> list[tuple[str, int, float]]:
    """Score a list of examples, returning (docno, symptom_id, score) tuples.

    Score = expected relevance: sum(r * p_cal(r)) for r in {0,1,2,3}.
    """
    model.eval()
    model.to(device)

    dataset = ScoringDataset(examples, tokenizer, include_context=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ScoringDataset.collate_fn,
    )

    results = []
    idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                symptom_ids=batch["symptom_id"],
                pre_input_ids=batch.get("pre_input_ids"),
                pre_attention_mask=batch.get("pre_attention_mask"),
                post_input_ids=batch.get("post_input_ids"),
                post_attention_mask=batch.get("post_attention_mask"),
            )

            # Calibrated expected score
            scores = calibration.expected_score(
                logits.cpu(), batch["symptom_id"].cpu(),
            )

            for i in range(scores.size(0)):
                ex = examples[idx]
                results.append((
                    ex.text,  # docno stored via the pipeline
                    ex.symptom_id,
                    scores[i].item(),
                ))
                idx += 1

    return results


def run_inference(
    checkpoint_dir: Path,
    candidates_dir: Path,
    output_dir: Path,
    backbone_name: str = "mpnet",
    stage: str = "stage_b",
    batch_size: int = 64,
    device: str | None = None,
    top_n: int = 1000,
) -> None:
    """Run inference for a single backbone model.

    Reads candidates from output/candidates/symptom_{id}.json,
    scores them, and writes results to output/{scores_subdir}/symptom_{id}.json.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = BACKBONES.get(backbone_name, backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model from best checkpoint
    ckpt_path = checkpoint_dir / stage / backbone_name / f"{stage}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = SymptomConditionedEncoder.load_checkpoint(
        ckpt_path, backbone_name=backbone_name,
        num_symptoms=18 if stage == "stage_b" else 21,
    )

    # Load calibration
    cal_dir = checkpoint_dir / stage / backbone_name / "calibration"
    calibration = CalibrationPipeline(
        num_symptoms=18 if stage == "stage_b" else 21,
    )
    if cal_dir.exists():
        calibration.load(cal_dir)
        logger.info("Loaded calibration from %s", cal_dir)
    else:
        logger.warning("No calibration found at %s, using uncalibrated", cal_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for symptom_id in range(1, 19):
        cand_path = candidates_dir / f"symptom_{symptom_id}.json"
        if not cand_path.exists():
            continue

        with open(cand_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Convert to ScoringExamples
        examples = [
            ScoringExample(
                text=c["text"],
                pre=c.get("pre", ""),
                post=c.get("post", ""),
                symptom_id=symptom_id,
                label=0,  # unused for inference
            )
            for c in candidates
        ]

        if not examples:
            continue

        # Score
        dataset = ScoringDataset(examples, tokenizer, include_context=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ScoringDataset.collate_fn,
        )

        model.eval()
        model.to(device)
        scores = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    symptom_ids=batch["symptom_id"],
                    pre_input_ids=batch.get("pre_input_ids"),
                    pre_attention_mask=batch.get("pre_attention_mask"),
                    post_input_ids=batch.get("post_input_ids"),
                    post_attention_mask=batch.get("post_attention_mask"),
                )
                batch_scores = calibration.expected_score(
                    logits.cpu(), batch["symptom_id"].cpu(),
                )
                scores.extend(batch_scores.tolist())

        # Combine with docnos and sort
        scored = [
            {"docno": candidates[i]["docno"], "score": scores[i]}
            for i in range(len(scores))
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = scored[:top_n]

        # Write
        out_path = output_dir / f"symptom_{symptom_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)

        logger.info(
            "Symptom %d: scored %d, top score=%.4f",
            symptom_id, len(scored),
            scored[0]["score"] if scored else 0.0,
        )


def run_ensemble_inference(
    checkpoint_dir: Path,
    candidates_dir: Path,
    output_dir: Path,
    backbones: list[str] | None = None,
    stage: str = "stage_b",
    batch_size: int = 64,
    device: str | None = None,
    top_n: int = 1000,
) -> None:
    """Run ensemble inference across multiple backbones.

    phi_ensemble(s,q) = (1/K) * sum(phi_k(s,q)) for k in backbones
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if backbones is None:
        backbones = ENSEMBLE_BACKBONES

    logger.info("Ensemble inference with %d backbones: %s", len(backbones), backbones)

    # Score per backbone into separate dirs
    backbone_dirs = []
    for bb in backbones:
        bb_dir = output_dir / f"_single_{bb}"
        try:
            run_inference(
                checkpoint_dir=checkpoint_dir,
                candidates_dir=candidates_dir,
                output_dir=bb_dir,
                backbone_name=bb,
                stage=stage,
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

        # Average and rank
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
                "Symptom %d ensemble: %d sentences (from %d backbones), top=%.4f",
                symptom_id, len(averaged), len(backbone_dirs),
                averaged[0]["score"],
            )
