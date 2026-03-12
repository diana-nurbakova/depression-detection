"""Run 1: HiPerT_full — Full HiPerT-ADHD pipeline.

Complete system: bi-encoder retrieval → LLM cascade → silver labels →
encoder training (Stages A→B→C) → calibrated inference → ensemble.

Scoring function:
    φ(s, q) = Σ_{r=0}^{3} r · p̂_cal(r | s, q)
    φ_ensemble = (1/3) · [φ_MentalRoBERTa + φ_ClinicalBERT + φ_mpnet]

Status: STUB — requires encoder training infrastructure (Stages A, B, C).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)

# Path where trained encoder scores would be saved
ENCODER_SCORES_DIR = "output/encoder_scores"


@register_run(1)
def generate_hipert_full(config: PipelineConfig) -> Rankings:
    """Generate Run 1 rankings from trained encoder scores.

    Looks for pre-computed encoder scores at:
        output/encoder_scores/symptom_{id}.json

    Each file: list of {"docno": str, "score": float} sorted by score desc.

    If encoder scores are not available, raises RuntimeError.
    """
    scores_dir = config.output_dir / "encoder_scores"
    if not scores_dir.exists():
        raise RuntimeError(
            f"Encoder scores not found at {scores_dir}. "
            "Run encoder training (Stages A→B→C) and inference first."
        )

    rankings: Rankings = {}
    found = 0

    for symptom_id in range(1, 19):
        scores_path = scores_dir / f"symptom_{symptom_id}.json"
        if not scores_path.exists():
            logger.warning("No encoder scores for symptom %d", symptom_id)
            continue

        with open(scores_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scored = [(item["docno"], item["score"]) for item in data]
        scored.sort(key=lambda x: x[1], reverse=True)
        rankings[symptom_id] = scored[:1000]
        found += 1

    if found == 0:
        raise RuntimeError(
            f"No encoder score files found in {scores_dir}. "
            "Run encoder training and inference first."
        )

    logger.info("Run 1: loaded encoder scores for %d/18 symptoms", found)
    return rankings
