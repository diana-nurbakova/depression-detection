"""Run 2: HiPerT_full — Cross-encoder reranker (v2).

v2 change: HiPerT is now the secondary submission (Run 2).

Cross-encoder reranker trained on LLM silver labels via CORAL ordinal
regression or ListMLE. Ensemble of 3 backbones × 5 folds.

Scoring function (CORAL):
    φ(s,q) = (1/3) Σ_backbone (1/5) Σ_fold [σ(f₁)+σ(f₂)+σ(f₃)]

Reads pre-computed encoder scores from output/encoder_scores_v2/.

Spec reference: hipert_v2_spec.md Section 8.3
"""

from __future__ import annotations

import json
import logging

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)


@register_run(2)
def generate_hipert_v2(config: PipelineConfig) -> Rankings:
    """Generate Run 2 rankings from cross-encoder v2 scores.

    Looks for pre-computed scores at:
        output/encoder_scores_v2/symptom_{id}.json

    Falls back to v1 encoder scores at:
        output/encoder_scores/symptom_{id}.json
    """
    # Try v2 scores first, then v1 fallback
    scores_dir = config.output_dir / "encoder_scores_v2"
    if not scores_dir.exists():
        scores_dir = config.output_dir / "encoder_scores"

    if not scores_dir.exists():
        raise RuntimeError(
            f"Encoder scores not found. Run 'hipert train-v2' and 'hipert infer-v2' first."
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
            "Run 'hipert train-v2' and 'hipert infer-v2' first."
        )

    logger.info("Run 2: loaded encoder scores for %d/18 symptoms", found)
    return rankings
