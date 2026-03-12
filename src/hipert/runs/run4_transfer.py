"""Run 4: DepTransfer — Depression-only cross-condition transfer.

Train encoder on depression data only (BDI-Sen + eRisk 2025 T1 + RedSM5),
skip ADHD silver label fine-tuning. At inference, apply the
depression-trained encoder using BDI-II → ASRS symptom mappings.

For unmapped symptoms (3, 14, 15-18), falls back to cosine similarity (Run 5).

Status: STUB — requires Stage A encoder training.
"""

from __future__ import annotations

import json
import logging

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)

# BDI-II → ASRS symptom mapping (spec Section 7.3)
BDI_ASRS_MAPPING = {
    # ASRS item → BDI-II symptom ID
    8: 19, 9: 19, 10: 19, 11: 19,   # Concentration Difficulty
    5: 11, 6: 11, 12: 11, 13: 11,   # Agitation
    4: 15,                            # Loss of Energy
    7: 16,                            # Sleep Changes
    1: 13, 2: 13,                     # Indecisiveness
}

# ASRS items with no BDI-II mapping → use cosine fallback
UNMAPPED_ITEMS = {3, 14, 15, 16, 17, 18}


@register_run(4)
def generate_dep_transfer(config: PipelineConfig) -> Rankings:
    """Generate Run 4 rankings from depression-trained encoder.

    Looks for pre-computed transfer scores at:
        output/transfer_scores/symptom_{id}.json

    For unmapped symptoms, falls back to Run 5 (BiEnc baseline).
    """
    scores_dir = config.output_dir / "transfer_scores"
    rankings: Rankings = {}

    # Try to load transfer encoder scores for mapped symptoms
    mapped_count = 0
    if scores_dir.exists():
        for symptom_id in range(1, 19):
            if symptom_id in UNMAPPED_ITEMS:
                continue

            scores_path = scores_dir / f"symptom_{symptom_id}.json"
            if not scores_path.exists():
                continue

            with open(scores_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            scored = [(item["docno"], item["score"]) for item in data]
            scored.sort(key=lambda x: x[1], reverse=True)
            rankings[symptom_id] = scored[:1000]
            mapped_count += 1

    if mapped_count == 0:
        raise RuntimeError(
            f"Transfer scores not found at {scores_dir}. "
            "Run Stage A encoder training and inference first."
        )

    logger.info(
        "Run 4: loaded transfer scores for %d mapped symptoms", mapped_count,
    )

    # Fall back to Run 5 (BiEnc) for unmapped symptoms
    from hipert.runs.registry import RUN_REGISTRY
    if 5 in RUN_REGISTRY:
        bienc_rankings = RUN_REGISTRY[5](config)
        for symptom_id in UNMAPPED_ITEMS:
            if symptom_id in bienc_rankings:
                rankings[symptom_id] = bienc_rankings[symptom_id]
                logger.debug(
                    "Symptom %d (unmapped): using BiEnc fallback", symptom_id,
                )

    return rankings
