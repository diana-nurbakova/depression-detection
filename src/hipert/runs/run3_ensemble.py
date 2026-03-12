"""Run 3: Ensemble_1+2 — Weighted ensemble of Run 1 and Run 2.

Combines HiPerT-ADHD encoder scores (Run 1) with calibrated LLM cascade
scores (Run 2). Uses Reciprocal Rank Fusion (RRF) as the default
parameter-free method.

Scoring function (RRF):
    φ_rrf(s, q) = 1/(k + rank_encoder(s)) + 1/(k + rank_llm(s))
    where k = 60 (standard RRF constant)

Falls back to Run 2 only if Run 1 is not available yet.
"""

from __future__ import annotations

import logging

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF constant


def reciprocal_rank_fusion(
    rankings_a: Rankings,
    rankings_b: Rankings,
    k: int = RRF_K,
    top_n: int = 1000,
) -> Rankings:
    """Combine two sets of rankings using Reciprocal Rank Fusion.

    For each symptom, computes:
        score(s) = 1/(k + rank_a(s)) + 1/(k + rank_b(s))

    Sentences appearing in only one ranking get 0 for the missing rank
    contribution (effectively rank = infinity).
    """
    all_symptoms = set(rankings_a.keys()) | set(rankings_b.keys())
    fused: Rankings = {}

    for symptom_id in sorted(all_symptoms):
        # Build rank lookup for each system
        rank_a: dict[str, int] = {}
        for rank, (docno, _score) in enumerate(
            rankings_a.get(symptom_id, []), 1,
        ):
            rank_a[docno] = rank

        rank_b: dict[str, int] = {}
        for rank, (docno, _score) in enumerate(
            rankings_b.get(symptom_id, []), 1,
        ):
            rank_b[docno] = rank

        # Union of all sentence IDs
        all_docnos = set(rank_a.keys()) | set(rank_b.keys())

        # Compute RRF score for each
        scored = []
        for docno in all_docnos:
            rrf_score = 0.0
            if docno in rank_a:
                rrf_score += 1.0 / (k + rank_a[docno])
            if docno in rank_b:
                rrf_score += 1.0 / (k + rank_b[docno])
            scored.append((docno, rrf_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        fused[symptom_id] = scored[:top_n]

    return fused


@register_run(3)
def generate_ensemble(config: PipelineConfig) -> Rankings:
    """Generate Run 3 rankings by fusing Run 1 and Run 2."""
    from hipert.runs.registry import RUN_REGISTRY

    # Generate Run 2 (LLM cascade)
    if 2 not in RUN_REGISTRY:
        raise RuntimeError("Run 2 (LLM_cascade) must be available for ensemble")

    run2_rankings = RUN_REGISTRY[2](config)

    # Try Run 1 (full pipeline)
    if 1 in RUN_REGISTRY:
        try:
            run1_rankings = RUN_REGISTRY[1](config)
            if run1_rankings:
                logger.info(
                    "Ensemble: fusing Run 1 (%d symptoms) + Run 2 (%d symptoms) via RRF (k=%d)",
                    len(run1_rankings), len(run2_rankings), RRF_K,
                )
                return reciprocal_rank_fusion(run1_rankings, run2_rankings)
        except Exception as e:
            logger.warning("Run 1 unavailable for ensemble: %s. Falling back.", e)

    # Fallback: fuse Run 5 (bienc) + Run 2 (llm) instead
    if 5 in RUN_REGISTRY:
        logger.info(
            "Run 1 not available. Fusing Run 5 (BiEnc) + Run 2 (LLM) as fallback."
        )
        run5_rankings = RUN_REGISTRY[5](config)
        return reciprocal_rank_fusion(run5_rankings, run2_rankings)

    # No fusion partner — return Run 2 as-is
    logger.warning("No fusion partner for ensemble. Returning Run 2 as-is.")
    return run2_rankings
