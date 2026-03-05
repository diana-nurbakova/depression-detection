"""Two-pass scoring pipeline with Bayesian severity prior."""

from __future__ import annotations

import logging
from statistics import mean
from typing import Optional

from .models import (
    ASSESSOR_ITEMS,
    BDI_ITEMS,
    FAST_SCREEN_ITEMS,
    TIER_1_ITEMS,
    TIER_2_ITEMS,
    TIER_3_ITEMS,
    AssessorOutput,
    ItemScore,
    ItemState,
    LinguisticFeatures,
    SeverityBand,
    score_to_band,
)
from .linguistic import compute_cumulative_features, estimate_engagement_band

logger = logging.getLogger(__name__)


def _majority_vote(*bands: SeverityBand) -> SeverityBand:
    """Return the most common band. Tie-break with the first argument."""
    from collections import Counter
    counts = Counter(bands)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    # If tie, return the first band passed in
    for b in bands:
        if counts[b] == top_count:
            return b
    return bands[0]


def collect_item_scores(
    assessor_outputs: dict[str, AssessorOutput],
) -> dict[int, ItemScore]:
    """Flatten all assessor outputs into a single dict keyed by item_id."""
    scores = {}
    for assessor_name, output in assessor_outputs.items():
        for item in output.items:
            scores[item.item_id] = item
    return scores


def pass1_score(item_scores: dict[int, ItemScore]) -> int:
    """Compute Pass 1 total: sum of SCORED items only."""
    total = 0
    for item in item_scores.values():
        if item.state == ItemState.SCORED and item.score is not None:
            total += item.score
    return total


def compute_preliminary_consensus(
    pass1_total: int,
    features_history: list[LinguisticFeatures],
) -> tuple[SeverityBand, SeverityBand, SeverityBand, SeverityBand]:
    """Compute preliminary severity from 3 independent signals.

    Returns: (consensus_band, assessor_band, absolutist_band, engagement_band)
    """
    assessor_band = score_to_band(pass1_total)
    cum = compute_cumulative_features(features_history)
    absolutist_band = cum["absolutist_band"]
    engagement_band = estimate_engagement_band(features_history)
    consensus = _majority_vote(assessor_band, absolutist_band, engagement_band)
    return consensus, assessor_band, absolutist_band, engagement_band


def pass2_bayesian_prior(
    item_scores: dict[int, ItemScore],
    pass1_total: int,
    consensus_band: SeverityBand,
    assessor_band: SeverityBand,
) -> dict[int, ItemScore]:
    """Apply Bayesian prior to NO_EVIDENCE items.

    Priors are only applied when Pass 1 band disagrees with consensus.
    """
    # If Pass 1 band matches consensus, skip prior
    if assessor_band == consensus_band:
        logger.info(
            "Pass 2: Skip prior — assessor band (%s) matches consensus", assessor_band.value
        )
        return item_scores

    logger.info(
        "Pass 2: Applying prior — assessor band (%s) != consensus (%s)",
        assessor_band.value, consensus_band.value,
    )

    updated = dict(item_scores)

    for item_id in range(1, 22):
        if item_id not in updated:
            continue
        item = updated[item_id]

        if item.state != ItemState.NO_EVIDENCE:
            continue

        # Determine prior based on consensus band and item tier
        prior_score = 0
        if consensus_band == SeverityBand.MINIMAL:
            prior_score = 0
        elif consensus_band == SeverityBand.MILD:
            if item_id in TIER_3_ITEMS:
                prior_score = 0
            else:
                prior_score = 1
        elif consensus_band == SeverityBand.MODERATE:
            prior_score = 1
        elif consensus_band == SeverityBand.SEVERE:
            if item_id in FAST_SCREEN_ITEMS:
                prior_score = 2
            else:
                prior_score = 1

        if prior_score > 0:
            updated[item_id] = ItemScore(
                item_id=item_id,
                item_name=item.item_name,
                score=prior_score,
                confidence=0.3,
                state=ItemState.SCORED,
                evidence=f"Bayesian prior ({consensus_band.value} consensus)",
                source="prior",
            )
            logger.debug(
                "Prior applied: Item %d (%s) → %d", item_id, item.item_name, prior_score
            )

    return updated


def compute_final_total(item_scores: dict[int, ItemScore]) -> int:
    """Compute final BDI-II total from all scored items."""
    total = 0
    for item in item_scores.values():
        if item.score is not None and item.state in (ItemState.SCORED, ItemState.EVIDENCE_OF_ABSENCE):
            total += item.score
    return total


def select_top4_mechanical(item_scores: dict[int, ItemScore]) -> list[ItemScore]:
    """Select top 4 symptoms by confidence × severity (mechanical selection).

    The Justificator will refine this.
    """
    scored = [
        item for item in item_scores.values()
        if item.state == ItemState.SCORED and item.score is not None and item.score > 0
    ]

    # Sort by confidence * score descending, then by Fast Screen membership
    scored.sort(
        key=lambda x: (
            x.confidence * x.score,
            1 if x.item_id in FAST_SCREEN_ITEMS else 0,
        ),
        reverse=True,
    )

    return scored[:4]


def run_scoring_pipeline(
    assessor_outputs: dict[str, AssessorOutput],
    features_history: list[LinguisticFeatures],
) -> dict:
    """Execute the full 2-pass scoring pipeline.

    Returns a dict with all scoring metadata.
    """
    # Collect scores
    item_scores = collect_item_scores(assessor_outputs)

    # Pass 1
    p1_total = pass1_score(item_scores)
    logger.info("Pass 1 total: %d (%s)", p1_total, score_to_band(p1_total).value)

    # Preliminary consensus
    consensus, assessor_band, abs_band, eng_band = compute_preliminary_consensus(
        p1_total, features_history
    )
    logger.info(
        "Preliminary consensus: %s (assessor=%s, absolutist=%s, engagement=%s)",
        consensus.value, assessor_band.value, abs_band.value, eng_band.value,
    )

    # Pass 2
    item_scores = pass2_bayesian_prior(item_scores, p1_total, consensus, assessor_band)
    p2_total = compute_final_total(item_scores)
    p2_band = score_to_band(p2_total)
    logger.info("Pass 2 total: %d (%s)", p2_total, p2_band.value)

    # Mechanical top-4
    top4 = select_top4_mechanical(item_scores)

    cum_features = compute_cumulative_features(features_history)

    return {
        "item_scores": item_scores,
        "pass1_total": p1_total,
        "pass2_total": p2_total,
        "pass2_band": p2_band,
        "consensus_band": consensus,
        "assessor_band": assessor_band,
        "absolutist_band": abs_band,
        "engagement_band": eng_band,
        "top4_mechanical": top4,
        "linguistic_cumulative": cum_features,
    }
