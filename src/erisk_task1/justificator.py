"""Justificator agent — post-conversation coherence check and final scoring."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .llm_client import LLMClient, parse_json_response
from .models import (
    BDI_ITEMS,
    AssessorOutput,
    ItemScore,
    ItemState,
    JustificatorOutput,
    LinguisticFeatures,
    SeverityBand,
    score_to_band,
)
from .linguistic import compute_cumulative_features, detect_persona_profile
from .prompts import JUSTIFICATOR_PROMPT

logger = logging.getLogger(__name__)


def run_justificator(
    client: LLMClient,
    persona_id: str,
    transcript: str,
    assessor_outputs: dict[str, AssessorOutput],
    item_scores: dict[int, ItemScore],
    pass2_total: int,
    pass2_band: SeverityBand,
    features_history: list[LinguisticFeatures],
) -> JustificatorOutput:
    """Run the Justificator agent for final scoring and coherence check."""

    # Build input payload
    assessor_data = {}
    for name, output in assessor_outputs.items():
        assessor_data[name] = {
            "items": [
                {
                    "id": item.item_id,
                    "name": item.item_name,
                    "score": item.score,
                    "state": item.state.value,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in output.items
            ],
            "cross_observations": output.cross_observations,
        }

    pass2_scores = {}
    for item_id, item in item_scores.items():
        key = f"{item_id}_{BDI_ITEMS.get(item_id, 'unknown').lower().replace(' ', '_')}"
        pass2_scores[key] = {
            "score": item.score,
            "confidence": item.confidence,
            "state": item.state.value,
        }

    cum = compute_cumulative_features(features_history)
    profile = detect_persona_profile(features_history)

    input_data = json.dumps({
        "persona_name": persona_id,
        "conversation_transcript": transcript,
        "assessor_outputs": assessor_data,
        "pass2_scores": pass2_scores,
        "pass2_total": pass2_total,
        "pass2_band": pass2_band.value,
        "linguistic_features": {
            "absolutist_density": cum["absolutist_density"],
            "absolutist_band": cum["absolutist_band"].value,
            "coping_count": cum["total_coping"],
            "hedging_count": cum["total_hedging"],
            "profile": profile,
        },
    }, indent=2)

    messages = [
        {"role": "system", "content": JUSTIFICATOR_PROMPT},
        {"role": "user", "content": input_data},
    ]

    logger.info("Running Justificator for persona %s...", persona_id)
    response_text = client.complete(messages)
    parsed = parse_json_response(response_text)

    if parsed is None:
        logger.warning("Justificator returned unparseable response")
        # Return pass-through (no adjustments)
        return JustificatorOutput(
            patterns_detected=[],
            adjustments_made=[],
            final_total=pass2_total,
            final_band=pass2_band,
            top_4_symptoms=[],
            clinical_narrative="Justificator response could not be parsed.",
            item_scores={
                f"{k}_{BDI_ITEMS.get(k, '').lower().replace(' ', '_')}": v.score or 0
                for k, v in item_scores.items()
            },
        )

    # Parse coherence check
    coherence = parsed.get("coherence_check", {})
    patterns = coherence.get("patterns_detected", [])
    adjustments = coherence.get("adjustments_made", [])

    # Parse final scores
    final_scores = parsed.get("final_scores", {})
    final_total = final_scores.get("total", pass2_total)
    final_band_str = final_scores.get("band", pass2_band.value)
    try:
        final_band = SeverityBand(final_band_str)
    except ValueError:
        final_band = score_to_band(final_total)

    final_item_scores = final_scores.get("item_scores", {})

    # Parse top-4 symptoms
    top4 = parsed.get("top_4_symptoms", [])

    # Parse narrative
    narrative = parsed.get("clinical_narrative", "")

    # Apply adjustments to item_scores
    for adj in adjustments:
        adj_item_id = adj.get("item_id")
        if adj_item_id and adj_item_id in item_scores:
            new_score = adj.get("adjusted_score")
            if new_score is not None:
                old = item_scores[adj_item_id]
                item_scores[adj_item_id] = ItemScore(
                    item_id=adj_item_id,
                    item_name=old.item_name,
                    score=int(new_score),
                    confidence=old.confidence,
                    state=ItemState.SCORED,
                    evidence=adj.get("reason", old.evidence),
                    source="justificator_adjusted",
                )

    logger.info(
        "Justificator done: total=%d (%s), %d adjustments, %d patterns",
        final_total, final_band.value, len(adjustments), len(patterns),
    )

    return JustificatorOutput(
        patterns_detected=patterns,
        adjustments_made=adjustments,
        final_total=final_total,
        final_band=final_band,
        top_4_symptoms=top4,
        clinical_narrative=narrative,
        item_scores=final_item_scores,
    )
