"""Assessor agents — run 4 specialised BDI-II assessors on conversation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .llm_client import LLMClient, parse_json_response
from .models import (
    ASSESSOR_ITEMS,
    BDI_ITEMS,
    AssessorOutput,
    ItemScore,
    ItemState,
)
from .prompts import get_assessor_prompt

logger = logging.getLogger(__name__)


def _parse_assessor_response(
    assessor_name: str, raw_json: Optional[dict]
) -> AssessorOutput:
    """Parse an assessor's JSON response into AssessorOutput."""
    items = []
    cross_obs = ""

    if raw_json is None:
        logger.warning("Assessor %s returned unparseable response", assessor_name)
        # Return NO_EVIDENCE for all items
        for item_id in ASSESSOR_ITEMS[assessor_name]:
            items.append(
                ItemScore(
                    item_id=item_id,
                    item_name=BDI_ITEMS[item_id],
                    score=None,
                    confidence=0.0,
                    state=ItemState.NO_EVIDENCE,
                    evidence="Assessor response could not be parsed",
                )
            )
        return AssessorOutput(assessor_name=assessor_name, items=items)

    cross_obs = raw_json.get("cross_observations", "")

    for item_data in raw_json.get("items", []):
        item_id = item_data.get("id")
        if item_id is None:
            continue

        score = item_data.get("score")
        confidence = float(item_data.get("confidence") or 0.0)
        state_str = item_data.get("state", "")
        evidence = item_data.get("evidence", "")

        # Determine state
        if state_str in ("SCORED", "EVIDENCE_OF_ABSENCE", "NO_EVIDENCE"):
            state = ItemState(state_str)
        elif score is None:
            state = ItemState.NO_EVIDENCE
        elif score == 0 and confidence >= 0.5:
            state = ItemState.EVIDENCE_OF_ABSENCE
        elif score == 0 and confidence < 0.5:
            state = ItemState.NO_EVIDENCE
        else:
            state = ItemState.SCORED

        # Ensure score is int or None
        if score is not None:
            score = int(score)

        items.append(
            ItemScore(
                item_id=item_id,
                item_name=BDI_ITEMS.get(item_id, f"Item {item_id}"),
                score=score,
                confidence=confidence,
                state=state,
                evidence=evidence,
            )
        )

    # Fill in any missing items as NO_EVIDENCE
    seen_ids = {item.item_id for item in items}
    for item_id in ASSESSOR_ITEMS[assessor_name]:
        if item_id not in seen_ids:
            items.append(
                ItemScore(
                    item_id=item_id,
                    item_name=BDI_ITEMS[item_id],
                    score=None,
                    confidence=0.0,
                    state=ItemState.NO_EVIDENCE,
                    evidence="Not included in assessor response",
                )
            )

    return AssessorOutput(
        assessor_name=assessor_name, items=items, cross_observations=cross_obs
    )


def run_single_assessor(
    assessor_name: str,
    client: LLMClient,
    transcript: str,
    linguistic_summary: str = "",
) -> AssessorOutput:
    """Run a single assessor on the conversation transcript."""
    system_prompt = get_assessor_prompt(assessor_name)

    user_message = (
        f"Analyse this conversation transcript and score the relevant BDI-II items.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    if linguistic_summary:
        user_message += f"\n\nLINGUISTIC FEATURES:\n{linguistic_summary}"

    logger.info("Running %s assessor...", assessor_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response_text = client.complete(messages)

    raw_json = parse_json_response(response_text)
    output = _parse_assessor_response(assessor_name, raw_json)
    output.raw_response = response_text

    scored_count = sum(1 for i in output.items if i.state == ItemState.SCORED)
    logger.info(
        "%s assessor done: %d/%d items scored",
        assessor_name, scored_count, len(output.items),
    )
    return output


def run_all_assessors(
    client: LLMClient,
    transcript: str,
    linguistic_summary: str = "",
    parallel: bool = True,
) -> dict[str, AssessorOutput]:
    """Run all 4 assessors on the conversation transcript.

    Returns dict keyed by assessor name.
    """
    assessor_names = list(ASSESSOR_ITEMS.keys())

    if parallel:
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    run_single_assessor, name, client, transcript, linguistic_summary
                ): name
                for name in assessor_names
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error("Assessor %s failed: %s", name, e)
                    # Return empty output
                    results[name] = _parse_assessor_response(name, None)
        return results
    else:
        return {
            name: run_single_assessor(name, client, transcript, linguistic_summary)
            for name in assessor_names
        }
