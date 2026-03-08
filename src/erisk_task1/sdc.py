"""Score Distribution Constraint (SDC) for BDI-II totals.

Addresses the "expressive LoRA" problem where moderate-depression personas
use maximum-severity language, causing assessors to assign score=3 on many
items and pushing raw totals to 34-36 — indistinguishable from genuinely
severe personas.

See specs/task-1/sdc_spec.md for full specification.

Pipeline position: Assessors → Item scores → **SDC adjustment** → Correction → Final BDI
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import ItemScore, ItemState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists for moderate signal detection (from spec Section "Keyword
# lists for moderate signals")
# ---------------------------------------------------------------------------

MINIMIZERS = [
    "i guess", "i suppose", "maybe", "a bit", "probably",
    "not sure", "a little", "sort of", "kind of",
]

FUTURE_ORIENTATION = [
    "i hope", "willing to try", "maybe i should",
    "i want to", "looking forward", "plan to",
    "going to try", "i could",
]

FUNCTIONAL_MARKERS = [
    "work", "job", "kids", "school", "cooking",
    "cleaning", "shopping", "driving", "commute",
]


@dataclass
class SDCResult:
    """Result of applying SDC to a persona's item scores."""
    applied: bool
    n_score3_before: int
    n_downgrades: int
    raw_total_before: int
    adjusted_total: int
    downgraded_items: list[int] = field(default_factory=list)
    moderate_signals: list[str] = field(default_factory=list)
    reason_skipped: str = ""

    def to_dict(self) -> dict:
        return {
            "sdc_applied": self.applied,
            "n_score3_before": self.n_score3_before,
            "n_downgrades": self.n_downgrades,
            "raw_total_before": self.raw_total_before,
            "adjusted_total": self.adjusted_total,
            "downgraded_items": self.downgraded_items,
            "moderate_signals": self.moderate_signals,
            "reason_skipped": self.reason_skipped,
        }


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def detect_moderate_signals(
    item_scores: dict[int, ItemScore],
    transcript: str,
) -> list[str]:
    """Detect moderate signals from the conversation transcript and item scores.

    Returns a list of signal names that were detected.
    See spec Section "Moderate Signals (Step 2c)".
    """
    signals: list[str] = []

    # 1. Minimizing language detected
    if _count_keyword_hits(transcript, MINIMIZERS) >= 3:
        signals.append("minimizing_language")

    # 2. Future orientation present
    if _count_keyword_hits(transcript, FUTURE_ORIENTATION) >= 1:
        signals.append("future_orientation")

    # 3. Functional activity described
    if _count_keyword_hits(transcript, FUNCTIONAL_MARKERS) >= 2:
        signals.append("functional_activity")

    # 4. Suicidal ideation absent (Item 9 score=0 with EVIDENCE_OF_ABSENCE)
    item9 = item_scores.get(9)
    if item9 is not None:
        if (item9.score == 0 or item9.score is None) and item9.state == ItemState.EVIDENCE_OF_ABSENCE:
            signals.append("suicidal_ideation_absent")

    # 5. Domain gaps: more than 6 items are NO_EVIDENCE
    n_no_evidence = sum(
        1 for item in item_scores.values()
        if item.state == ItemState.NO_EVIDENCE
    )
    if n_no_evidence > 6:
        signals.append("domain_gaps")

    return signals


def apply_sdc(
    item_scores: dict[int, ItemScore],
    transcript: str,
    min_signals: int = 2,
) -> tuple[dict[int, ItemScore], SDCResult]:
    """Apply Score Distribution Constraint to item scores.

    Args:
        item_scores: Dict of item_id → ItemScore from assessors.
        transcript: Full conversation transcript for moderate signal detection.
        min_signals: Minimum number of moderate signals required to trigger SDC.

    Returns:
        Tuple of (adjusted_item_scores, SDCResult).
    """
    # Compute raw total from scored items
    raw_total = sum(
        item.score for item in item_scores.values()
        if item.state == ItemState.SCORED and item.score is not None
    )

    # Step 1: Count score=3 items
    score3_items = [
        item for item in item_scores.values()
        if item.state == ItemState.SCORED and item.score == 3
    ]
    n_score3 = len(score3_items)

    # Step 2: Check if redistribution is warranted
    # a) raw_total >= 28
    if raw_total < 28:
        return item_scores, SDCResult(
            applied=False, n_score3_before=n_score3, n_downgrades=0,
            raw_total_before=raw_total, adjusted_total=raw_total,
            reason_skipped=f"raw_total={raw_total} < 28",
        )

    # b) n_score3 >= 6
    if n_score3 < 6:
        return item_scores, SDCResult(
            applied=False, n_score3_before=n_score3, n_downgrades=0,
            raw_total_before=raw_total, adjusted_total=raw_total,
            reason_skipped=f"n_score3={n_score3} < 6",
        )

    # c) At least min_signals moderate signals
    signals = detect_moderate_signals(item_scores, transcript)
    if len(signals) < min_signals:
        return item_scores, SDCResult(
            applied=False, n_score3_before=n_score3, n_downgrades=0,
            raw_total_before=raw_total, adjusted_total=raw_total,
            moderate_signals=signals,
            reason_skipped=f"only {len(signals)} moderate signals (need {min_signals})",
        )

    # Step 3: Determine number of downgrades
    n_downgrades = min(
        n_score3 - 4,           # keep at least 4 items at score=3
        (raw_total - 28) // 1,  # bring total closer to moderate range
    )
    n_downgrades = min(n_downgrades, 4)  # cap at 4

    if n_downgrades <= 0:
        return item_scores, SDCResult(
            applied=False, n_score3_before=n_score3, n_downgrades=0,
            raw_total_before=raw_total, adjusted_total=raw_total,
            moderate_signals=signals,
            reason_skipped="n_downgrades=0 after calculation",
        )

    # Step 4: Select items to downgrade
    # Sort score=3 items by confidence ascending (lowest confidence first)
    score3_items.sort(key=lambda x: (x.confidence, x.item_id))

    # Downgrade the n_downgrades lowest-confidence items from 3 → 2
    adjusted = dict(item_scores)
    downgraded_ids: list[int] = []

    for item in score3_items[:n_downgrades]:
        adjusted[item.item_id] = ItemScore(
            item_id=item.item_id,
            item_name=item.item_name,
            score=2,
            confidence=item.confidence,
            state=item.state,
            evidence=item.evidence,
            source="sdc_downgrade",
        )
        downgraded_ids.append(item.item_id)

    # Step 5: Recalculate total
    adjusted_total = raw_total - n_downgrades

    logger.info(
        "SDC applied: raw=%d -> adjusted=%d (-%d), "
        "n_score3=%d->%d, signals=%s, downgraded=%s",
        raw_total, adjusted_total, n_downgrades,
        n_score3, n_score3 - n_downgrades,
        signals, downgraded_ids,
    )

    return adjusted, SDCResult(
        applied=True,
        n_score3_before=n_score3,
        n_downgrades=n_downgrades,
        raw_total_before=raw_total,
        adjusted_total=adjusted_total,
        downgraded_items=downgraded_ids,
        moderate_signals=signals,
    )
