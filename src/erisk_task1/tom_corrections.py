"""ToM-informed scoring corrections for eRisk Task 1.

Two complementary corrections:
  C1: Low-confidence gate — drops pass2 items with conf < threshold
  C2: Somatic coverage boost — adds estimated somatic mass when coverage gap detected

See: specs/task-1/tom/tom_correction_implementation_spec.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .models import ItemScore, ItemState

logger = logging.getLogger(__name__)

# BDI-II items classified as Somatic/Low-ToM
SOMATIC_ITEMS = frozenset({11, 15, 16, 18, 20, 21})


@dataclass
class TomCorrectionConfig:
    """Configuration for ToM-informed corrections."""
    enabled: bool = False

    # C1: Confidence gate
    conf_threshold: float = 0.5

    # C2: Somatic coverage boost
    base_threshold: int = 20       # gated_total must be >= this to trigger
    boost_amount: int = 9          # points added when somatic=0/6
    partial_scale_1: float = 0.6   # scale when somatic=1/6
    partial_scale_2: float = 0.3   # scale when somatic=2/6

    # Optional W_align filter for C2
    walign_threshold: Optional[float] = None  # None = disabled


@dataclass
class TomCorrectionResult:
    """Result of applying ToM corrections."""
    original_total: int
    gated_total: int
    boost_applied: int
    final_total: int
    items_gated: int
    somatic_evidence: int
    correction_log: list[str]

    def to_dict(self) -> dict:
        return {
            "original_total": self.original_total,
            "gated_total": self.gated_total,
            "boost_applied": self.boost_applied,
            "final_total": self.final_total,
            "items_gated": self.items_gated,
            "somatic_evidence": self.somatic_evidence,
            "correction_log": self.correction_log,
        }


def apply_tom_corrections(
    item_scores: dict[int, ItemScore],
    pass1_total: int,
    tom_summary: Optional[dict] = None,
    config: Optional[TomCorrectionConfig] = None,
) -> TomCorrectionResult:
    """Apply ToM-informed scoring corrections (C1 + C2).

    Args:
        item_scores: Dict of item_id → ItemScore from the scoring pipeline.
        pass1_total: Pass 1 total (sum of SCORED items, no Bayesian priors).
        tom_summary: ToM tracker summary dict (for W_align filter). Optional.
        config: Correction parameters. Uses defaults if None.

    Returns:
        TomCorrectionResult with final_total and audit trail.
    """
    if config is None:
        config = TomCorrectionConfig()

    correction_log: list[str] = []

    # Compute original total from item_scores
    original_total = sum(
        item.score for item in item_scores.values()
        if item.score is not None and item.state in (ItemState.SCORED, ItemState.EVIDENCE_OF_ABSENCE)
    )

    # ── C1: Confidence gate ──────────────────────────────────────────────────
    items_gated = 0
    gated_total = 0

    for item_id, item in item_scores.items():
        if item.score is None or item.score == 0:
            continue
        if item.state not in (ItemState.SCORED, ItemState.EVIDENCE_OF_ABSENCE):
            continue

        if item.confidence < config.conf_threshold:
            items_gated += 1
            correction_log.append(
                f"C1: Item {item_id} ({item.item_name}) gated "
                f"(score={item.score}, conf={item.confidence:.2f})"
            )
        else:
            gated_total += item.score

    # Safeguard: if gating zeroed everything but pass1 had evidence
    if gated_total == 0 and pass1_total > 0:
        gated_total = pass1_total
        correction_log.append(
            f"C1-safeguard: gated_total=0, falling back to pass1={pass1_total}"
        )

    if items_gated > 0:
        logger.info(
            "C1: Gated %d items (conf < %.2f), total %d → %d",
            items_gated, config.conf_threshold, original_total, gated_total,
        )

    # ── C2: Somatic coverage boost ───────────────────────────────────────────
    # Count somatic items with evidence after gating
    somatic_evidence = 0
    for item_id in SOMATIC_ITEMS:
        item = item_scores.get(item_id)
        if item is None:
            continue
        # Item must have score > 0 AND survive confidence gate
        if (item.score is not None and item.score > 0
                and item.confidence >= config.conf_threshold):
            somatic_evidence += 1

    # Determine boost
    boost = 0
    if somatic_evidence == 0 and gated_total >= config.base_threshold:
        boost = config.boost_amount
    elif somatic_evidence == 1 and gated_total >= config.base_threshold + 5:
        boost = round(config.boost_amount * config.partial_scale_1)
    elif somatic_evidence == 2 and gated_total >= config.base_threshold + 8:
        boost = round(config.boost_amount * config.partial_scale_2)

    # Optional W_align filter
    if boost > 0 and config.walign_threshold is not None and tom_summary:
        w_align = tom_summary.get("W_align", {})
        if w_align:
            last_turn = str(max(int(k) for k in w_align.keys()))
            final_walign = w_align[last_turn]
            if final_walign < config.walign_threshold:
                correction_log.append(
                    f"C2-suppressed: W_align={final_walign:.3f} < "
                    f"{config.walign_threshold} "
                    f"(good convergence, somatic absence likely genuine)"
                )
                boost = 0

    if boost > 0:
        correction_log.append(
            f"C2: Somatic boost +{boost} "
            f"(somatic_evidence={somatic_evidence}/6, "
            f"gated_total={gated_total})"
        )
        logger.info(
            "C2: Somatic boost +%d (somatic=%d/6, gated_total=%d)",
            boost, somatic_evidence, gated_total,
        )

    final_total = min(63, gated_total + boost)

    return TomCorrectionResult(
        original_total=original_total,
        gated_total=gated_total,
        boost_applied=boost,
        final_total=final_total,
        items_gated=items_gated,
        somatic_evidence=somatic_evidence,
        correction_log=correction_log,
    )
