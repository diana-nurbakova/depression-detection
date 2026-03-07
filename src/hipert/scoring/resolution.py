"""Label resolution logic for the Llama → GPT cascade.

Implements the resolution table from spec Section 4.5.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from hipert.models import LLMOutput

logger = logging.getLogger(__name__)

# Base symptom weights from spec Section 6.2
_BASE_SYMPTOM_WEIGHTS = {
    # Motor H/I (items 5, 6, 12-14)
    5: 1.0, 6: 1.0, 12: 1.0, 13: 1.0, 14: 1.0,
    # Verbal H/I (items 15-18)
    15: 0.9, 16: 0.9, 17: 0.9, 18: 0.9,
    # Organization/Memory (items 1-4)
    1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7,
    # Sustained Attention (items 7-11)
    7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
}

# Active weights (may be adjusted by DepreSym bias correction)
_SYMPTOM_WEIGHTS = dict(_BASE_SYMPTOM_WEIGHTS)


def apply_depresym_bias_correction(project_root: Path) -> None:
    """Apply DepreSym-derived weight adjustments to symptom weights.

    Loads depresym_analysis/symptom_weight_adjustments.json and multiplies
    base weights by the recommended correction factor for overlapping symptoms.
    Formula: adjusted = base_weight * recommended_symptom_weight
    """
    adjustments_path = project_root / "depresym_analysis" / "symptom_weight_adjustments.json"
    if not adjustments_path.exists():
        logger.warning(
            "DepreSym bias correction enabled but %s not found. "
            "Run scripts/depresym_bias_profiling.py first. Using base weights.",
            adjustments_path,
        )
        return

    with open(adjustments_path, encoding="utf-8") as f:
        adjustments = json.load(f)

    for asrs_str, adj in adjustments.items():
        asrs_id = int(asrs_str)
        if asrs_id in _SYMPTOM_WEIGHTS:
            base = _BASE_SYMPTOM_WEIGHTS[asrs_id]
            correction_factor = adj["recommended_symptom_weight"]
            _SYMPTOM_WEIGHTS[asrs_id] = round(base * correction_factor, 4)
            logger.info(
                "  ASRS %d: %.3f -> %.3f (BDI-%d %s, FPR=%.3f)",
                asrs_id, base, _SYMPTOM_WEIGHTS[asrs_id],
                adj["source_bdi_symptom"], adj["source_name"], adj["gpt4_fpr"],
            )

    logger.info("DepreSym bias correction applied to %d ASRS items.", len(adjustments))


def reset_weights() -> None:
    """Reset symptom weights to base values (for testing)."""
    _SYMPTOM_WEIGHTS.update(_BASE_SYMPTOM_WEIGHTS)


def resolve_label(
    llama_output: LLMOutput,
    gpt_output: Optional[LLMOutput],
    escalated: bool,
    symptom_id: int,
) -> tuple[int, float]:
    """Resolve final label and composite confidence weight.

    Args:
        llama_output: Parsed Llama output.
        gpt_output: Parsed GPT output (None if not escalated).
        escalated: Whether the case was escalated.
        symptom_id: ASRS item number (1-18).

    Returns:
        Tuple of (final_label, confidence_weight).
        confidence_weight = resolution_weight * symptom_weight
    """
    symptom_weight = _SYMPTOM_WEIGHTS.get(symptom_id, 0.7)

    if not escalated or gpt_output is None:
        # No escalation — use Llama label directly
        if llama_output.confidence >= 4:
            resolution_weight = 0.80
        else:
            # CONFIDENCE == 3 (moderate) — lower weight
            resolution_weight = 0.60
        final_label = llama_output.score
    else:
        # Escalated — compare Llama and GPT
        score_diff = abs(llama_output.score - gpt_output.score)

        if score_diff == 0:
            # GPT agrees with Llama
            resolution_weight = 0.85
            final_label = llama_output.score
        elif score_diff == 1:
            # GPT differs by 1 — trust GPT
            resolution_weight = 0.65
            final_label = gpt_output.score
        elif gpt_output.confidence <= 2 and llama_output.confidence <= 2:
            # Both models uncertain — take mean, round down
            resolution_weight = 0.30
            final_label = math.floor(
                (llama_output.score + gpt_output.score) / 2,
            )
        else:
            # GPT differs by >= 2 — trust GPT but low confidence
            resolution_weight = 0.45
            final_label = gpt_output.score

    # Clamp final label to valid range
    final_label = max(0, min(3, final_label))

    confidence_weight = round(resolution_weight * symptom_weight, 4)
    return final_label, confidence_weight
