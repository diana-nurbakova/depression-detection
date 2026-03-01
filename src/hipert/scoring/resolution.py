"""Label resolution logic for the Llama → GPT cascade.

Implements the resolution table from spec Section 4.5.
"""

from __future__ import annotations

import math
from typing import Optional

from hipert.models import LLMOutput

# Symptom weights from spec Section 6.2
_SYMPTOM_WEIGHTS = {
    # Motor H/I (items 5, 6, 12-14)
    5: 1.0, 6: 1.0, 12: 1.0, 13: 1.0, 14: 1.0,
    # Verbal H/I (items 15-18)
    15: 0.9, 16: 0.9, 17: 0.9, 18: 0.9,
    # Organization/Memory (items 1-4)
    1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7,
    # Sustained Attention (items 7-11)
    7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
}


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
