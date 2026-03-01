"""Escalation trigger logic for the Llama → GPT cascade.

Implements the 5 escalation rules from spec Section 4.2.
"""

from __future__ import annotations

from hipert.models import LLMOutput

# Inattention cluster: items 7-11 (Sustained Attention/Distractibility)
_INATTENTION_ITEMS = {7, 8, 9, 10, 11}

# Keywords that indicate cross-diagnostic confounders
_CROSS_DIAGNOSTIC_KEYWORDS = {
    "depression", "depressive", "depressed",
    "anxiety", "anxious", "gad",
    "fatigue", "tired", "exhaustion", "exhausted",
    "sleep", "insomnia",
}


def check_escalation(
    output: LLMOutput,
    symptom_id: int,
) -> tuple[bool, list[str]]:
    """Check whether a Llama output should be escalated to GPT.

    Args:
        output: Parsed LLM output from Llama.
        symptom_id: ASRS item number (1-18).

    Returns:
        Tuple of (should_escalate, list of trigger descriptions).
    """
    triggers: list[str] = []

    # Rule 1: Low confidence
    if output.confidence <= 2:
        triggers.append(
            f"Rule 1: CONFIDENCE={output.confidence} <= 2 (high uncertainty)",
        )

    # Rule 2: Internal inconsistency between fields and score
    _check_inconsistency(output, triggers)

    # Rule 3: Recognized confounders on borderline cases
    if (
        output.confounders.upper() != "NONE"
        and output.score in {1, 2}
    ):
        triggers.append(
            f"Rule 3: CONFOUNDERS='{output.confounders}' with "
            f"borderline SCORE={output.score}",
        )

    # Rule 4: Cross-diagnostic overlap for inattention items
    if (
        symptom_id in _INATTENTION_ITEMS
        and output.score >= 2
        and _has_cross_diagnostic_confounders(output.confounders)
    ):
        triggers.append(
            f"Rule 4: Inattention item {symptom_id} with SCORE={output.score} "
            f"and cross-diagnostic confounders: '{output.confounders}'",
        )

    # Rule 5: Moderate confidence on boundary cases
    if output.score in {1, 2} and output.confidence == 3:
        triggers.append(
            f"Rule 5: Boundary SCORE={output.score} with "
            f"CONFIDENCE=3 (moderate confidence)",
        )

    return bool(triggers), triggers


def _check_inconsistency(output: LLMOutput, triggers: list[str]) -> None:
    """Check for internal inconsistencies between fields and score."""
    sm = output.symptom_match.upper()
    sr = output.self_reference.upper()
    dl = output.detail_level.upper()
    score = output.score

    if sm == "NO" and score >= 2:
        triggers.append(
            f"Rule 2: SYMPTOM_MATCH=NO but SCORE={score} >= 2",
        )
    if sm == "YES" and score <= 0:
        triggers.append(
            f"Rule 2: SYMPTOM_MATCH=YES but SCORE={score} <= 0",
        )
    if sr == "NONE" and score >= 1:
        triggers.append(
            f"Rule 2: SELF_REFERENCE=NONE but SCORE={score} >= 1",
        )
    if dl == "NONE" and score >= 2:
        triggers.append(
            f"Rule 2: DETAIL_LEVEL=NONE but SCORE={score} >= 2",
        )
    if dl == "HIGH" and score <= 1:
        triggers.append(
            f"Rule 2: DETAIL_LEVEL=HIGH but SCORE={score} <= 1",
        )


def _has_cross_diagnostic_confounders(confounders: str) -> bool:
    """Check if confounders text mentions depression/anxiety/fatigue."""
    lower = confounders.lower()
    return any(kw in lower for kw in _CROSS_DIAGNOSTIC_KEYWORDS)
