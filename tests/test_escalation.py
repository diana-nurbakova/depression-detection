"""Tests for escalation trigger logic."""

import pytest

from hipert.models import LLMOutput
from hipert.scoring.escalation import check_escalation


def _make_output(**kwargs) -> LLMOutput:
    defaults = {
        "symptom_match": "YES",
        "self_reference": "DIRECT",
        "detail_level": "MEDIUM",
        "confounders": "NONE",
        "score": 2,
        "confidence": 4,
        "reasoning": "Test",
        "raw_text": "",
    }
    defaults.update(kwargs)
    return LLMOutput(**defaults)


def test_rule1_low_confidence() -> None:
    output = _make_output(confidence=2)
    should, triggers = check_escalation(output, symptom_id=5)
    assert should
    assert any("Rule 1" in t for t in triggers)


def test_rule1_high_confidence_no_escalation() -> None:
    output = _make_output(confidence=5, score=3)
    should, triggers = check_escalation(output, symptom_id=5)
    # No Rule 1 trigger at confidence=5
    assert not any("Rule 1" in t for t in triggers)


def test_rule2_symptom_no_but_score_high() -> None:
    output = _make_output(symptom_match="NO", score=2)
    should, triggers = check_escalation(output, symptom_id=5)
    assert should
    assert any("Rule 2" in t and "SYMPTOM_MATCH=NO" in t for t in triggers)


def test_rule2_symptom_yes_but_score_zero() -> None:
    output = _make_output(symptom_match="YES", score=0)
    should, triggers = check_escalation(output, symptom_id=5)
    assert should
    assert any("Rule 2" in t and "SYMPTOM_MATCH=YES" in t for t in triggers)


def test_rule2_self_reference_none_but_score_positive() -> None:
    output = _make_output(self_reference="NONE", score=1)
    should, triggers = check_escalation(output, symptom_id=5)
    assert any("Rule 2" in t and "SELF_REFERENCE=NONE" in t for t in triggers)


def test_rule2_detail_none_but_score_high() -> None:
    output = _make_output(detail_level="NONE", score=2)
    should, triggers = check_escalation(output, symptom_id=5)
    assert any("Rule 2" in t and "DETAIL_LEVEL=NONE" in t for t in triggers)


def test_rule2_detail_high_but_score_low() -> None:
    output = _make_output(detail_level="HIGH", score=1)
    should, triggers = check_escalation(output, symptom_id=5)
    assert any("Rule 2" in t and "DETAIL_LEVEL=HIGH" in t for t in triggers)


def test_rule3_confounders_on_borderline() -> None:
    output = _make_output(
        confounders="Could be depression or anxiety", score=1, confidence=4,
    )
    should, triggers = check_escalation(output, symptom_id=1)
    assert should
    assert any("Rule 3" in t for t in triggers)


def test_rule3_confounders_on_clear_score_no_trigger() -> None:
    output = _make_output(
        confounders="Could be anxiety", score=3, confidence=5,
    )
    should, triggers = check_escalation(output, symptom_id=1)
    assert not any("Rule 3" in t for t in triggers)


def test_rule4_inattention_cross_diagnostic() -> None:
    output = _make_output(
        confounders="depression-related concentration difficulty",
        score=2, confidence=4,
    )
    # Item 9 is in the inattention cluster (7-11)
    should, triggers = check_escalation(output, symptom_id=9)
    assert should
    assert any("Rule 4" in t for t in triggers)


def test_rule4_non_inattention_no_trigger() -> None:
    output = _make_output(
        confounders="depression-related issue",
        score=2, confidence=4,
    )
    # Item 5 is Motor H/I, not inattention
    should, triggers = check_escalation(output, symptom_id=5)
    assert not any("Rule 4" in t for t in triggers)


def test_rule5_boundary_moderate_confidence() -> None:
    output = _make_output(score=1, confidence=3)
    should, triggers = check_escalation(output, symptom_id=15)
    assert should
    assert any("Rule 5" in t for t in triggers)


def test_no_escalation_clean_high_confidence() -> None:
    output = _make_output(
        symptom_match="YES",
        self_reference="DIRECT",
        detail_level="HIGH",
        confounders="NONE",
        score=3,
        confidence=5,
    )
    should, triggers = check_escalation(output, symptom_id=5)
    assert not should
    assert len(triggers) == 0


def test_multiple_triggers() -> None:
    output = _make_output(
        symptom_match="NO",
        self_reference="NONE",
        score=2,
        confidence=1,
        confounders="anxiety, depression",
    )
    should, triggers = check_escalation(output, symptom_id=9)
    assert should
    # Should have multiple rules triggered
    assert len(triggers) >= 3  # Rule 1, Rule 2 (multiple), Rule 3, Rule 4
