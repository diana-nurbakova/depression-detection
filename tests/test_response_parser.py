"""Tests for the LLM response parser."""

import pytest

from hipert.scoring.response_parser import parse_llm_response


def test_parse_well_formed_response() -> None:
    raw = """\
SYMPTOM_MATCH: YES
SELF_REFERENCE: DIRECT
DETAIL_LEVEL: MEDIUM
CONFOUNDERS: NONE
SCORE: 2
CONFIDENCE: 4
REASONING: Writer describes personal experience of conversational inattention with specific detail."""

    output, warnings = parse_llm_response(raw)
    assert output.symptom_match == "YES"
    assert output.self_reference == "DIRECT"
    assert output.detail_level == "MEDIUM"
    assert output.confounders == "NONE"
    assert output.score == 2
    assert output.confidence == 4
    assert "conversational inattention" in output.reasoning
    assert len(warnings) == 0


def test_parse_with_extra_text_before() -> None:
    raw = """\
Let me analyze this sentence carefully.

SYMPTOM_MATCH: NO
SELF_REFERENCE: NONE
DETAIL_LEVEL: NONE
CONFOUNDERS: NONE
SCORE: 0
CONFIDENCE: 5
REASONING: Not relevant to this symptom."""

    output, warnings = parse_llm_response(raw)
    assert output.symptom_match == "NO"
    assert output.score == 0
    assert output.confidence == 5


def test_parse_with_confounders_text() -> None:
    raw = """\
SYMPTOM_MATCH: PARTIAL
SELF_REFERENCE: DIRECT
DETAIL_LEVEL: LOW
CONFOUNDERS: Could be depression-related concentration difficulty or general fatigue
SCORE: 1
CONFIDENCE: 2
REASONING: Ambiguous case with multiple possible explanations."""

    output, warnings = parse_llm_response(raw)
    assert output.confounders == "Could be depression-related concentration difficulty or general fatigue"
    assert output.score == 1
    assert output.confidence == 2


def test_parse_missing_fields() -> None:
    raw = """\
SCORE: 1
CONFIDENCE: 3
REASONING: Some reasoning here."""

    output, warnings = parse_llm_response(raw)
    assert output.score == 1
    assert output.confidence == 3
    assert len(warnings) > 0  # Warnings for missing fields
    # Defaults applied for missing fields
    assert output.symptom_match == "NO"
    assert output.self_reference == "NONE"
    assert output.detail_level == "NONE"


def test_parse_invalid_score_clamped() -> None:
    raw = """\
SYMPTOM_MATCH: YES
SELF_REFERENCE: DIRECT
DETAIL_LEVEL: HIGH
CONFOUNDERS: NONE
SCORE: 5
CONFIDENCE: 4
REASONING: Test."""

    output, warnings = parse_llm_response(raw)
    assert output.score == 3  # Clamped to max valid


def test_parse_completely_malformed() -> None:
    raw = "This is not a structured response at all."

    output, warnings = parse_llm_response(raw)
    assert output.score == 0  # Default
    assert output.confidence == 1  # Default
    assert len(warnings) > 0


def test_parse_case_insensitive() -> None:
    raw = """\
symptom_match: Yes
self_reference: Direct
detail_level: Medium
confounders: None
score: 2
confidence: 4
reasoning: Case insensitive test."""

    output, warnings = parse_llm_response(raw)
    assert output.symptom_match == "YES"
    assert output.self_reference == "DIRECT"
    assert output.score == 2
