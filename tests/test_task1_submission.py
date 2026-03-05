"""Tests for Task 1 submission formatting."""

import pytest

from erisk_task1.models import (
    ConversationTurn,
    PersonaResult,
    SeverityBand,
)
from erisk_task1.submission import format_interactions, format_results


def _make_result(persona_id: str, persona_number: int, total: int, band: SeverityBand, symptoms: list[str]) -> PersonaResult:
    return PersonaResult(
        persona_id=persona_id,
        persona_number=persona_number,
        conversation=[
            ConversationTurn(role="user", message="How are you?", turn_number=1),
            ConversationTurn(role="assistant", message="I'm okay.", turn_number=1),
        ],
        assessor_outputs={},
        linguistic_features_history=[],
        final_total=total,
        final_band=band,
        top_4_symptoms=symptoms,
    )


class TestFormatInteractions:
    def test_basic_format(self):
        results = [_make_result("00", 0, 5, SeverityBand.MINIMAL, [])]
        interactions = format_interactions(results)

        assert len(interactions) == 1
        assert interactions[0]["LLM"] == "1"  # 1-based string
        assert len(interactions[0]["conversation"]) == 2
        assert interactions[0]["conversation"][0]["role"] == "user"
        assert interactions[0]["conversation"][0]["message"] == "How are you?"
        assert interactions[0]["conversation"][1]["role"] == "assistant"

    def test_llm_id_is_string(self):
        results = [_make_result("04", 4, 15, SeverityBand.MILD, ["Sadness"])]
        interactions = format_interactions(results)
        assert interactions[0]["LLM"] == "5"  # 1-based

    def test_message_key_not_text(self):
        results = [_make_result("00", 0, 5, SeverityBand.MINIMAL, [])]
        interactions = format_interactions(results)
        conv = interactions[0]["conversation"][0]
        assert "message" in conv
        assert "text" not in conv


class TestFormatResults:
    def test_basic_format(self):
        results = [
            _make_result("00", 0, 37, SeverityBand.SEVERE, [
                "Self-criticalness", "Pessimism", "Loss of pleasure", "Loss of energy"
            ])
        ]
        erisk = format_results(results)

        assert len(erisk) == 1
        assert erisk[0]["LLM"] == "1"
        assert erisk[0]["bdi-score"] == 37
        assert len(erisk[0]["key-symptoms"]) == 4

    def test_bdi_score_is_integer(self):
        results = [_make_result("00", 0, 15, SeverityBand.MILD, ["Sadness"])]
        erisk = format_results(results)
        assert isinstance(erisk[0]["bdi-score"], int)

    def test_key_symptoms_uses_hyphen(self):
        results = [_make_result("00", 0, 15, SeverityBand.MILD, ["Sadness"])]
        erisk = format_results(results)
        assert "key-symptoms" in erisk[0]
        assert "key_symptoms" not in erisk[0]

    def test_minimal_has_empty_symptoms(self):
        results = [_make_result("00", 0, 5, SeverityBand.MINIMAL, [])]
        erisk = format_results(results)
        assert erisk[0]["key-symptoms"] == []

    def test_max_4_symptoms(self):
        results = [_make_result("00", 0, 30, SeverityBand.SEVERE, [
            "A", "B", "C", "D", "E"
        ])]
        erisk = format_results(results)
        assert len(erisk[0]["key-symptoms"]) <= 4

    def test_multiple_personas(self):
        results = [
            _make_result("00", 0, 5, SeverityBand.MINIMAL, []),
            _make_result("01", 1, 37, SeverityBand.SEVERE, ["Sadness", "Pessimism"]),
        ]
        erisk = format_results(results)
        assert len(erisk) == 2
        assert erisk[0]["LLM"] == "1"
        assert erisk[1]["LLM"] == "2"
