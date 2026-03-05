"""Tests for Task 1 prompt templates."""

import pytest

from erisk_task1.prompts import (
    PERSONA_SYSTEM_PROMPT,
    INTERVIEWER_SYSTEM_PROMPT,
    ASSESSOR_SHARED_PREAMBLE,
    ORCHESTRATOR_REASONING_PROMPT,
    JUSTIFICATOR_PROMPT,
    get_assessor_prompt,
)


class TestPersonaPrompt:
    def test_verbatim_content(self):
        assert "simulated patient" in PERSONA_SYSTEM_PROMPT
        assert "contextual realism" in PERSONA_SYSTEM_PROMPT
        assert "Do not mention you are an AI" in PERSONA_SYSTEM_PROMPT

    def test_no_clinical_terms(self):
        # The persona prompt should NOT contain clinical terms
        forbidden = ["depression", "BDI", "PHQ", "diagnosis", "screening"]
        for word in forbidden:
            assert word.lower() not in PERSONA_SYSTEM_PROMPT.lower()


class TestInterviewerPrompt:
    def test_contains_oars(self):
        assert "OARS" in INTERVIEWER_SYSTEM_PROMPT

    def test_forbidden_words_listed(self):
        assert "depression" in INTERVIEWER_SYSTEM_PROMPT
        assert "NEVER use these words" in INTERVIEWER_SYSTEM_PROMPT

    def test_topic_areas_listed(self):
        for topic in [
            "EMOTIONAL_STATE", "ACTIVITIES_INTERESTS", "DAILY_ROUTINE",
            "SELF_PERCEPTION", "FUTURE_OUTLOOK", "DECISION_MAKING",
        ]:
            assert topic in INTERVIEWER_SYSTEM_PROMPT

    def test_short_message_rule(self):
        assert "2-4 sentences" in INTERVIEWER_SYSTEM_PROMPT


class TestAssessorPrompts:
    def test_all_four_assessors_exist(self):
        for name in ["AFFECTIVE", "COGNITIVE", "SOMATIC", "FUNCTIONAL"]:
            prompt = get_assessor_prompt(name)
            assert len(prompt) > 100

    def test_shared_preamble_included(self):
        for name in ["AFFECTIVE", "COGNITIVE", "SOMATIC", "FUNCTIONAL"]:
            prompt = get_assessor_prompt(name)
            assert "SCORED" in prompt
            assert "NO_EVIDENCE" in prompt
            assert "EVIDENCE_OF_ABSENCE" in prompt

    def test_affective_items(self):
        prompt = get_assessor_prompt("AFFECTIVE")
        assert "Item 1: Sadness" in prompt
        assert "Item 4: Loss of Pleasure" in prompt
        assert "Item 17: Irritability" in prompt
        assert "Non-sadness depression" in prompt

    def test_cognitive_items(self):
        prompt = get_assessor_prompt("COGNITIVE")
        assert "Item 2: Pessimism" in prompt
        assert "Item 9: Suicidal Thoughts" in prompt
        assert "NEVER score above 1" in prompt

    def test_somatic_items(self):
        prompt = get_assessor_prompt("SOMATIC")
        assert "Item 16: Changes in Sleeping Pattern" in prompt
        assert "BIDIRECTIONAL" in prompt

    def test_functional_items(self):
        prompt = get_assessor_prompt("FUNCTIONAL")
        assert "Item 13: Indecisiveness" in prompt
        assert "Item 21: Loss of Interest in Sex" in prompt

    def test_invalid_assessor_raises(self):
        with pytest.raises(KeyError):
            get_assessor_prompt("INVALID")


class TestOrchestratorPrompt:
    def test_contains_key_elements(self):
        assert "IDENTIFY GAPS" in ORCHESTRATOR_REASONING_PROMPT
        assert "CONTINUE" in ORCHESTRATOR_REASONING_PROMPT
        assert "TERMINATE" in ORCHESTRATOR_REASONING_PROMPT

    def test_json_output_format(self):
        assert "valid JSON" in ORCHESTRATOR_REASONING_PROMPT


class TestJustificatorPrompt:
    def test_contains_patterns(self):
        assert "Pattern A" in JUSTIFICATOR_PROMPT
        assert "Pattern F" in JUSTIFICATOR_PROMPT

    def test_override_rules(self):
        assert "confidence < 0.5" in JUSTIFICATOR_PROMPT
        assert "NEVER adjust Item 9 upward" in JUSTIFICATOR_PROMPT

    def test_top4_selection_criteria(self):
        assert "centrality" in JUSTIFICATOR_PROMPT
        assert "specificity" in JUSTIFICATOR_PROMPT
