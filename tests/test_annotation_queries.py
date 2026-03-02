"""Tests for annotation-specific query expansion."""

import pytest

from hipert.models import SymptomDefinition, SymptomFactor, SymptomSubcluster
from hipert.retrieval.annotation_queries import (
    build_all_annotation_queries,
    build_annotation_query,
    extract_l3_expressions,
)


@pytest.fixture
def sample_symptom() -> SymptomDefinition:
    return SymptomDefinition(
        item_number=5,
        text="How often do you fidget or squirm with your hands or feet when you have to sit for a long time?",
        factor=SymptomFactor.MOTOR_HI,
        subcluster=SymptomSubcluster.FIDGETING_RESTLESSNESS,
        clinical_definition="DSM-5-TR Criterion HI-a...",
        adult_manifestation="Leg bouncing...",
        discussion_topics=(
            '"My leg is always bouncing and I don\'t even notice."\n'
            '"I have to have something in my hands or I can\'t pay attention."\n'
            '"Sitting through a two-hour meeting is physically painful."'
        ),
        differential_markers="...",
        token_budget="minimal_2",
    )


@pytest.fixture
def symptom_no_l3() -> SymptomDefinition:
    return SymptomDefinition(
        item_number=99,
        text="Test symptom with no discussion topics.",
        factor=SymptomFactor.INATTENTION,
        subcluster=SymptomSubcluster.SUSTAINED_ATTENTION,
        clinical_definition="",
        adult_manifestation="",
        discussion_topics="",
        differential_markers="",
        token_budget="minimal_2",
    )


class TestExtractL3Expressions:
    def test_extracts_quoted_strings(self, sample_symptom):
        exprs = extract_l3_expressions(sample_symptom.discussion_topics)
        assert len(exprs) == 3
        assert "My leg is always bouncing" in exprs[0]
        assert "two-hour meeting" in exprs[2]

    def test_empty_string(self):
        assert extract_l3_expressions("") == []

    def test_no_quotes(self):
        assert extract_l3_expressions("plain text without quotes") == []

    def test_single_expression(self):
        exprs = extract_l3_expressions('"just one expression."')
        assert exprs == ["just one expression."]


class TestBuildAnnotationQuery:
    def test_includes_asrs_text(self, sample_symptom):
        query = build_annotation_query(sample_symptom)
        assert "fidget or squirm" in query

    def test_includes_l3_expressions(self, sample_symptom):
        query = build_annotation_query(sample_symptom)
        assert "My leg is always bouncing" in query
        assert "something in my hands" in query

    def test_respects_max_paraphrases(self, sample_symptom):
        query = build_annotation_query(sample_symptom, max_paraphrases=2)
        assert "My leg is always bouncing" in query
        assert "something in my hands" in query
        # Third expression should be excluded
        assert "two-hour meeting" not in query

    def test_max_paraphrases_1(self, sample_symptom):
        query = build_annotation_query(sample_symptom, max_paraphrases=1)
        assert "My leg is always bouncing" in query
        assert "something in my hands" not in query

    def test_no_l3_data(self, symptom_no_l3):
        query = build_annotation_query(symptom_no_l3)
        assert query == "Test symptom with no discussion topics."

    def test_max_paraphrases_exceeds_available(self, sample_symptom):
        # Asking for 10 but only 3 exist — should just use all 3
        query = build_annotation_query(sample_symptom, max_paraphrases=10)
        assert "My leg is always bouncing" in query
        assert "two-hour meeting" in query


class TestBuildAllAnnotationQueries:
    def test_builds_for_all_symptoms(self, sample_symptom, symptom_no_l3):
        symptoms = {5: sample_symptom, 99: symptom_no_l3}
        queries = build_all_annotation_queries(symptoms)
        assert set(queries.keys()) == {5, 99}
        assert "fidget" in queries[5]
        assert "Test symptom" in queries[99]

    def test_passes_max_paraphrases(self, sample_symptom):
        symptoms = {5: sample_symptom}
        queries = build_all_annotation_queries(symptoms, max_paraphrases=1)
        assert "My leg is always bouncing" in queries[5]
        assert "two-hour meeting" not in queries[5]
