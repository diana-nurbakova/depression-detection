"""Tests for the annotation candidate retrieval pipeline."""

import csv
import json
from pathlib import Path

import pytest

from hipert.models import Sentence, SymptomDefinition, SymptomFactor, SymptomSubcluster

# Import functions under test
from scripts.retrieve_candidates import (
    build_score0_pool,
    create_annotation_templates,
    select_score0_for_symptom,
    write_score0_pool_tsv,
    write_symptom_candidates_tsv,
)

import random


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sentence(docno: str, text: str, file_id: str = "test") -> Sentence:
    return Sentence(
        docno=docno, pre="", text=text, post="", file_id=file_id,
    )


@pytest.fixture
def first_person_sentences() -> list[Sentence]:
    """Corpus of 10 sentences, 7 with first-person markers."""
    return [
        _make_sentence("u1_0_0", "I can't focus on anything."),              # 0 - first person
        _make_sentence("u1_0_1", "The weather is nice today."),               # 1 - no first person
        _make_sentence("u2_0_0", "My brain never shuts off."),                # 2 - first person
        _make_sentence("u2_0_1", "This is a random sentence."),               # 3 - no first person
        _make_sentence("u3_0_0", "I keep losing my keys everywhere."),        # 4 - first person
        _make_sentence("u3_0_1", "The dog ran across the yard."),             # 5 - no first person
        _make_sentence("u4_0_0", "I'm always bouncing my leg."),              # 6 - first person
        _make_sentence("u4_0_1", "I forget to eat sometimes."),               # 7 - first person
        _make_sentence("u5_0_0", "I've tried every planner app."),            # 8 - first person
        _make_sentence("u5_0_1", "I can never sit still in meetings."),       # 9 - first person
    ]


@pytest.fixture
def sample_symptoms() -> dict[int, SymptomDefinition]:
    return {
        1: SymptomDefinition(
            item_number=1,
            text="How often do you have trouble wrapping up the final details?",
            factor=SymptomFactor.INATTENTION,
            subcluster=SymptomSubcluster.ORGANIZATION_PLANNING,
            clinical_definition="DSM-5-TR...",
            adult_manifestation="Projects sit at 90%...",
            discussion_topics='"I can start but can\'t finish."',
            differential_markers="Depression...",
            token_budget="compressed_3",
        ),
        5: SymptomDefinition(
            item_number=5,
            text="How often do you fidget?",
            factor=SymptomFactor.MOTOR_HI,
            subcluster=SymptomSubcluster.FIDGETING_RESTLESSNESS,
            clinical_definition="DSM-5-TR...",
            adult_manifestation="Leg bouncing...",
            discussion_topics='"My leg is always bouncing."',
            differential_markers="...",
            token_budget="minimal_2",
        ),
    }


# ---------------------------------------------------------------------------
# Tests: write_symptom_candidates_tsv
# ---------------------------------------------------------------------------


class TestWriteSymptomCandidatesTsv:
    def test_creates_file(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "symptom_01_candidates.tsv"
        retrieval = [
            (first_person_sentences[0], 1, 0.85),
            (first_person_sentences[2], 2, 0.72),
        ]
        score0 = [first_person_sentences[7]]

        total = write_symptom_candidates_tsv(filepath, retrieval, score0)

        assert filepath.exists()
        assert total == 3

    def test_tsv_columns(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "symptom_01_candidates.tsv"
        retrieval = [(first_person_sentences[0], 1, 0.85)]
        score0 = [first_person_sentences[7]]

        write_symptom_candidates_tsv(filepath, retrieval, score0)

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            assert set(reader.fieldnames) == {
                "rank", "docno", "pre", "text", "post", "source",
            }

    def test_source_labels(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "sub" / "symptom_01_candidates.tsv"
        retrieval = [(first_person_sentences[0], 1, 0.85)]
        score0 = [first_person_sentences[7]]

        write_symptom_candidates_tsv(filepath, retrieval, score0)

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["source"] == "retrieval"
        assert rows[1]["source"] == "random"

    def test_rank_continuity(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "symptom_01_candidates.tsv"
        retrieval = [
            (first_person_sentences[0], 1, 0.85),
            (first_person_sentences[2], 2, 0.72),
        ]
        score0 = [first_person_sentences[7], first_person_sentences[8]]

        write_symptom_candidates_tsv(filepath, retrieval, score0)

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            ranks = [int(row["rank"]) for row in reader]

        assert ranks == [1, 2, 3, 4]

    def test_empty_retrieval(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "symptom_01_candidates.tsv"
        score0 = [first_person_sentences[7]]

        total = write_symptom_candidates_tsv(filepath, [], score0)
        assert total == 1


# ---------------------------------------------------------------------------
# Tests: write_score0_pool_tsv
# ---------------------------------------------------------------------------


class TestWriteScore0PoolTsv:
    def test_creates_file(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "score0_pool.tsv"
        pool = first_person_sentences[:3]

        write_score0_pool_tsv(filepath, pool)

        assert filepath.exists()

    def test_correct_columns(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "score0_pool.tsv"
        write_score0_pool_tsv(filepath, first_person_sentences[:2])

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            assert set(reader.fieldnames) == {"docno", "pre", "text", "post"}

    def test_row_count(self, tmp_path, first_person_sentences):
        filepath = tmp_path / "score0_pool.tsv"
        write_score0_pool_tsv(filepath, first_person_sentences[:4])

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            assert len(list(reader)) == 4


# ---------------------------------------------------------------------------
# Tests: build_score0_pool
# ---------------------------------------------------------------------------


class TestBuildScore0Pool:
    def test_excludes_top200_indices(self, first_person_sentences):
        # Exclude indices 0, 2, 4 (all first-person)
        exclusion = {0, 2, 4}
        pool = build_score0_pool(
            first_person_sentences, exclusion, pool_size=3,
            rng=random.Random(42),
        )

        pool_docnos = {s.docno for s in pool}
        # Excluded sentences should not be in pool
        assert "u1_0_0" not in pool_docnos  # index 0
        assert "u2_0_0" not in pool_docnos  # index 2
        assert "u3_0_0" not in pool_docnos  # index 4

    def test_only_first_person(self, first_person_sentences):
        pool = build_score0_pool(
            first_person_sentences, set(), pool_size=20,
            rng=random.Random(42),
        )

        for sent in pool:
            assert sent.has_first_person

    def test_respects_pool_size(self, first_person_sentences):
        pool = build_score0_pool(
            first_person_sentences, set(), pool_size=3,
            rng=random.Random(42),
        )
        assert len(pool) == 3

    def test_pool_size_exceeds_eligible(self, first_person_sentences):
        # With all 10 sentences, 7 are first-person. Asking for 20 should
        # return all 7.
        pool = build_score0_pool(
            first_person_sentences, set(), pool_size=20,
            rng=random.Random(42),
        )
        assert len(pool) == 7


# ---------------------------------------------------------------------------
# Tests: select_score0_for_symptom
# ---------------------------------------------------------------------------


class TestSelectScore0ForSymptom:
    def test_returns_requested_count(self, first_person_sentences):
        shared_pool = [first_person_sentences[6], first_person_sentences[7],
                       first_person_sentences[8], first_person_sentences[9]]
        exclusion = {0, 2}

        result = select_score0_for_symptom(
            first_person_sentences, exclusion, shared_pool, count=3,
            rng=random.Random(42),
        )
        assert len(result) == 3

    def test_prefers_shared_pool(self, first_person_sentences):
        shared_pool = [first_person_sentences[6], first_person_sentences[7]]
        exclusion = set()

        result = select_score0_for_symptom(
            first_person_sentences, exclusion, shared_pool, count=2,
            rng=random.Random(42),
        )

        result_docnos = {s.docno for s in result}
        pool_docnos = {s.docno for s in shared_pool}
        assert result_docnos.issubset(pool_docnos)

    def test_supplements_when_pool_insufficient(self, first_person_sentences):
        # Pool only has 1 sentence, but we need 3
        shared_pool = [first_person_sentences[6]]
        exclusion = set()

        result = select_score0_for_symptom(
            first_person_sentences, exclusion, shared_pool, count=3,
            rng=random.Random(42),
        )
        assert len(result) == 3
        # First result should be from the pool
        assert first_person_sentences[6] in result


# ---------------------------------------------------------------------------
# Tests: create_annotation_templates
# ---------------------------------------------------------------------------


class TestCreateAnnotationTemplates:
    def test_creates_per_symptom_files(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        assert (tmp_path / "symptom_01_examples.json").exists()
        assert (tmp_path / "symptom_05_examples.json").exists()

    def test_json_schema(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        with open(tmp_path / "symptom_01_examples.json") as f:
            data = json.load(f)

        assert data["symptom_id"] == 1
        assert data["symptom_factor"] == "Inattention"
        assert len(data["examples"]) == 4
        scores = [ex["score"] for ex in data["examples"]]
        assert scores == [0, 1, 2, 3]

    def test_annotation_fields(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        with open(tmp_path / "symptom_05_examples.json") as f:
            data = json.load(f)

        example = data["examples"][0]
        ann = example["annotation"]
        assert "symptom_match" in ann
        assert "self_reference" in ann
        assert "detail_level" in ann
        assert "confounders" in ann
        assert "score" in ann
        assert "confidence" in ann
        assert "reasoning" in ann

    def test_summary_tsv(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        summary_path = tmp_path / "annotation_summary.tsv"
        assert summary_path.exists()

        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2
        assert "symptom_id" in reader.fieldnames
        assert "score_0_docno" in reader.fieldnames
        assert "synthetic_count" in reader.fieldnames

    def test_synthetic_examples_file(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        with open(tmp_path / "synthetic_examples.json") as f:
            data = json.load(f)

        assert data == []

    def test_score0_shared_pool_file(self, tmp_path, sample_symptoms):
        create_annotation_templates(tmp_path, sample_symptoms)

        with open(tmp_path / "score0_shared_pool.json") as f:
            data = json.load(f)

        assert "constraint" in data
        assert "sentences" in data
        assert data["sentences"] == []
