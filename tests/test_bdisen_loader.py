"""Tests for BDI-Sen-2.0 dataset loader."""

from pathlib import Path

import pytest

from hipert.data.bdisen_loader import (
    BDISenAnnotation,
    filter_confounder_candidates,
    load_annotations,
    load_annotations_by_symptom,
)


@pytest.fixture
def sample_bdisen_jsonl(tmp_path: Path) -> Path:
    """Create a minimal bdi_majority_vote.jsonl for testing."""
    filepath = tmp_path / "bdi_majority_vote.jsonl"
    filepath.write_text(
        '{"sentence": "I can\'t focus on anything anymore", "symptom": "Concentration_difficulty", "severity": 1, "label": 1}\n'
        '{"sentence": "The weather is nice today", "symptom": "Concentration_difficulty", "severity": 0, "label": 0}\n'
        '{"sentence": "I feel so restless all the time", "symptom": "Agitation", "severity": 2, "label": 1}\n'
        '{"sentence": "People seem agitated these days", "symptom": "Agitation", "severity": 1, "label": 1}\n'
        '{"sentence": "I have no energy to do anything", "symptom": "Loss_of_energy", "severity": 3, "label": 1}\n'
        '{"sentence": "I just can\'t make up my mind about anything", "symptom": "Indecision", "severity": 1, "label": 1}\n'
        '{"sentence": "Great movie last night", "symptom": "Sadness", "severity": 0, "label": 0}\n',
        encoding="utf-8",
    )
    return filepath


class TestLoadAnnotations:

    def test_loads_all_rows(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        assert len(annotations) == 7

    def test_parses_fields_correctly(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        first = annotations[0]
        assert "focus" in first.sentence
        assert first.symptom == "Concentration_difficulty"
        assert first.severity == 1
        assert first.label == 1

    def test_severity_values(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        severities = {a.severity for a in annotations}
        assert severities == {0, 1, 2, 3}

    def test_label_values(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        labels = {a.label for a in annotations}
        assert labels == {0, 1}


class TestLoadAnnotationsBySymptom:

    def test_groups_correctly(self, sample_bdisen_jsonl: Path) -> None:
        grouped = load_annotations_by_symptom(sample_bdisen_jsonl)
        assert "Concentration_difficulty" in grouped
        assert "Agitation" in grouped
        assert len(grouped["Concentration_difficulty"]) == 2
        assert len(grouped["Agitation"]) == 2

    def test_all_symptoms_present(self, sample_bdisen_jsonl: Path) -> None:
        grouped = load_annotations_by_symptom(sample_bdisen_jsonl)
        expected = {"Concentration_difficulty", "Agitation", "Loss_of_energy", "Indecision", "Sadness"}
        assert set(grouped.keys()) == expected


class TestFirstPersonDetection:

    def test_first_person_present(self) -> None:
        ann = BDISenAnnotation(
            sentence="I can't focus on anything",
            symptom="X", severity=1, label=1,
        )
        assert ann.has_first_person is True

    def test_first_person_absent(self) -> None:
        ann = BDISenAnnotation(
            sentence="People seem agitated these days",
            symptom="X", severity=1, label=1,
        )
        assert ann.has_first_person is False


class TestFilterConfounderCandidates:

    def test_filters_by_symptom_and_label(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        candidates = filter_confounder_candidates(
            annotations, "Concentration_difficulty",
            severity_levels=(1,), require_first_person=False,
        )
        assert len(candidates) == 1
        assert candidates[0].label == 1
        assert candidates[0].severity == 1

    def test_filters_by_severity(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        # Agitation has severity=2 (label=1) and severity=1 (label=1)
        candidates = filter_confounder_candidates(
            annotations, "Agitation",
            severity_levels=(1, 2), require_first_person=False,
        )
        assert len(candidates) == 2

    def test_severity_1_only(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        candidates = filter_confounder_candidates(
            annotations, "Agitation",
            severity_levels=(1,), require_first_person=False,
        )
        assert len(candidates) == 1
        assert candidates[0].severity == 1

    def test_first_person_filter(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        # "People seem agitated these days" has no first-person
        candidates = filter_confounder_candidates(
            annotations, "Agitation",
            severity_levels=(1,), require_first_person=True,
        )
        assert len(candidates) == 0

    def test_first_person_filter_keeps_valid(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        # "I feel so restless all the time" has first-person
        candidates = filter_confounder_candidates(
            annotations, "Agitation",
            severity_levels=(2,), require_first_person=True,
        )
        assert len(candidates) == 1

    def test_empty_symptom(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        candidates = filter_confounder_candidates(
            annotations, "Nonexistent",
            severity_levels=(1,), require_first_person=False,
        )
        assert len(candidates) == 0

    def test_excludes_label_0(self, sample_bdisen_jsonl: Path) -> None:
        annotations = load_annotations(sample_bdisen_jsonl)
        # Concentration_difficulty has severity=0, label=0 — should be excluded
        candidates = filter_confounder_candidates(
            annotations, "Concentration_difficulty",
            severity_levels=(0, 1, 2, 3), require_first_person=False,
        )
        # Only severity=1, label=1 should remain
        assert all(c.label == 1 for c in candidates)


class TestRealData:
    """Tests against real BDI-Sen data. Skipped if not available."""

    DATA_PATH = Path("data/BDI-Sen/full_dataset/bdi_majority_vote.jsonl")

    def test_real_data_loads(self) -> None:
        if not self.DATA_PATH.exists():
            pytest.skip("BDI-Sen data not available")
        annotations = load_annotations(self.DATA_PATH)
        assert len(annotations) > 4000

    def test_real_concentration_candidates(self) -> None:
        if not self.DATA_PATH.exists():
            pytest.skip("BDI-Sen data not available")
        annotations = load_annotations(self.DATA_PATH)
        candidates = filter_confounder_candidates(
            annotations, "Concentration_difficulty",
            severity_levels=(1, 2, 3), require_first_person=True,
        )
        assert len(candidates) > 0

    def test_real_21_symptoms(self) -> None:
        if not self.DATA_PATH.exists():
            pytest.skip("BDI-Sen data not available")
        grouped = load_annotations_by_symptom(self.DATA_PATH)
        assert len(grouped) == 21
