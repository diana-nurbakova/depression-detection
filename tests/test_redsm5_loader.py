"""Tests for RedSM5 dataset loader."""

from pathlib import Path

import pytest

from hipert.data.redsm5_loader import (
    RedSM5Annotation,
    filter_confounder_candidates,
    load_annotations,
    load_annotations_by_symptom,
)


@pytest.fixture
def sample_redsm5_csv(tmp_path: Path) -> Path:
    """Create a minimal redsm5_annotations.csv for testing."""
    filepath = tmp_path / "redsm5_annotations.csv"
    filepath.write_text(
        'post_id,sentence_id,sentence_text,DSM5_symptom,status,explanation\n'
        's_100_5,s_100_5_0,"I can\'t concentrate on anything anymore",COGNITIVE_ISSUES,1,"Cognitive impairment related to depression"\n'
        's_100_5,s_100_5_1,"The weather was nice today",COGNITIVE_ISSUES,0,"Not relevant to cognitive issues"\n'
        's_200_3,s_200_3_0,"I feel so tired all the time",FATIGUE,1,"Fatigue associated with depressive episode"\n'
        's_200_3,s_200_3_1,"My friend told me about a movie",FATIGUE,0,"Not relevant"\n'
        's_300_1,s_300_1_0,"I just can\'t enjoy things like I used to",ANHEDONIA,1,"Loss of interest consistent with anhedonia"\n'
        's_400_2,s_400_2_0,"People move around a lot when they are anxious",PSYCHOMOTOR,1,"Psychomotor agitation, but no first-person"\n'
        's_500_1,s_500_1_0,"I can\'t sleep, my mind races all night",SLEEP_ISSUES,1,"Sleep disruption with racing thoughts"\n',
        encoding="utf-8",
    )
    return filepath


class TestLoadAnnotations:

    def test_loads_all_rows(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        assert len(annotations) == 7

    def test_parses_fields_correctly(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        first = annotations[0]
        assert first.post_id == "s_100_5"
        assert first.sentence_id == "s_100_5_0"
        assert "concentrate" in first.sentence_text
        assert first.dsm5_symptom == "COGNITIVE_ISSUES"
        assert first.status == 1
        assert "depression" in first.explanation.lower()

    def test_status_values(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        statuses = [a.status for a in annotations]
        assert set(statuses) == {0, 1}

    def test_handles_quoted_fields(self, sample_redsm5_csv: Path) -> None:
        """Verify CSV quoting handles commas and apostrophes in text."""
        annotations = load_annotations(sample_redsm5_csv)
        # "I can't concentrate" has an apostrophe
        assert any("can't" in a.sentence_text for a in annotations)


class TestLoadAnnotationsBySymptom:

    def test_groups_correctly(self, sample_redsm5_csv: Path) -> None:
        grouped = load_annotations_by_symptom(sample_redsm5_csv)
        assert "COGNITIVE_ISSUES" in grouped
        assert "FATIGUE" in grouped
        assert "ANHEDONIA" in grouped
        assert len(grouped["COGNITIVE_ISSUES"]) == 2
        assert len(grouped["FATIGUE"]) == 2

    def test_all_categories_present(self, sample_redsm5_csv: Path) -> None:
        grouped = load_annotations_by_symptom(sample_redsm5_csv)
        expected = {"COGNITIVE_ISSUES", "FATIGUE", "ANHEDONIA", "PSYCHOMOTOR", "SLEEP_ISSUES"}
        assert set(grouped.keys()) == expected


class TestFirstPersonDetection:

    def test_first_person_present(self) -> None:
        ann = RedSM5Annotation(
            post_id="p", sentence_id="s", dsm5_symptom="X",
            sentence_text="I can't concentrate on anything",
            status=1, explanation="",
        )
        assert ann.has_first_person is True

    def test_first_person_absent(self) -> None:
        ann = RedSM5Annotation(
            post_id="p", sentence_id="s", dsm5_symptom="X",
            sentence_text="People move around when anxious",
            status=1, explanation="",
        )
        assert ann.has_first_person is False


class TestFilterConfounderCandidates:

    def test_filters_by_status_1(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        candidates = filter_confounder_candidates(
            annotations, "COGNITIVE_ISSUES", require_first_person=False,
        )
        assert len(candidates) == 1
        assert candidates[0].status == 1

    def test_filters_by_category(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        candidates = filter_confounder_candidates(
            annotations, "FATIGUE", require_first_person=False,
        )
        assert all(c.dsm5_symptom == "FATIGUE" for c in candidates)

    def test_first_person_filter(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        # PSYCHOMOTOR has one status=1 but no first-person
        candidates = filter_confounder_candidates(
            annotations, "PSYCHOMOTOR", require_first_person=True,
        )
        assert len(candidates) == 0

    def test_first_person_filter_keeps_valid(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        candidates = filter_confounder_candidates(
            annotations, "COGNITIVE_ISSUES", require_first_person=True,
        )
        assert len(candidates) == 1
        assert candidates[0].has_first_person is True

    def test_empty_category(self, sample_redsm5_csv: Path) -> None:
        annotations = load_annotations(sample_redsm5_csv)
        candidates = filter_confounder_candidates(
            annotations, "WORTHLESSNESS", require_first_person=False,
        )
        assert len(candidates) == 0


class TestRealData:
    """Tests against real RedSM5 data. Skipped if data not available."""

    ANNOTATIONS_PATH = Path("data/RedSM5/redsm5_annotations.csv")

    def test_real_data_loads(self) -> None:
        if not self.ANNOTATIONS_PATH.exists():
            pytest.skip("RedSM5 data not available")
        annotations = load_annotations(self.ANNOTATIONS_PATH)
        assert len(annotations) > 100

    def test_cognitive_issues_has_candidates(self) -> None:
        if not self.ANNOTATIONS_PATH.exists():
            pytest.skip("RedSM5 data not available")
        annotations = load_annotations(self.ANNOTATIONS_PATH)
        candidates = filter_confounder_candidates(
            annotations, "COGNITIVE_ISSUES", require_first_person=True,
        )
        # Spec says ~119 total COGNITIVE_ISSUES, some should pass first-person
        assert len(candidates) > 0
