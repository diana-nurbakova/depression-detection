"""Tests for Task 1 scoring pipeline."""

import pytest

from erisk_task1.models import (
    AssessorOutput,
    ItemScore,
    ItemState,
    SeverityBand,
    score_to_band,
)
from erisk_task1.scoring import (
    collect_item_scores,
    compute_final_total,
    compute_preliminary_consensus,
    pass1_score,
    pass2_bayesian_prior,
    select_top4_mechanical,
)
from erisk_task1.linguistic import extract_features


def _make_item(item_id, name, score, confidence, state):
    return ItemScore(
        item_id=item_id,
        item_name=name,
        score=score,
        confidence=confidence,
        state=state,
    )


def _make_assessor_output(name, items):
    return AssessorOutput(assessor_name=name, items=items)


class TestPass1Score:
    def test_only_scored_items(self):
        scores = {
            1: _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
            4: _make_item(4, "Loss of pleasure", 1, 0.70, ItemState.SCORED),
            10: _make_item(10, "Crying", None, 0.0, ItemState.NO_EVIDENCE),
            12: _make_item(12, "Loss of interest", 0, 0.80, ItemState.EVIDENCE_OF_ABSENCE),
        }
        assert pass1_score(scores) == 3  # 2 + 1, skip null and 0

    def test_empty_scores(self):
        assert pass1_score({}) == 0

    def test_all_no_evidence(self):
        scores = {
            1: _make_item(1, "Sadness", None, 0.0, ItemState.NO_EVIDENCE),
            2: _make_item(2, "Pessimism", None, 0.0, ItemState.NO_EVIDENCE),
        }
        assert pass1_score(scores) == 0


class TestCollectItemScores:
    def test_flattens_assessor_outputs(self):
        outputs = {
            "AFFECTIVE": _make_assessor_output("AFFECTIVE", [
                _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
                _make_item(4, "Loss of pleasure", 1, 0.70, ItemState.SCORED),
            ]),
            "COGNITIVE": _make_assessor_output("COGNITIVE", [
                _make_item(2, "Pessimism", 3, 0.90, ItemState.SCORED),
            ]),
        }
        scores = collect_item_scores(outputs)
        assert 1 in scores
        assert 2 in scores
        assert 4 in scores
        assert scores[1].score == 2
        assert scores[2].score == 3


class TestPass2BayesianPrior:
    def test_skip_when_bands_agree(self):
        """Prior should NOT be applied when assessor band matches consensus."""
        scores = {
            1: _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
            10: _make_item(10, "Crying", None, 0.0, ItemState.NO_EVIDENCE),
        }
        result = pass2_bayesian_prior(
            scores, pass1_total=5,
            consensus_band=SeverityBand.MINIMAL,
            assessor_band=SeverityBand.MINIMAL,
        )
        # NO_EVIDENCE item should remain unchanged
        assert result[10].state == ItemState.NO_EVIDENCE

    def test_apply_when_bands_disagree(self):
        """Prior should be applied when assessor band differs from consensus."""
        scores = {
            1: _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
            10: _make_item(10, "Crying", None, 0.0, ItemState.NO_EVIDENCE),
        }
        result = pass2_bayesian_prior(
            scores, pass1_total=12,
            consensus_band=SeverityBand.MILD,
            assessor_band=SeverityBand.MINIMAL,
        )
        # Tier 2 item (Crying=10) should get prior of 1 for Mild consensus
        assert result[10].score == 1
        assert result[10].source == "prior"

    def test_tier3_gets_zero_for_mild(self):
        """Tier 3 items get prior 0 for Mild consensus."""
        scores = {
            6: _make_item(6, "Punishment feelings", None, 0.0, ItemState.NO_EVIDENCE),
        }
        result = pass2_bayesian_prior(
            scores, pass1_total=12,
            consensus_band=SeverityBand.MILD,
            assessor_band=SeverityBand.MINIMAL,
        )
        assert result[6].state == ItemState.NO_EVIDENCE  # Still NO_EVIDENCE (prior=0)

    def test_does_not_modify_scored_items(self):
        """Prior should not overwrite already-scored items."""
        scores = {
            1: _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
        }
        result = pass2_bayesian_prior(
            scores, pass1_total=12,
            consensus_band=SeverityBand.SEVERE,
            assessor_band=SeverityBand.MINIMAL,
        )
        assert result[1].score == 2  # Unchanged


class TestComputeFinalTotal:
    def test_sums_scored_and_evidence_of_absence(self):
        scores = {
            1: _make_item(1, "Sadness", 2, 0.85, ItemState.SCORED),
            2: _make_item(2, "Pessimism", 0, 0.80, ItemState.EVIDENCE_OF_ABSENCE),
            10: _make_item(10, "Crying", None, 0.0, ItemState.NO_EVIDENCE),
        }
        assert compute_final_total(scores) == 2  # 2 + 0


class TestTop4Selection:
    def test_selects_top_by_confidence_times_score(self):
        scores = {
            1: _make_item(1, "Sadness", 2, 0.90, ItemState.SCORED),
            2: _make_item(2, "Pessimism", 3, 0.80, ItemState.SCORED),
            4: _make_item(4, "Loss of pleasure", 1, 0.70, ItemState.SCORED),
            15: _make_item(15, "Loss of energy", 3, 0.85, ItemState.SCORED),
            16: _make_item(16, "Changes in sleeping pattern", 2, 0.75, ItemState.SCORED),
        }
        top4 = select_top4_mechanical(scores)
        assert len(top4) == 4

        # Item 15 (3*0.85=2.55) and Item 2 (3*0.80=2.40) should be top
        ids = [t.item_id for t in top4]
        assert 15 in ids
        assert 2 in ids

    def test_excludes_zero_scores(self):
        scores = {
            1: _make_item(1, "Sadness", 0, 0.80, ItemState.EVIDENCE_OF_ABSENCE),
            2: _make_item(2, "Pessimism", 1, 0.60, ItemState.SCORED),
        }
        top4 = select_top4_mechanical(scores)
        assert len(top4) == 1
        assert top4[0].item_id == 2

    def test_returns_fewer_than_4_if_not_enough_scored(self):
        scores = {
            1: _make_item(1, "Sadness", 1, 0.50, ItemState.SCORED),
        }
        top4 = select_top4_mechanical(scores)
        assert len(top4) == 1
