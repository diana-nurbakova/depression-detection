"""Tests for Task 1 data models."""

import pytest

from erisk_task1.models import (
    BDI_ITEMS,
    ASSESSOR_ITEMS,
    FAST_SCREEN_ITEMS,
    TIER_1_ITEMS,
    TIER_2_ITEMS,
    TIER_3_ITEMS,
    ItemScore,
    ItemState,
    SeverityBand,
    score_to_band,
)


class TestSeverityBands:
    def test_minimal_range(self):
        for score in [0, 5, 10, 13]:
            assert score_to_band(score) == SeverityBand.MINIMAL

    def test_mild_range(self):
        for score in [14, 15, 19]:
            assert score_to_band(score) == SeverityBand.MILD

    def test_moderate_range(self):
        for score in [20, 25, 28]:
            assert score_to_band(score) == SeverityBand.MODERATE

    def test_severe_range(self):
        for score in [29, 37, 40, 63]:
            assert score_to_band(score) == SeverityBand.SEVERE

    def test_boundary_13_14(self):
        assert score_to_band(13) == SeverityBand.MINIMAL
        assert score_to_band(14) == SeverityBand.MILD

    def test_boundary_19_20(self):
        assert score_to_band(19) == SeverityBand.MILD
        assert score_to_band(20) == SeverityBand.MODERATE

    def test_boundary_28_29(self):
        assert score_to_band(28) == SeverityBand.MODERATE
        assert score_to_band(29) == SeverityBand.SEVERE


class TestBDIItems:
    def test_21_items(self):
        assert len(BDI_ITEMS) == 21

    def test_item_ids_are_1_to_21(self):
        assert set(BDI_ITEMS.keys()) == set(range(1, 22))

    def test_canonical_names(self):
        assert BDI_ITEMS[1] == "Sadness"
        assert BDI_ITEMS[9] == "Suicidal thoughts or wishes"
        assert BDI_ITEMS[16] == "Changes in sleeping pattern"
        assert BDI_ITEMS[21] == "Loss of interest in sex"


class TestAssessorItems:
    def test_four_assessors(self):
        assert set(ASSESSOR_ITEMS.keys()) == {"AFFECTIVE", "COGNITIVE", "SOMATIC", "FUNCTIONAL"}

    def test_all_21_items_covered(self):
        all_items = set()
        for items in ASSESSOR_ITEMS.values():
            all_items.update(items)
        assert all_items == set(range(1, 22))

    def test_no_overlap(self):
        seen = set()
        for items in ASSESSOR_ITEMS.values():
            for item in items:
                assert item not in seen, f"Item {item} appears in multiple assessors"
                seen.add(item)

    def test_affective_items(self):
        assert ASSESSOR_ITEMS["AFFECTIVE"] == [1, 4, 10, 12, 17]

    def test_cognitive_items(self):
        assert ASSESSOR_ITEMS["COGNITIVE"] == [2, 3, 5, 6, 7, 8, 9, 14]

    def test_somatic_items(self):
        assert ASSESSOR_ITEMS["SOMATIC"] == [11, 15, 16, 18, 20]

    def test_functional_items(self):
        assert ASSESSOR_ITEMS["FUNCTIONAL"] == [13, 19, 21]


class TestTierClassification:
    def test_all_items_in_exactly_one_tier(self):
        for item_id in range(1, 22):
            tiers = sum([
                item_id in TIER_1_ITEMS,
                item_id in TIER_2_ITEMS,
                item_id in TIER_3_ITEMS,
            ])
            assert tiers == 1, f"Item {item_id} in {tiers} tiers"

    def test_tier3_items(self):
        assert TIER_3_ITEMS == {6, 9, 21}


class TestItemScore:
    def test_scored_state(self):
        item = ItemScore(
            item_id=1, item_name="Sadness", score=2,
            confidence=0.85, state=ItemState.SCORED,
        )
        assert item.score == 2
        assert item.state == ItemState.SCORED

    def test_no_evidence_state(self):
        item = ItemScore(
            item_id=10, item_name="Crying", score=None,
            confidence=0.0, state=ItemState.NO_EVIDENCE,
        )
        assert item.score is None

    def test_evidence_of_absence(self):
        item = ItemScore(
            item_id=1, item_name="Sadness", score=0,
            confidence=0.8, state=ItemState.EVIDENCE_OF_ABSENCE,
        )
        assert item.score == 0
        assert item.confidence == 0.8
