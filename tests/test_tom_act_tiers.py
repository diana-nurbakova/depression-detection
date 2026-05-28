"""Tests for tom_act.tiers and tier-aware candidate filtering."""

from mentalriskes.tom_act import tiers
from mentalriskes.tom_act.data import RoundData
from mentalriskes.tom_act.gemma_signals import _candidates_for


def _round(gold):
    return RoundData(session_id="S", round=1, patient_input="p",
                     therapist_response=None,
                     options={"option_1": "a", "option_2": "b", "option_3": "c"},
                     gold_option=gold)


def test_tier_definitions():
    assert set(tiers.TIERS) == {"T0", "T1", "T2", "T3"}
    assert tiers.TIERS["T1"].run_llama is True
    assert tiers.TIERS["T0"].run_llama is False
    assert tiers.TIERS["T0"].sessions == ("S07",)
    assert tiers.TIERS["T0"].max_rounds == 5
    # T3 codes only the rejected candidates.
    assert tiers.TIERS["T3"].candidate_filter == "rejected"
    assert set(tiers.TIERS["T3"].gemma_signals) == {"tom_stance", "presencia"}


def test_candidate_filter_gold():
    assert _candidates_for(_round(2), "gold") == [2]


def test_candidate_filter_rejected():
    assert sorted(_candidates_for(_round(2), "rejected")) == [1, 3]


def test_candidate_filter_all():
    assert sorted(_candidates_for(_round(2), "all")) == [1, 2, 3]


def test_candidate_filter_gold_missing():
    # No gold recorded -> nothing to code under gold filter.
    assert _candidates_for(_round(None), "gold") == []


def test_t0_pilot_call_budget():
    """T0 = 7 signal types/round, stance+presencia on gold only."""
    t0 = tiers.TIERS["T0"]
    sig = set(t0.gemma_signals)
    r = _round(2)
    views = len([s for s in ("self_a", "self_b", "observer_p", "observer_pt") if s in sig])
    tier_calls = 1 if "tom_tier_patient" in sig else 0
    cand = len(_candidates_for(r, t0.candidate_filter))
    per_round = views + tier_calls + cand * (("tom_stance" in sig) + ("presencia" in sig))
    assert per_round == 7   # 4 views + tier + 1 stance(gold) + 1 presencia(gold)
