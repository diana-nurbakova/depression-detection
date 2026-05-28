"""Tests for tom_act.data — gold parsing, merge, cross-task alignment.

Uses the real released test data if present; otherwise builds a tiny fixture.
"""

import json
from pathlib import Path

import pytest

from mentalriskes.tom_act import data

REAL_T1 = Path("data/MentalRiskES-2026/test/task1/test/data")
REAL_T2 = Path("data/MentalRiskES-2026/test/task2/test/data")
REAL_GOLD = Path("data/MentalRiskES-2026/test/task2/test/gold")
REAL_SESSION_GOLD = Path("data/MentalRiskES-2026/test/task1/test/gold_label.json")

_real = REAL_T1.exists() and REAL_T2.exists() and REAL_GOLD.exists()


def test_parse_gold_option():
    assert data.parse_gold_option("option_2") == 2
    assert data.parse_gold_option(3) == 3
    assert data.parse_gold_option("option_1") == 1
    assert data.parse_gold_option(None) is None


@pytest.mark.skipif(not _real, reason="released test data not present")
def test_load_real_sessions_counts():
    sessions = data.load_sessions(REAL_T1, REAL_T2, REAL_GOLD, REAL_SESSION_GOLD,
                                  ["S07", "S09"])
    assert sessions["S07"].n_rounds == 30
    assert sessions["S09"].n_rounds == 67
    assert len(sessions["S07"].gold_compact10) == 10


@pytest.mark.skipif(not _real, reason="released test data not present")
def test_cross_task_alignment_s01():
    """task2 round_t gold option text == task1 round_{t+1}.therapist_response."""
    sessions = data.load_sessions(REAL_T1, REAL_T2, REAL_GOLD, REAL_SESSION_GOLD, ["S01"])
    s = sessions["S01"]
    r1 = s.round(1)
    assert r1.gold_option is not None
    delivered_next = s.round(2).therapist_response
    assert r1.gold_option_text == delivered_next


@pytest.mark.skipif(not _real, reason="released test data not present")
def test_context_builders():
    sessions = data.load_sessions(REAL_T1, REAL_T2, REAL_GOLD, REAL_SESSION_GOLD, ["S07"])
    s = sessions["S07"]
    assert data.patient_turn(s, 1) == s.round(1).patient_input
    cp = data.cumulative_patient(s, 3)
    assert cp.count("PACIENTE") == 3
    cd = data.cumulative_dialogue(s, 3)
    # rounds 2,3 have a therapist turn; round 1 does not -> 2 therapist + 3 patient
    assert cd.count("TERAPEUTA") == 2
    assert cd.count("PACIENTE") == 3


def test_merge_with_fixture(tmp_path):
    t1 = tmp_path / "t1"; t2 = tmp_path / "t2"; gold = tmp_path / "gold"
    for d in (t1, t2, gold):
        d.mkdir()
    (t1 / "round_1.json").write_text(json.dumps(
        {"SX": {"round": 1, "patient_input": "hola"}}), encoding="utf-8")
    (t1 / "round_2.json").write_text(json.dumps(
        {"SX": {"round": 2, "therapist_response": "T2", "patient_input": "p2"}}), encoding="utf-8")
    (t2 / "round_1.json").write_text(json.dumps(
        {"SX": {"round": 1, "patient_input": "hola",
                "option_1": "a", "option_2": "b", "option_3": "c"}}), encoding="utf-8")
    (t2 / "round_2.json").write_text(json.dumps(
        {"SX": {"round": 2, "patient_input": "p2",
                "option_1": "T2", "option_2": "x", "option_3": "y"}}), encoding="utf-8")
    (gold / "round_1_gold.json").write_text(json.dumps(
        {"SX": {"correct_option": "option_2"}}), encoding="utf-8")
    (gold / "round_2_gold.json").write_text(json.dumps(
        {"SX": {"correct_option": "option_1"}}), encoding="utf-8")
    sg = tmp_path / "gold_label.json"
    sg.write_text(json.dumps({"SX": {"PHQ-9": [0]*9, "GAD-7": [0]*7,
                                     "CompACT-10": [0]*10}}), encoding="utf-8")

    sessions = data.load_sessions(t1, t2, gold, sg, ["SX"])
    s = sessions["SX"]
    assert s.n_rounds == 2
    assert s.round(1).gold_option == 2
    assert s.round(1).gold_option_text == "b"
    assert s.round(2).therapist_response == "T2"   # delivered before patient turn 2
