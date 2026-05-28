"""Tests for the combined-assessor aggregation path."""

import json
from pathlib import Path

from mentalriskes.tom_act import aggregator
from mentalriskes.tom_act.llama_regen import SIG_ASSESS_COMBINED


def _write_log(run_root: Path, signal_type: str, rec: dict):
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"{signal_type}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _combined_payload():
    return {
        "phq9": {"items": [{"item": i, "score": 1} for i in range(1, 10)]},
        "gad7": {"items": [{"item": i, "score": 2} for i in range(1, 8)]},
        "compact10": {"items": [{"item": i, "score": 3} for i in range(1, 11)]},
    }


def test_combined_assessor_aggregates_to_26_rows(tmp_path):
    _write_log(tmp_path, SIG_ASSESS_COMBINED, {
        "session_id": "S07", "round": 4, "candidate": None,
        "parse_success": True, "response_parsed": _combined_payload(),
    })
    df = aggregator.build_llama_assessors_long(tmp_path)
    assert len(df) == 26
    assert set(df["instrument"]) == {"PHQ-9", "GAD-7", "CompACT-10"}
    phq = df[df["instrument"] == "PHQ-9"]
    assert len(phq) == 9 and (phq["score"] == 1).all()
    assert "source" not in df.columns


def test_per_instrument_takes_precedence_over_combined(tmp_path):
    # Combined says CompACT item scores = 3; per-instrument says 5 -> per-instrument wins.
    _write_log(tmp_path, SIG_ASSESS_COMBINED, {
        "session_id": "S07", "round": 4, "candidate": None,
        "parse_success": True, "response_parsed": _combined_payload(),
    })
    _write_log(tmp_path, "llama_assess_compact10", {
        "session_id": "S07", "round": 4, "candidate": None,
        "parse_success": True, "response_parsed": {"CompACT-10": [5] * 10},
    })
    df = aggregator.build_llama_assessors_long(tmp_path)
    comp = df[df["instrument"] == "CompACT-10"]
    assert len(comp) == 10
    assert (comp["score"] == 5).all()   # per-instrument overrides combined
