"""Extract candidate examples for §2.2 Figure 1 + §5.2/§5.3 round-trace boxes.

Driven by `specs/MentalRiskES/example-candidates-extraction-spec.md`.

Outputs a single Markdown candidate-table document at
`analysis/MentalRiskES_test/outputs/example_candidates.md` and a sidecar
JSON file `outputs/example_candidates_raw.json` for downstream tooling.

Run:
    python analysis/MentalRiskES_test/extract_example_candidates.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "MentalRiskES-2026"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_T1_ROUNDS = DATA / "test" / "task1" / "test" / "data"
TEST_T1_GOLD = DATA / "test" / "task1" / "test" / "gold_label.json"
TEST_T2_ROUNDS = DATA / "test" / "task2" / "test" / "data"
TEST_T2_GOLD = DATA / "test" / "task2" / "test" / "gold"

TRIAL_T1_ROUNDS = DATA / "task1_trial" / "data"
TRIAL_T2_ROUNDS = DATA / "task2_trial" / "data"

REPLAY_T1_PRED = ROOT / "output" / "mentalriskes_test_replay" / "predictions"

ACTALK_RUN_LOGS = {
    "run0": ROOT / "output" / "mentalriskes" / "logs" / "predictions_run0_A5.jsonl",
    "run1": ROOT / "output" / "mentalriskes" / "logs" / "predictions_run1_A3.jsonl",
    "run2": ROOT / "output" / "mentalriskes" / "logs" / "predictions_run2_A1.jsonl",
}

TRIAL_RESULTS = ROOT / "output" / "mentalriskes" / "trial_results_all.json"

TRIAL_TASK2_RUN_FILES = {
    # run0 = PERM voting, run1 = FUNC fixed, run2 = HYB B+ fixed
    "run0": ROOT / "output" / "mentalriskes_task2" / "ablation" /
            "B_Llama-3.3-70B-Instruct-Turbo_es_FUNC_PERM_W3.jsonl",
    "run1": ROOT / "output" / "mentalriskes_task2" / "ablation" /
            "B_Llama-3.3-70B-Instruct-Turbo_es_FUNC_FIX_W3.jsonl",
    "run2": ROOT / "output" / "mentalriskes_task2" / "ablation" /
            "B+_Llama-3.3-70B-Instruct-Turbo_es_HYB_FIX_W3.jsonl",
}

TRIAL_GROUND_TRUTH = {
    1: 2, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 3, 8: 3, 9: 3,
    10: 2, 11: 1, 12: 3, 13: 1, 14: 3, 15: 1, 16: 1, 17: 3, 18: 2,
}

# ============================================================================
# PHQ-9 / GAD-7 item descriptions (English)
# ============================================================================

PHQ9_ITEMS = {
    1: "Anhedonia (little interest or pleasure in doing things)",
    2: "Depressed mood (feeling down, depressed, or hopeless)",
    3: "Sleep disturbance",
    4: "Fatigue / low energy",
    5: "Appetite change",
    6: "Self-worth (feeling like a failure or letting family down)",
    7: "Concentration",
    8: "Psychomotor (slow or fidgety)",
    9: "Suicidality / self-harm",
}
GAD7_ITEMS = {
    1: "Nervousness / on edge",
    2: "Inability to control worry",
    3: "Excessive worry about different things",
    4: "Trouble relaxing",
    5: "Restlessness (so restless hard to sit still)",
    6: "Irritability",
    7: "Dread (afraid as if something terrible will happen)",
}
COMPACT10_ITEMS = {
    1: "Item 1 BA (rushing through activities)",
    2: "Item 2 VA (acting on values)",
    3: "Item 3 OtE (suppressing thoughts/feelings — reversed)",
    4: "Item 4 VA (clear values)",
    5: "Item 5 OtE (avoiding emotional pain — reversed)",
    6: "Item 6 BA (autopilot)",
    7: "Item 7 VA (persistence in valued direction)",
    8: "Item 8 OtE (emotional suppression — reversed)",
    9: "Item 9 BA (lack of present-moment focus)",
    10: "Item 10 VA (perseverance towards goals)",
}

# CompACT-10: 0..6 per item (0 = strongly disagree, 6 = strongly agree).
# OtE items (3, 5, 8) are reversed in scoring (high = avoidance = inflexibility).
COMPACT_OTE = {3, 5, 8}
COMPACT_BA = {1, 6, 9}
COMPACT_VA = {2, 4, 7, 10}

# ============================================================================
# Severity band mappings
# ============================================================================

def phq9_band(total: int) -> str:
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    if total <= 19:
        return "moderately_severe"
    return "severe"


def gad7_band(total: int) -> str:
    if total <= 4:
        return "minimal"
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def phq9_cell_band(total: int) -> str | None:
    """Maps PHQ-9 total to the Figure-1 cells: mild / moderate / severe.

    Per the spec, figure has only 3 PHQ-9 cells; we fold moderately_severe and
    severe into the spec's 'severe' cell.
    """
    if total <= 4:
        return None  # minimal — not a Figure-1 cell
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def gad7_cell_band(total: int) -> str | None:
    if total <= 4:
        return None
    if total <= 9:
        return "mild"
    if total <= 14:
        return "moderate"
    return "severe"


def compact_cell_band(total: int) -> str:
    """Empirical thirds of 0..60 — low / medium / high (informal)."""
    if total <= 20:
        return "low"
    if total <= 40:
        return "medium"
    return "high"


# ============================================================================
# Loading helpers
# ============================================================================

def load_test_t1_gold() -> dict:
    with open(TEST_T1_GOLD, encoding="utf-8") as f:
        return json.load(f)


def load_test_t1_rounds() -> dict[int, dict]:
    """{round_n: {session_id: patient_input}}"""
    out: dict[int, dict[str, str]] = {}
    for path in sorted(TEST_T1_ROUNDS.glob("round_*.json")):
        rn = int(path.stem.split("_")[1])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[rn] = {sid: blob["patient_input"] for sid, blob in data.items()}
    return out


def load_actalk_predictions(run: str) -> dict[tuple[str, int], dict]:
    """{(session_id, round): full prediction record}"""
    path = ACTALK_RUN_LOGS[run]
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[(rec["session_id"], rec["round"])] = rec
    return out


def load_replay_t1_run(run_idx: int) -> dict[tuple[str, int], dict]:
    """{(session_id, round): {'phq9':..., 'gad7':..., 'compact10':...}}"""
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(REPLAY_T1_PRED.glob(f"round*_run{run_idx}.json")):
        rn = int(path.stem.replace(f"_run{run_idx}", "").replace("round", ""))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            for pred in entry.get("predictions", []):
                p = pred["prediction"]
                out[(pred["id"], rn)] = {
                    "phq9": p["PHQ-9"],
                    "gad7": p["GAD-7"],
                    "compact10": p["CompACT-10"],
                }
    return out


def load_trial_t1_patient_turns() -> dict[int, str]:
    """{round_n: patient_turn}"""
    out: dict[int, str] = {}
    for path in sorted(TRIAL_T1_ROUNDS.glob("round_*.json")):
        rn = int(path.stem.split("_")[1])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[rn] = data["trial"]["patient_input"]
    return out


def load_trial_t2_rounds() -> dict[int, dict]:
    """{round_n: {patient_input, option_1, option_2, option_3}}"""
    out: dict[int, dict] = {}
    for path in sorted(TRIAL_T2_ROUNDS.glob("round_*.json")):
        rn = int(path.stem.split("_")[1])
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[rn] = data["trial"]
    return out


def load_trial_t2_runs() -> dict[str, dict[int, dict]]:
    """{run: {round_n: round_record}} for trial Task 2."""
    out: dict[str, dict[int, dict]] = {}
    for run, path in TRIAL_TASK2_RUN_FILES.items():
        if not path.exists():
            out[run] = {}
            continue
        rounds: dict[int, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "round":
                    rounds[obj["round_id"]] = obj
        out[run] = rounds
    return out


def load_trial_t1_runs() -> dict:
    with open(TRIAL_RESULTS, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Final-round predicted totals per session/instrument/run
# ============================================================================

def compute_session_final_predictions(
    run_preds: dict[tuple[str, int], dict],
    session_id: str,
) -> dict | None:
    """Return prediction at the largest available round for this session.

    Falls back to None if no predictions for this session.
    """
    rounds = [(sid, rn) for (sid, rn) in run_preds.keys() if sid == session_id]
    if not rounds:
        return None
    last = max(rn for (_, rn) in rounds)
    return run_preds[(session_id, last)]


# ============================================================================
# Slot 1: §2.2 Figure 1 candidate extraction
# ============================================================================

def slot1_extract_candidates() -> dict:
    gold = load_test_t1_gold()
    rounds = load_test_t1_rounds()

    # Limit to the 10 sessions that appear in the released test data
    sessions_in_test: set[str] = set()
    for round_data in rounds.values():
        sessions_in_test.update(round_data.keys())
    print(f"[slot1] sessions in test data: {sorted(sessions_in_test)}", file=sys.stderr)

    # ACTalk run predictions — both replay (full coverage) and submitted (R1-30)
    replay_preds = {ri: load_replay_t1_run(ri) for ri in (0, 1, 2)}
    submitted_preds = {run: load_actalk_predictions(run) for run in ("run0", "run1", "run2")}

    # Per-session gold totals
    session_summary: dict[str, dict] = {}
    for sid in sorted(sessions_in_test):
        if sid not in gold:
            continue
        g = gold[sid]
        phq_total = sum(g["PHQ-9"])
        gad_total = sum(g["GAD-7"])
        cpt_total = sum(g["CompACT-10"])
        session_summary[sid] = {
            "phq9_gold_total": phq_total,
            "gad7_gold_total": gad_total,
            "compact_gold_total": cpt_total,
            "phq9_band_full": phq9_band(phq_total),
            "gad7_band_full": gad7_band(gad_total),
            "phq9_cell": phq9_cell_band(phq_total),
            "gad7_cell": gad7_cell_band(gad_total),
            "compact_cell": compact_cell_band(cpt_total),
            "phq9_gold_items": g["PHQ-9"],
            "gad7_gold_items": g["GAD-7"],
            "compact_gold_items": g["CompACT-10"],
        }
        # Add per-run final totals (replay where available; falls back to submitted)
        for run_idx, run_key in [(0, "run0"), (1, "run1"), (2, "run2")]:
            pred = compute_session_final_predictions(replay_preds[run_idx], sid)
            if pred is None:
                pred = compute_session_final_predictions(submitted_preds[run_key], sid)
                pred = (
                    {"phq9": pred["phq9"], "gad7": pred["gad7"], "compact10": pred["compact10"]}
                    if pred else None
                )
            if pred is None:
                session_summary[sid][f"{run_key}_phq9_total"] = None
                session_summary[sid][f"{run_key}_gad7_total"] = None
                session_summary[sid][f"{run_key}_compact_total"] = None
            else:
                session_summary[sid][f"{run_key}_phq9_total"] = sum(pred["phq9"])
                session_summary[sid][f"{run_key}_gad7_total"] = sum(pred["gad7"])
                session_summary[sid][f"{run_key}_compact_total"] = sum(pred["compact10"])

    # For per-turn item-evidence, use Run 2's step_1_detection (where present)
    run2_preds = submitted_preds["run2"]

    def step1_status(rec: dict, instrument_key: str, item_idx: int) -> tuple[str, str]:
        """Return (status, evidence_text) for item_idx (1-based) on given instrument."""
        steps = rec.get(f"{instrument_key}_steps", {})
        det = steps.get("step_1_detection", {})
        item = det.get(f"item_{item_idx}", {})
        return item.get("status", ""), item.get("evidence", "")

    def step2_score(rec: dict, instrument_key: str, item_idx: int) -> int | None:
        steps = rec.get(f"{instrument_key}_steps", {})
        det = steps.get("step_2_temporal" if instrument_key != "CompACT-10" else "step_2_endorsement", {})
        item = det.get(f"item_{item_idx}", {})
        return item.get("score") if isinstance(item, dict) else None

    # Candidate harvest: for each (instrument, cell), look at the relevant sessions
    PHQ_CELLS = ["mild", "moderate", "severe"]
    GAD_CELLS = ["mild", "moderate", "severe"]
    COMPACT_CELLS = ["low", "medium", "high"]

    def count_sentences(text: str) -> int:
        # crude sentence boundary
        return sum(text.count(p) for p in ".!?")

    candidates: dict[str, list[dict]] = defaultdict(list)

    # PHQ-9
    for cell in PHQ_CELLS:
        target_sessions = [s for s, info in session_summary.items() if info["phq9_cell"] == cell]
        for sid in target_sessions:
            sess = session_summary[sid]
            for rn in range(1, 83):
                turn = rounds.get(rn, {}).get(sid)
                if not turn or len(turn.split()) < 5 or len(turn.split()) > 100:
                    continue
                if count_sentences(turn) > 4:
                    continue
                rec = run2_preds.get((sid, rn))
                if rec is None:
                    continue
                # Find items marked present
                present_items: list[tuple[int, str, int | None]] = []
                for ii in range(1, 10):
                    status, ev = step1_status(rec, "PHQ-9", ii)
                    score = step2_score(rec, "PHQ-9", ii) or 0
                    if status == "present" and score >= 1:
                        present_items.append((ii, ev, score))
                if not present_items:
                    continue
                # Strongest-signal item (highest score)
                present_items.sort(key=lambda t: (t[2] or 0), reverse=True)
                target_item, ev, score = present_items[0]
                # Anti-criteria: skip if multi-instrument coactivation in same turn
                multi = False
                for ii in range(1, 8):
                    s, _ = step1_status(rec, "GAD-7", ii)
                    if s == "present":
                        multi = True
                        break
                # Track all signals
                candidates[f"PHQ9_{cell}"].append({
                    "cell_id": f"PHQ9_{cell}",
                    "session_id": sid,
                    "round_n": rn,
                    "patient_turn_es": turn,
                    "target_item_idx": target_item,
                    "target_item_desc": PHQ9_ITEMS[target_item],
                    "target_item_score_run2": score,
                    "evidence_run2": ev,
                    "gold_total": sess["phq9_gold_total"],
                    "gold_band": sess["phq9_band_full"],
                    "run0_total": sess.get("run0_phq9_total"),
                    "run1_total": sess.get("run1_phq9_total"),
                    "run2_total": sess.get("run2_phq9_total"),
                    "multi_instrument_in_turn": multi,
                    "n_present_items": len(present_items),
                    "sentence_count": count_sentences(turn),
                })

    # GAD-7
    for cell in GAD_CELLS:
        target_sessions = [s for s, info in session_summary.items() if info["gad7_cell"] == cell]
        for sid in target_sessions:
            sess = session_summary[sid]
            for rn in range(1, 83):
                turn = rounds.get(rn, {}).get(sid)
                if not turn or len(turn.split()) < 5 or len(turn.split()) > 100:
                    continue
                if count_sentences(turn) > 4:
                    continue
                rec = run2_preds.get((sid, rn))
                if rec is None:
                    continue
                present_items: list[tuple[int, str, int]] = []
                for ii in range(1, 8):
                    status, ev = step1_status(rec, "GAD-7", ii)
                    score = step2_score(rec, "GAD-7", ii) or 0
                    if status == "present" and score >= 1:
                        present_items.append((ii, ev, score))
                if not present_items:
                    continue
                present_items.sort(key=lambda t: (t[2] or 0), reverse=True)
                target_item, ev, score = present_items[0]
                multi = False
                for ii in range(1, 10):
                    s, _ = step1_status(rec, "PHQ-9", ii)
                    if s == "present":
                        multi = True
                        break
                candidates[f"GAD7_{cell}"].append({
                    "cell_id": f"GAD7_{cell}",
                    "session_id": sid,
                    "round_n": rn,
                    "patient_turn_es": turn,
                    "target_item_idx": target_item,
                    "target_item_desc": GAD7_ITEMS[target_item],
                    "target_item_score_run2": score,
                    "evidence_run2": ev,
                    "gold_total": sess["gad7_gold_total"],
                    "gold_band": sess["gad7_band_full"],
                    "run0_total": sess.get("run0_gad7_total"),
                    "run1_total": sess.get("run1_gad7_total"),
                    "run2_total": sess.get("run2_gad7_total"),
                    "multi_instrument_in_turn": multi,
                    "n_present_items": len(present_items),
                    "sentence_count": count_sentences(turn),
                })

    # CompACT-10 — uses item statuses from step_1_detection (status='present')
    for cell in COMPACT_CELLS:
        target_sessions = [s for s, info in session_summary.items() if info["compact_cell"] == cell]
        for sid in target_sessions:
            sess = session_summary[sid]
            for rn in range(1, 83):
                turn = rounds.get(rn, {}).get(sid)
                if not turn or len(turn.split()) < 5 or len(turn.split()) > 100:
                    continue
                if count_sentences(turn) > 4:
                    continue
                rec = run2_preds.get((sid, rn))
                if rec is None:
                    continue
                present_items: list[tuple[int, str, int]] = []
                for ii in range(1, 11):
                    status, ev = step1_status(rec, "CompACT-10", ii)
                    score = step2_score(rec, "CompACT-10", ii) or 3
                    if status == "present" and (score <= 1 or score >= 5):
                        present_items.append((ii, ev, score))
                if not present_items:
                    continue
                present_items.sort(key=lambda t: (abs((t[2] or 3) - 3), 0), reverse=True)
                target_item, ev, score = present_items[0]
                candidates[f"CompACT10_{cell}"].append({
                    "cell_id": f"CompACT10_{cell}",
                    "session_id": sid,
                    "round_n": rn,
                    "patient_turn_es": turn,
                    "target_item_idx": target_item,
                    "target_item_desc": COMPACT10_ITEMS[target_item],
                    "target_item_score_run2": score,
                    "evidence_run2": ev,
                    "gold_total": sess["compact_gold_total"],
                    "run0_total": sess.get("run0_compact_total"),
                    "run1_total": sess.get("run1_compact_total"),
                    "run2_total": sess.get("run2_compact_total"),
                    "n_present_items": len(present_items),
                    "sentence_count": count_sentences(turn),
                })

    # Rank and truncate candidates per cell. Preference order:
    # (1) single-item present (cleaner), (2) middle of gold band, (3) shorter snippet
    def rank_key(c: dict) -> tuple:
        return (
            c.get("multi_instrument_in_turn", False),
            c.get("n_present_items", 99),
            c.get("sentence_count", 99),
        )

    top_candidates = {cell: sorted(cs, key=rank_key)[:3] for cell, cs in candidates.items()}

    return {
        "session_summary": session_summary,
        "candidates": top_candidates,
        "candidate_counts": {k: len(v) for k, v in candidates.items()},
    }


# ============================================================================
# Slot 2: §5.2 reACT round-trace candidates from trial
# ============================================================================

# Per src/mentalriskes/task1/calibration.py
_OTE_IDX = [2, 4, 7]      # items 3, 5, 8
_BA_IDX = [0, 5, 8]       # items 1, 6, 9
_VA_IDX = [1, 3, 6, 9]    # items 2, 4, 7, 10
_PHQ9_SOMATIC_IDX = [2, 3, 4, 7]
_GAD7_SOMATIC_IDX = [0, 3, 4]


def _mean(items: list[int], idx: list[int]) -> float:
    return sum(items[i] for i in idx) / len(idx)


def _distress_band(phq_total: int, gad_total: int) -> str:
    if phq_total >= 15 or gad_total >= 15:
        return "high"
    if phq_total >= 10 or gad_total >= 10:
        return "moderate"
    return "low"


# Same expected ranges as src/mentalriskes/task1/calibration.py
_VA_EXPECTED = {"low": (3.5, 5.5), "moderate": (2.5, 4.5), "high": (1.5, 3.5)}
_OTE_EXPECTED = {"low": (3.5, 5.5), "moderate": (3.0, 5.0), "high": (2.5, 4.5)}
_SELF_CONTRADICTION_OTE = 2.5


def reapply_level_b(phq9: list[int], gad7: list[int], compact10: list[int]) -> list[dict]:
    """Re-apply Level B rules and return list of violations (rule, severity, msg).

    Mirrors src/mentalriskes/task1/calibration.py:apply_level_b_constraints,
    but only collects the violation records.
    """
    violations: list[dict] = []
    phq_total = sum(phq9)
    gad_total = sum(gad7)

    phq_norm = phq_total / 27.0
    gad_norm = gad_total / 21.0
    if abs(phq_norm - gad_norm) > 0.40:
        violations.append({
            "rule": "C1",
            "severity": "high",
            "name": "PHQ-9/GAD-7 normalised discordance",
            "message": f"Δnorm={abs(phq_norm-gad_norm):.2f} > 0.40 "
                       f"(PHQ-9 {phq_total}/27, GAD-7 {gad_total}/21).",
        })
    if abs(phq_total - gad_total) > 8:
        violations.append({
            "rule": "C2",
            "severity": "medium",
            "name": "Inter-instrument gap",
            "message": f"|PHQ-9-GAD-7| = {abs(phq_total-gad_total)} > 8",
        })
    if len(phq9) >= 9 and len(gad7) >= 7:
        phs = _mean(phq9, _PHQ9_SOMATIC_IDX)
        gas = _mean(gad7, _GAD7_SOMATIC_IDX)
        if abs(phs - gas) > 1.5:
            violations.append({
                "rule": "C3",
                "severity": "medium",
                "name": "Somatic-factor tracking",
                "message": f"PHQ-9 somatic {phs:.2f} vs GAD-7 somatic {gas:.2f} (Δ={abs(phs-gas):.2f})",
            })
    if len(compact10) >= 10:
        band = _distress_band(phq_total, gad_total)
        va = _mean(compact10, _VA_IDX)
        ote = _mean(compact10, _OTE_IDX)
        va_low, va_high = _VA_EXPECTED.get(band, (2.0, 4.5))
        ote_low, _ = _OTE_EXPECTED.get(band, (2.5, 5.0))
        if va > va_high + 1.0:
            if ote < _SELF_CONTRADICTION_OTE:
                violations.append({
                    "rule": "C4",
                    "severity": "high",
                    "name": "CompACT-VA rescaling (self-contradiction guard)",
                    "message": (f"VA mean {va:.2f} > {va_high+1.0:.2f} but OtE {ote:.2f} < "
                                f"{_SELF_CONTRADICTION_OTE} → self-contradiction; VA NOT corrected"),
                })
            else:
                violations.append({
                    "rule": "C4",
                    "severity": "high",
                    "name": "CompACT-VA distress-conditional rescaling",
                    "message": (f"VA mean {va:.2f} > {va_high+1.0:.2f} (distress band={band}); "
                                f"OtE {ote:.2f} ≥ {_SELF_CONTRADICTION_OTE}; "
                                f"−1 applied to items {[i+1 for i in _VA_IDX]}"),
                    "correction_applied": True,
                    "items_corrected": [i+1 for i in _VA_IDX],
                })
        if ote < ote_low - 0.5:
            violations.append({
                "rule": "C5",
                "severity": "medium",
                "name": "CompACT-OtE under-prediction",
                "message": f"OtE {ote:.2f} < {ote_low-0.5:.2f} (band={band}) — flag only",
            })
        for name, idx in [("OtE", _OTE_IDX), ("BA", _BA_IDX), ("VA", _VA_IDX)]:
            sub = [compact10[i] for i in idx]
            spread = max(sub) - min(sub)
            if spread > 3:
                violations.append({
                    "rule": "C6",
                    "severity": "medium",
                    "name": f"CompACT-{name} within-subprocess spread",
                    "message": f"spread {spread} > 3 across {sub}",
                })
    if len(phq9) >= 9 and phq9[8] > 0 and phq_total < 10:
        violations.append({
            "rule": "C7",
            "severity": "high",
            "name": "PHQ-9 item 9 safety flag",
            "message": f"item 9 = {phq9[8]} > 0 with PHQ-9 total {phq_total} < 10",
        })
    return violations


def slot2_extract_candidates() -> dict:
    """Re-applies Level B rules to the trial conversation's per-round scores.

    Caveat: the official trial run did not persist per-round Level B violation
    logs, so we reconstruct firings deterministically from the saved item-level
    scores (`output/mentalriskes/trial_results_all.json`). We use Run 2 (A1 =
    anchors only, no Level B) as the *pre*-rule input, because Run 1 (A3) and
    Run 0 (A5) already have C4 corrections folded in.
    """
    trial_runs = load_trial_t1_runs()
    turns = load_trial_t1_patient_turns()

    run0 = {r["round"]: r for r in trial_runs["runs"]["run0_primary"]["rounds"]}
    run1 = {r["round"]: r for r in trial_runs["runs"]["run1_comparison"]["rounds"]}
    run2 = {r["round"]: r for r in trial_runs["runs"]["run2_lightweight"]["rounds"]}

    gold = trial_runs["gold"]

    # T2 = early-weighted aggregation; T3 = stability-adaptive on CompACT-10.
    # We don't have the aggregator on hand, so just report running mean of the
    # per-round Run 0 scores up to the round.
    def running_mean(run_dict: dict[int, dict], up_to: int, key: str) -> list[float]:
        totals = [sum(run_dict[r][key]) for r in range(1, up_to + 1)]
        return totals

    candidates = []
    for rn in range(3, 19):  # mid-session = 5..17; we widen to 3..18 for harvest
        if rn not in run2:
            continue
        rec2 = run2[rn]
        violations = reapply_level_b(rec2["phq9"], rec2["gad7"], rec2["compact10"])
        if not violations:
            continue

        rec0 = run0.get(rn, rec2)
        rec1 = run1.get(rn, rec2)

        # Score change: Run 1 (with Level B) vs Run 2 (without)
        deltas = {
            "phq9": [a - b for a, b in zip(rec1["phq9"], rec2["phq9"])],
            "gad7": [a - b for a, b in zip(rec1["gad7"], rec2["gad7"])],
            "compact10": [a - b for a, b in zip(rec1["compact10"], rec2["compact10"])],
        }
        # Level C effect: Run 0 vs Run 1
        level_c_deltas = {
            "phq9": [a - b for a, b in zip(rec0["phq9"], rec1["phq9"])],
            "gad7": [a - b for a, b in zip(rec0["gad7"], rec1["gad7"])],
            "compact10": [a - b for a, b in zip(rec0["compact10"], rec1["compact10"])],
        }
        level_c_changed = any(abs(d) > 0 for k in level_c_deltas for d in level_c_deltas[k])
        level_b_changed = any(abs(d) > 0 for k in deltas for d in deltas[k])

        # Strongest rule by severity
        sev_rank = {"high": 0, "medium": 1, "low": 2}
        violations.sort(key=lambda v: sev_rank.get(v.get("severity", "low"), 9))

        candidates.append({
            "round_n": rn,
            "patient_turn_es": turns.get(rn, ""),
            "context_prev": [turns.get(rn - 2, ""), turns.get(rn - 1, "")],
            "rec_no_level_b": rec2,            # Run 2 = no Level B
            "rec_with_level_b": rec1,          # Run 1 = Level B applied
            "rec_with_level_c": rec0,          # Run 0 = Level B + Level C
            "level_b_violations": violations,
            "level_b_changed_scores": level_b_changed,
            "level_c_changed_scores": level_c_changed,
            "level_b_deltas": deltas,
            "level_c_deltas": level_c_deltas,
            "gold": gold,
            "n_violations": len(violations),
            "rule_codes": [v["rule"] for v in violations],
        })

    # Rank: mid-session + rule fires + score change preferred
    def rank_key(c: dict) -> tuple:
        mid = 1 if 5 <= c["round_n"] <= 17 else 0
        return (
            -mid,
            -int(c["level_b_changed_scores"]),
            -c["n_violations"],
            -int("C4" in c["rule_codes"] or "C7" in c["rule_codes"]),
            c["round_n"],
        )

    candidates.sort(key=rank_key)
    return {"candidates": candidates[:5], "all_candidate_count": len(candidates)}


# ============================================================================
# Slot 3: §5.3 tACT round-trace candidates from trial
# ============================================================================

def slot3_extract_candidates() -> dict:
    t2_rounds = load_trial_t2_rounds()
    runs = load_trial_t2_runs()
    gold = TRIAL_GROUND_TRUTH

    candidates = []
    for rn in range(2, 19):  # rounds 5..17 preferred per spec
        if rn not in gold:
            continue
        sel0 = runs["run0"].get(rn, {}).get("chosen_option") if runs.get("run0") else None
        sel1 = runs["run1"].get(rn, {}).get("chosen_option") if runs.get("run1") else None
        sel2 = runs["run2"].get(rn, {}).get("chosen_option") if runs.get("run2") else None

        if sel2 is None or sel2 == gold[rn]:
            continue  # Slot 3 requires system-gold disagreement

        rec0 = runs["run0"].get(rn, {})
        rec1 = runs["run1"].get(rn, {})
        rec2 = runs["run2"].get(rn, {})

        candidates.append({
            "round_n": rn,
            "patient_turn_es": t2_rounds[rn]["patient_input"],
            "option_1": t2_rounds[rn]["option_1"],
            "option_2": t2_rounds[rn]["option_2"],
            "option_3": t2_rounds[rn]["option_3"],
            "gold_selection": gold[rn],
            "tact_run0": sel0,
            "tact_run1": sel1,
            "tact_run2": sel2,
            "state_snapshot_run2": rec2.get("state_snapshot", {}),
            "raw_evaluation_run2": rec2.get("raw_evaluation", {}),
            "raw_evaluation_run1": rec1.get("raw_evaluation", {}),
            "reasoning_run2": rec2.get("reasoning", ""),
            "primary_tag_run2": rec2.get("primary_tag", ""),
            "all_runs_disagree": sel0 != gold[rn] and sel1 != gold[rn] and sel2 != gold[rn],
            "all_runs_agree_among_themselves": sel0 == sel1 == sel2,
            "candidates_distinct": True,  # we don't have a reliable similarity metric; mark for review
        })

    # Rank per spec: mid-session, gold=3, all-systems-wrong, runs disagree
    def rank_key(c: dict) -> tuple:
        mid = 1 if 5 <= c["round_n"] <= 17 else 0
        return (
            -mid,
            -int(c["gold_selection"] == 3),
            -int(c["all_runs_disagree"]),
            -int(not c["all_runs_agree_among_themselves"]),
            c["round_n"],
        )

    candidates.sort(key=rank_key)
    return {"candidates": candidates[:5], "all_candidate_count": len(candidates)}


# ============================================================================
# Markdown writers
# ============================================================================

def escape_md_cell(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return s.strip()


def md_slot1(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Slot 1 — §2.2 Figure 1 candidates")
    lines.append("")
    lines.append(
        "Auto-harvested by `analysis/MentalRiskES_test/extract_example_candidates.py`. "
        "Each candidate's `target_item` is the LLM (Run 2 / A1) step_1_detection item "
        "marked `present` with the highest item score; `evidence_run2` is the assessor's "
        "own evidence string for that item. Final selection still needs human review."
    )
    lines.append("")

    lines.append("## Per-session gold-band reference")
    lines.append("")
    lines.append(
        "| Session | PHQ-9 gold / band / cell | GAD-7 gold / band / cell | CompACT-10 gold / cell "
        "| Run 0 (P/G/C) | Run 1 (P/G/C) | Run 2 (P/G/C) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for sid in sorted(data["session_summary"].keys()):
        s = data["session_summary"][sid]
        def fmt_run(prefix: str) -> str:
            return (
                f"{s.get(f'{prefix}_phq9_total')}/"
                f"{s.get(f'{prefix}_gad7_total')}/"
                f"{s.get(f'{prefix}_compact_total')}"
            )
        lines.append(
            f"| {sid} | {s['phq9_gold_total']} / {s['phq9_band_full']} / {s['phq9_cell'] or '—'} "
            f"| {s['gad7_gold_total']} / {s['gad7_band_full']} / {s['gad7_cell'] or '—'} "
            f"| {s['compact_gold_total']} / {s['compact_cell']} "
            f"| {fmt_run('run0')} | {fmt_run('run1')} | {fmt_run('run2')} |"
        )
    lines.append("")

    cell_order = [
        "PHQ9_mild", "PHQ9_moderate", "PHQ9_severe",
        "GAD7_mild", "GAD7_moderate", "GAD7_severe",
        "CompACT10_low", "CompACT10_medium", "CompACT10_high",
    ]
    for cell in cell_order:
        cs = data["candidates"].get(cell, [])
        n_total = data["candidate_counts"].get(cell, 0)
        lines.append(f"### Cell `{cell}` ({len(cs)} of {n_total} harvested shown)")
        lines.append("")
        if not cs:
            lines.append(
                "_No candidates harvested — none of the 10 released test sessions sit in this "
                "band, or no patient turns of suitable length had clean single-item evidence. "
                "Per outstanding decision 1, document the gap rather than padding._"
            )
            lines.append("")
            continue
        lines.append(
            "| # | session | round | patient_turn_es | target_item | target_score | "
            "evidence (assessor, en) | gold total / band | run0 / run1 / run2 (total) | notes |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|"
        )
        for i, c in enumerate(cs, 1):
            instrument_total_keys = {
                "PHQ9": ("run0_total", "run1_total", "run2_total", "gold_total", "gold_band"),
                "GAD7": ("run0_total", "run1_total", "run2_total", "gold_total", "gold_band"),
                "CompACT10": ("run0_total", "run1_total", "run2_total", "gold_total", None),
            }
            inst = cell.split("_")[0]
            keys = instrument_total_keys[inst]
            run_totals = f"{c.get(keys[0])} / {c.get(keys[1])} / {c.get(keys[2])}"
            gold_band = (
                f"{c.get(keys[3])} / {c.get(keys[4])}"
                if keys[4]
                else f"{c.get(keys[3])}"
            )
            notes_bits = []
            if c.get("multi_instrument_in_turn"):
                notes_bits.append("multi-instrument in turn")
            if c.get("n_present_items") and c["n_present_items"] > 1:
                notes_bits.append(f"{c['n_present_items']} items present")
            notes = "; ".join(notes_bits)
            lines.append(
                f"| {i} | {c['session_id']} | {c['round_n']} "
                f"| {escape_md_cell(c['patient_turn_es'])} "
                f"| {c['target_item_idx']} — {escape_md_cell(c['target_item_desc'])} "
                f"| {c['target_item_score_run2']} "
                f"| {escape_md_cell(c['evidence_run2'])} "
                f"| {gold_band} "
                f"| {run_totals} "
                f"| {escape_md_cell(notes)} |"
            )
        lines.append("")
    return "\n".join(lines)


def md_slot2(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Slot 2 — §5.2 reACT round-trace candidates (trial)")
    lines.append("")
    lines.append(
        "**Caveat:** the official trial run did not persist per-round Level B violation logs; "
        "we deterministically re-apply the 7-rule Level B system from "
        "`src/mentalriskes/task1/calibration.py` to Run 2 (A1, no Level B) outputs to identify "
        "rule firings on each round. Level B / C deltas are derived by comparing Run 2 (no Level B) "
        "to Run 1 (Level B applied) and Run 0 (Level B + Level C)."
    )
    lines.append("")

    lines.append(
        f"_{data['all_candidate_count']} candidate rounds in the harvest pool (rounds 3–18 with at "
        f"least one rule firing). Top {len(data['candidates'])} shown below._"
    )
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| # | round | patient_turn_es (excerpt) | rules fired (codes) | level B changed scores? "
        "| level C changed scores? | PHQ-9 (Run 2 / Run 1 / Run 0) | GAD-7 same | CompACT-10 same |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for i, c in enumerate(data["candidates"], 1):
        rule_codes = ",".join(c["rule_codes"])
        excerpt = (c["patient_turn_es"][:80] + "…") if len(c["patient_turn_es"]) > 80 else c["patient_turn_es"]

        def trio(field: str) -> str:
            return (
                f"{sum(c['rec_no_level_b'][field])} / "
                f"{sum(c['rec_with_level_b'][field])} / "
                f"{sum(c['rec_with_level_c'][field])}"
            )

        lines.append(
            f"| {i} | {c['round_n']} | {escape_md_cell(excerpt)} | {rule_codes} "
            f"| {'yes' if c['level_b_changed_scores'] else 'no'} "
            f"| {'yes' if c['level_c_changed_scores'] else 'no'} "
            f"| {trio('phq9')} | {trio('gad7')} | {trio('compact10')} |"
        )
    lines.append("")

    lines.append("## Per-candidate detail")
    lines.append("")
    for i, c in enumerate(data["candidates"], 1):
        lines.append(f"### Candidate {i} — round {c['round_n']}")
        lines.append("")
        lines.append(f"**patient_turn_es:** {c['patient_turn_es']}")
        lines.append("")
        lines.append("**windowed context (rounds r−2, r−1):**")
        lines.append("")
        for ctx in c["context_prev"]:
            if ctx:
                lines.append(f"- {ctx}")
        lines.append("")
        lines.append("**reACT scores at this round:**")
        lines.append("")
        lines.append(
            "| Instrument | Pre-Level-B (Run 2, A1) | Post-Level-B (Run 1, A3) | Post-Level-C (Run 0, A5) | Gold |"
        )
        lines.append("|---|---|---|---|---|")
        for key, label in [("phq9", "PHQ-9"), ("gad7", "GAD-7"), ("compact10", "CompACT-10")]:
            gold_key = {"phq9": "PHQ-9", "gad7": "GAD-7", "compact10": "CompACT-10"}[key]
            lines.append(
                f"| {label} | {c['rec_no_level_b'][key]} | {c['rec_with_level_b'][key]} "
                f"| {c['rec_with_level_c'][key]} | {c['gold'][gold_key]} |"
            )
        lines.append("")
        lines.append("**Level B violations (deterministically reconstructed):**")
        lines.append("")
        for v in c["level_b_violations"]:
            lines.append(
                f"- **{v['rule']}** [{v['severity']}] — {v['name']}: {v['message']}"
            )
        lines.append("")
        if c["level_b_changed_scores"]:
            lines.append(
                "**Level B Δ (Run 1 − Run 2):** PHQ-9 "
                f"{c['level_b_deltas']['phq9']}; GAD-7 {c['level_b_deltas']['gad7']}; "
                f"CompACT-10 {c['level_b_deltas']['compact10']}"
            )
            lines.append("")
        if c["level_c_changed_scores"]:
            lines.append(
                "**Level C Δ (Run 0 − Run 1):** PHQ-9 "
                f"{c['level_c_deltas']['phq9']}; GAD-7 {c['level_c_deltas']['gad7']}; "
                f"CompACT-10 {c['level_c_deltas']['compact10']}"
            )
            lines.append("")
    return "\n".join(lines)


def md_slot3(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Slot 3 — §5.3 tACT round-trace candidates (trial)")
    lines.append("")
    lines.append(
        "Candidates are restricted to trial rounds 2–18 where Run 2 (B+/HYB/FIX = our submitted "
        "Run 2 equivalent) disagrees with `TRIAL_GROUND_TRUTH` "
        "(`src/mentalriskes/task2/data.py`). Run 0 = `FUNC_PERM_W3`, Run 1 = `FUNC_FIX_W3`, "
        "Run 2 = `HYB_FIX_W3` (B+ pipeline) — these are the ablation files closest to the "
        "submitted-run configurations."
    )
    lines.append("")

    lines.append(
        f"_{data['all_candidate_count']} disagreement rounds in the pool; top {len(data['candidates'])} "
        "shown below._"
    )
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| # | round | turn_es (excerpt) | gold | run0 | run1 | run2 | all-runs-wrong | gold=3? |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for i, c in enumerate(data["candidates"], 1):
        excerpt = (c["patient_turn_es"][:80] + "…") if len(c["patient_turn_es"]) > 80 else c["patient_turn_es"]
        lines.append(
            f"| {i} | {c['round_n']} | {escape_md_cell(excerpt)} | {c['gold_selection']} "
            f"| {c['tact_run0']} | {c['tact_run1']} | {c['tact_run2']} "
            f"| {'yes' if c['all_runs_disagree'] else 'no'} "
            f"| {'yes' if c['gold_selection'] == 3 else 'no'} |"
        )
    lines.append("")

    lines.append("## Per-candidate detail")
    lines.append("")
    for i, c in enumerate(data["candidates"], 1):
        lines.append(f"### Candidate {i} — round {c['round_n']}  (gold {c['gold_selection']}, "
                     f"run0/1/2 = {c['tact_run0']}/{c['tact_run1']}/{c['tact_run2']})")
        lines.append("")
        lines.append(f"**patient_turn_es:** {c['patient_turn_es']}")
        lines.append("")
        lines.append(f"**option_1:** {c['option_1']}")
        lines.append("")
        lines.append(f"**option_2:** {c['option_2']}")
        lines.append("")
        lines.append(f"**option_3:** {c['option_3']}")
        lines.append("")
        snap = c.get("state_snapshot_run2", {})
        if snap:
            lines.append("**state tracker (Run 2 snapshot):**")
            lines.append("")
            lines.append(f"- fase: {snap.get('fase_terapeutica')}")
            estado = snap.get("estado_emocional", {})
            if estado:
                lines.append(
                    f"- estado emocional: valencia={estado.get('valencia')}, "
                    f"intensidad={estado.get('intensidad')}, "
                    f"orientación={estado.get('orientacion_accion')}"
                )
            procesos = snap.get("procesos_act", {})
            if procesos:
                active = {k: v for k, v in procesos.items() if v and v > 0}
                if active:
                    lines.append(f"- procesos ACT activos: {active}")
            mets = snap.get("metaforas_activas", [])
            if mets:
                lines.append(f"- metáforas activas: {mets}")
            resumen = snap.get("resumen_acumulado")
            if resumen:
                lines.append(f"- resumen acumulado: {resumen}")
            lines.append("")
        eval_run2 = c.get("raw_evaluation_run2", {})
        if eval_run2:
            lines.append("**evaluator (Run 2 HYB) — caracterización per candidate:**")
            lines.append("")
            chars = eval_run2.get("caracterización", {})
            for opt_key in ("opcion_1", "opcion_2", "opcion_3"):
                ch = chars.get(opt_key, {})
                if ch:
                    lines.append(
                        f"- {opt_key}: función_principal=\"{ch.get('función_principal','')}\"; "
                        f"consistencia={ch.get('etiquetas_consistencia', [])}; "
                        f"inconsistencia={ch.get('etiquetas_inconsistencia', [])}"
                    )
            lines.append("")
            sel = eval_run2.get("selección", {}).get("opcion_elegida", {})
            if sel:
                lines.append(
                    f"**Run 2 selection rationale:** option {sel.get('numero')} — "
                    f"{sel.get('razón_principal','')}"
                )
                lines.append("")
        lines.append(f"**Run 2 reasoning:** {c.get('reasoning_run2','')}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("[slot1] extracting Figure-1 candidates…", file=sys.stderr)
    slot1 = slot1_extract_candidates()
    print("[slot2] extracting reACT round-trace candidates…", file=sys.stderr)
    slot2 = slot2_extract_candidates()
    print("[slot3] extracting tACT round-trace candidates…", file=sys.stderr)
    slot3 = slot3_extract_candidates()

    raw = {"slot1": slot1, "slot2": slot2, "slot3": slot3}
    raw_path = OUT_DIR / "example_candidates_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)
    print(f"[wrote] {raw_path}", file=sys.stderr)

    md = "\n\n---\n\n".join([md_slot1(slot1), md_slot2(slot2), md_slot3(slot3)])
    md_path = OUT_DIR / "example_candidates.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[wrote] {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
