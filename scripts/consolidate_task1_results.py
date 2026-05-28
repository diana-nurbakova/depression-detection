"""Consolidate every eRisk 2026 Task 1 ablation summary and official-submission
prediction into a single JSON file: ``runs/task1_all_results.json``.

Inputs
------
* ``runs/ablation-1/ablation_summary.json``           — core pipeline ablation A0..A7 (run 1)
* ``runs/ablation-2/ablation_summary.json``           — core pipeline ablation A0..A7 (run 2, replicate)
* ``runs/ablation-3/ablation_summary.json``           — post-hoc correction variants on A0/A7
* ``runs/ablation_comparison/ablation_summary.json``  — cross-strategy comparison (submission proxies)
* ``runs/ablation_sdc/ablation_summary.json``         — Score Distribution Constraint
* ``runs/ablation_debug/ablation_summary.json``       — A0 smoke test
* ``runs/tom_ablation/tom_ablation_summary.json``     — ToM on/off ablation
* ``runs/task1/task1_results_<date>/personaXX/results_{1,2,3}.json``
                                                      — per-batch official submission predictions

Output
------
* ``runs/task1_all_results.json``
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
OUTPUT_PATH = RUNS_DIR / "task1_all_results.json"

GOLDEN_ROOT = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task1-llms"
    / "golden-data"
)
GOLDEN_FILE = GOLDEN_ROOT / "patients_data.jsonl"
SYMPTOM_MAPPINGS_FILE = GOLDEN_ROOT / "symptom_mappings.json"
BDI_LIST_FILE = GOLDEN_ROOT / "bdi_symptoms_list.json"

OFFICIAL_SUBMISSION_ROOT = RUNS_DIR / "task1" / "official_submission" / "task1-llms-results"

# Per eRisk 2026 preliminary results PDF, Section 1 — *official* depression bands.
OFFICIAL_BANDS = [(0, 9, "minimal"), (10, 18, "mild"), (19, 29, "moderate"), (30, 63, "severe")]

# Latency-aware parameters: tables 3-4 of the PDF were verified to use K0=10.
LATENCY_K0 = 10
LATENCY_P = math.log(3) / (LATENCY_K0 - 1)  # ≃ 0.1221

# Sentinel: from which patient_id onward Run 3 was switched to the ToM-based calibration.
# User indication: "starting from personas 12, 13 (I might be mistaken)". Folder persona12
# corresponds to patient_id 13 (Laura), persona13 -> patient_id 14 (Linda), which also
# matches the only _tom variant in the per-batch history (task1_results_20260331_tom).
RUN3_TOM_CUTOFF_PATIENT_ID = 13


METRIC_DEFINITIONS = {
    "dchr": "Diagnosis Certainty Hit Rate — fraction of personas whose predicted BDI-II band matches the golden band.",
    "mad": "Mean Absolute Deviation between predicted and golden BDI-II totals (points).",
    "adodl": "Average Deviation Over Diagonal Loss — 1 - |predicted-golden| / max_possible. Higher is better.",
    "ashr_proxy": "Approximate Symptom Hit Rate proxy — mean of per-persona symptom_hit_rate (top-k key-symptom overlap).",
    "boundary_accuracy": "Band accuracy on personas within ±2 points of a band boundary.",
    "band_accuracy_by_severity": "Per-severity-band recall (minimal / mild / moderate / severe).",
    "per_persona.deviation": "|predicted - golden| BDI-II points.",
    "per_persona.cr": "Closeness ratio = 1 - deviation / max_deviation (per persona).",
    "per_persona.symptom_hit_rate": "Fraction of golden top-4 symptoms recovered by the system.",
}


ABLATION_SOURCES = [
    {
        "key": "core_pipeline_ablation_run1",
        "path": RUNS_DIR / "ablation-1" / "ablation_summary.json",
        "description": (
            "Core stepwise (additive) pipeline ablation across 12 TalkDep personas. "
            "Configs progressively add: A0_baseline → A1_specialized assessors → "
            "A2_linguistic features → A3_prior (Bayesian) → A4_justificator → "
            "A5/A6 temperature sweep → A7_no_prior."
        ),
        "n_personas_expected": 12,
    },
    {
        "key": "core_pipeline_ablation_run2_replicate",
        "path": RUNS_DIR / "ablation-2" / "ablation_summary.json",
        "description": "Independent replicate of the core A0..A7 ablation (validation of stability).",
        "n_personas_expected": 12,
    },
    {
        "key": "post_hoc_correction_variants",
        "path": RUNS_DIR / "ablation-3" / "ablation_summary.json",
        "description": (
            "Targeted post-hoc correction variants: A0_band_aware (band-specific deltas), "
            "A0_minus5 (flat -5), A7_progressive (progressive weighting on A7 base)."
        ),
        "n_personas_expected": 12,
    },
    {
        "key": "correction_strategy_comparison",
        "path": RUNS_DIR / "ablation_comparison" / "ablation_summary.json",
        "description": (
            "Cross-strategy correction comparison on A0 base + A4 + A7. Includes the exact "
            "configurations used as proxies for the three official submission runs: "
            "A0_band_aware (Run 1 proxy), A0_flat_minus_2 (Run 2 proxy), "
            "A7_proportional_085 (alt strategy). Configs evaluated: "
            "A0_none, A0_baseline, A0_flat_minus_2, A0_minus5, A0_band_aware, "
            "A7_progressive, A7_proportional_085, A4_justificator."
        ),
        "n_personas_expected": 12,
    },
    {
        "key": "sdc_score_distribution_constraint",
        "path": RUNS_DIR / "ablation_sdc" / "ablation_summary.json",
        "description": (
            "Score Distribution Constraint ablations: A0_sdc (SDC only) and "
            "A0_sdc_band_aware (SDC + band_aware). SDC downgrades the lowest-confidence "
            "score-3 items when ≥3 minimizing-language / functional-activity signals appear."
        ),
        "n_personas_expected": 12,
    },
    {
        "key": "debug_smoke_test",
        "path": RUNS_DIR / "ablation_debug" / "ablation_summary.json",
        "description": "A0_baseline smoke run used during pipeline debugging.",
        "n_personas_expected": 12,
    },
]


TOM_ABLATION = {
    "key": "tom_theory_of_mind",
    "summary_file": RUNS_DIR / "tom_ablation" / "tom_ablation_summary.json",
    "per_persona_dirs": {
        "tom_off": RUNS_DIR / "tom_ablation" / "tom_off",
        "tom_on": RUNS_DIR / "tom_ablation" / "tom_on",
    },
    "description": (
        "Theory of Mind ablation: tom_off (full pipeline w/o ToM) vs tom_on (ToM "
        "perception tracking + coverage-gap guidance + C1/C2 corrections). "
        "Conclusion in the solution description: ToM tracking is enabled for "
        "orchestrator guidance, but C1/C2 corrections are DISABLED in the submission "
        "runs because the confidence gate over-prunes and the somatic boost mis-calibrates. "
        "Per-persona files in tom_on/ and tom_off/ contain full results for all 12 TalkDep "
        "personas; the legacy tom_ablation_summary.json was truncated to 2 personas and is "
        "re-aggregated here from the per-persona files."
    ),
}


BDI_MAX = 63
# Local ablation summaries use a slightly different band split (matches the team's
# TalkDep evaluation convention); the official eRisk bands are stored in OFFICIAL_BANDS.
BAND_THRESHOLDS = [(0, 13, "minimal"), (14, 19, "mild"), (20, 28, "moderate"), (29, 63, "severe")]


def _band_for(score: int, bands=BAND_THRESHOLDS) -> str:
    for lo, hi, name in bands:
        if lo <= score <= hi:
            return name
    return "severe"


def _is_band_boundary(score: int, bands=OFFICIAL_BANDS, window: int = 2) -> bool:
    for _, hi, _ in bands:
        if hi == bands[-1][1]:
            continue
        if abs(score - hi) <= window or abs(score - (hi + 1)) <= window:
            return True
    return False


def _speed_factor(k: int) -> float:
    if k <= 0:
        return 1.0
    penalty = -1.0 + 2.0 / (1.0 + math.exp(-LATENCY_P * (k - 1)))
    return 1.0 - penalty


def _load_golden(path: Path) -> dict[int, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["patient_id"])] = {
                "name": row["patient_name"],
                "bdi_score": int(row["bdi_score"]),
                "key_symptoms": list(row["patient_key_symptoms"]),
            }
    return out


def _load_symptom_normaliser(mappings_path: Path, bdi_list_path: Path) -> dict[str, str]:
    canonical_names = [item["name"] for item in _load_json(bdi_list_path)["items"]]
    canon_lower = {n.lower(): n for n in canonical_names}
    raw_mappings = _load_json(mappings_path).get("symptom_mappings", {})
    reverse: dict[str, str] = {n.lower(): n for n in canonical_names}
    for key, variants in raw_mappings.items():
        group = [key, *variants]
        # The canonical BDI symptom is whichever group member matches a BDI item by name.
        target = next((canon_lower[g.lower()] for g in group if g.lower() in canon_lower), None)
        if target is None:
            # Group has no BDI equivalent (e.g. organizer added a sub-clinical concept);
            # leave the surface form unmapped so downstream matching can drop it.
            continue
        for g in group:
            reverse[g.lower().strip()] = target
    return reverse


def _normalise_symptom(s: str, reverse_map: dict[str, str]) -> str:
    return reverse_map.get(s.lower().strip(), s.strip())


def _symptom_hit_rate(predicted: list[str], golden: list[str], reverse_map: dict[str, str]) -> tuple[float, list[str]]:
    pred_canon = {_normalise_symptom(s, reverse_map) for s in predicted}
    gold_canon = {_normalise_symptom(s, reverse_map) for s in golden}
    if not gold_canon:
        return 0.0, []
    matched = sorted(pred_canon & gold_canon)
    return len(matched) / len(gold_canon), matched


def _count_user_messages(interactions_path: Path) -> int | None:
    if not interactions_path.exists():
        return None
    data = _load_json(interactions_path)
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and isinstance(data.get("conversation"), list):
        return sum(1 for turn in data["conversation"] if turn.get("role") == "user")
    return None


def _aggregate_run(per_persona: list[dict]) -> dict:
    n = len(per_persona)
    if n == 0:
        return {"n_personas": 0}
    dchr = sum(1 for p in per_persona if p["band_ok"]) / n
    mad = sum(p["deviation"] for p in per_persona) / n
    adodl = sum(p["cr"] for p in per_persona) / n
    ashr = sum(p["symptom_hit_rate"] for p in per_persona) / n
    ldchr = sum((1 if p["band_ok"] else 0) * p["speed_factor"] for p in per_persona) / n
    lashr = sum(p["symptom_hit_rate"] * p["speed_factor"] for p in per_persona) / n
    by_band: dict[str, list[bool]] = defaultdict(list)
    for p in per_persona:
        by_band[p["golden_band"]].append(p["band_ok"])
    band_acc = {band: round(sum(flags) / len(flags), 4) for band, flags in by_band.items()}
    boundary_personas = [p for p in per_persona if p["near_boundary"]]
    boundary_acc = (
        round(sum(1 for p in boundary_personas if p["band_ok"]) / len(boundary_personas), 4)
        if boundary_personas else None
    )
    return {
        "n_personas": n,
        "dchr": round(dchr, 4),
        "mad": round(mad, 4),
        "adodl": round(adodl, 4),
        "ashr": round(ashr, 4),
        "ldchr_k0_10": round(ldchr, 4),
        "lashr_k0_10": round(lashr, 4),
        "band_accuracy_by_severity": band_acc,
        "boundary_accuracy_pm2": boundary_acc,
        "mean_user_messages": round(
            sum(p["user_messages"] for p in per_persona if p["user_messages"] is not None)
            / max(1, sum(1 for p in per_persona if p["user_messages"] is not None)),
            2,
        ),
    }


def score_official_submission(submission_root: Path) -> dict | None:
    if not submission_root.exists():
        return None
    golden = _load_golden(GOLDEN_FILE)
    reverse_map = _load_symptom_normaliser(SYMPTOM_MAPPINGS_FILE, BDI_LIST_FILE)
    runs: dict[str, list[dict]] = {"Run_1": [], "Run_2": [], "Run_3": []}
    missing_patients = set(golden.keys())
    persona_dirs = sorted(
        (p for p in submission_root.iterdir() if p.is_dir() and p.name.startswith("persona")),
        key=lambda d: int(d.name.replace("persona", "") or "0"),
    )
    for p_dir in persona_dirs:
        for run_idx in (1, 2, 3):
            res_file = p_dir / f"results_{run_idx}.json"
            if not res_file.exists():
                continue
            data = _load_json(res_file)
            if isinstance(data, list):
                if not data:
                    continue
                data = data[0]
            patient_id = int(data["LLM"])
            missing_patients.discard(patient_id)
            gold = golden.get(patient_id)
            if gold is None:
                continue
            predicted_score = int(data["bdi-score"])
            predicted_symptoms = list(data.get("key-symptoms", []))
            deviation = abs(predicted_score - gold["bdi_score"])
            golden_band = _band_for(gold["bdi_score"], OFFICIAL_BANDS)
            predicted_band = _band_for(predicted_score, OFFICIAL_BANDS)
            cr = (BDI_MAX - deviation) / BDI_MAX
            shr, matched = _symptom_hit_rate(predicted_symptoms, gold["key_symptoms"], reverse_map)
            k_u = _count_user_messages(p_dir / f"interactions_{run_idx}.json")
            speed = _speed_factor(k_u) if k_u is not None else 1.0
            entry = {
                "patient_id": patient_id,
                "persona_name": gold["name"],
                "persona_dir": p_dir.name,
                "golden_bdi": gold["bdi_score"],
                "predicted_bdi": predicted_score,
                "golden_band": golden_band,
                "predicted_band": predicted_band,
                "band_ok": golden_band == predicted_band,
                "deviation": deviation,
                "cr": round(cr, 4),
                "golden_symptoms": gold["key_symptoms"],
                "predicted_symptoms": predicted_symptoms,
                "matched_symptoms": matched,
                "symptom_hit_rate": round(shr, 4),
                "user_messages": k_u,
                "speed_factor": round(speed, 4),
                "near_boundary": _is_band_boundary(gold["bdi_score"]),
            }
            runs[f"Run_{run_idx}"].append(entry)

    aggregates: dict[str, dict] = {}
    for run_name, entries in runs.items():
        agg = _aggregate_run(entries)
        if run_name == "Run_3":
            pre = [e for e in entries if e["patient_id"] < RUN3_TOM_CUTOFF_PATIENT_ID]
            post = [e for e in entries if e["patient_id"] >= RUN3_TOM_CUTOFF_PATIENT_ID]
            agg["run3_split_by_calibration"] = {
                "_cutoff_note": (
                    f"Per user, Run 3 switched to ToM-based calibration starting at "
                    f"patient_id >= {RUN3_TOM_CUTOFF_PATIENT_ID} (persona folder >= "
                    f"persona{RUN3_TOM_CUTOFF_PATIENT_ID - 1}). Verify cutoff."
                ),
                "pre_cutoff_flat_minus_3": {
                    "patient_ids": [e["patient_id"] for e in pre],
                    **_aggregate_run(pre),
                },
                "post_cutoff_tom_calibrated": {
                    "patient_ids": [e["patient_id"] for e in post],
                    **_aggregate_run(post),
                },
            }
        aggregates[run_name] = {**agg, "per_persona": entries}
    # Side-by-side delta with the organizer-reported preliminary numbers — kept tiny.
    discrepancy = {}
    for run_name in ("Run_1", "Run_2", "Run_3"):
        pdf_row = OFFICIAL_PRELIMINARY_PDF["runs"][run_name]
        local_row = aggregates[run_name]
        discrepancy[run_name] = {
            "DCHR": {"local": local_row["dchr"], "pdf": pdf_row["DCHR"], "delta": round(local_row["dchr"] - pdf_row["DCHR"], 4)},
            "ADODL": {"local": local_row["adodl"], "pdf": pdf_row["ADODL"], "delta": round(local_row["adodl"] - pdf_row["ADODL"], 4)},
            "ASHR": {"local": local_row["ashr"], "pdf": pdf_row["ASHR"], "delta": round(local_row["ashr"] - pdf_row["ASHR"], 4)},
            "LDCHR": {"local": local_row["ldchr_k0_10"], "pdf": pdf_row["LDCHR"], "delta": round(local_row["ldchr_k0_10"] - pdf_row["LDCHR"], 4)},
            "LASHR": {"local": local_row["lashr_k0_10"], "pdf": pdf_row["LASHR"], "delta": round(local_row["lashr_k0_10"] - pdf_row["LASHR"], 4)},
        }

    return {
        "source_dir": str(submission_root.relative_to(REPO_ROOT).as_posix()),
        "golden_file": str(GOLDEN_FILE.relative_to(REPO_ROOT).as_posix()),
        "missing_patient_ids": sorted(missing_patients),
        "latency_settings": {"K0": LATENCY_K0, "p": round(LATENCY_P, 6)},
        "band_definition": "official_erisk_2026 (minimal 0-9 / mild 10-18 / moderate 19-29 / severe 30-63)",
        "discrepancy_vs_pdf": discrepancy,
        "discrepancy_note": (
            "Local re-scoring of `official_submission/task1-llms-results/` yields metrics "
            "noticeably better than the PDF row for INSA-Lyon (DCHR delta +0.16 to +0.26 "
            "across runs; ADODL delta +0.03 to +0.04; ASHR delta +0.06 to +0.13). "
            "scripts/diff_task1_submission.py verified that every file in this folder is "
            "byte-identical to at least one per-batch task1_results_<date>/ predecessor, so "
            "the gap is not caused by post-hoc overwrites. The most plausible explanations: "
            "(a) the eRisk server scored an earlier upload for some personas that we no "
            "longer have on disk, or (b) a silent rejection on the server side left an "
            "older submission as the scored one. The PDF values remain authoritative; "
            "the local re-score is kept for per-persona diagnostics only."
        ),
        "runs": aggregates,
    }


# ---------------------------------------------------------------------------
# Official preliminary results (eRisk 2026), Tables 3 & 4 of the PDF
# `data/eRisk-2026/eRisk_2026__Preliminary_results-with-Task3.pdf` for INSA-Lyon.

OFFICIAL_PRELIMINARY_PDF = {
    "source_pdf": "data/eRisk-2026/eRisk_2026__Preliminary_results-with-Task3.pdf",
    "table": "Table 4 (incomplete submissions; INSA-Lyon covered 19/20 personas)",
    "latency_K0": LATENCY_K0,
    "team": "INSA-Lyon",
    "interaction_stats": {"mean_msgs_per_conv": 6.2, "mean_chars_per_msg": 156.96},
    "runs": {
        "Run_1": {"personas": "19/20", "DCHR": 0.3158, "ADODL": 0.8388, "ASHR": 0.1579, "LDCHR": 0.2285, "LASHR": 0.1142,
                  "ranks": {"DCHR": 10, "ADODL": 19, "ASHR": 8, "LDCHR": 8, "LASHR": 9}},
        "Run_2": {"personas": "19/20", "DCHR": 0.2632, "ADODL": 0.8304, "ASHR": 0.1447, "LDCHR": 0.1767, "LASHR": 0.1013,
                  "ranks": {"DCHR": 12, "ADODL": 21, "ASHR": 10, "LDCHR": 15, "LASHR": 11}},
        "Run_3": {"personas": "19/20", "DCHR": 0.2632, "ADODL": 0.8471, "ASHR": 0.2237, "LDCHR": 0.1826, "LASHR": 0.1525,
                  "ranks": {"DCHR": 12, "ADODL": 15, "ASHR": 4, "LDCHR": 14, "LASHR": 2}},
    },
}


def _aggregate_tom_condition(condition: str, per_persona_dir: Path) -> dict:
    persona_files = sorted(per_persona_dir.glob(f"{condition}_*.json"))
    per_persona = []
    for pf in persona_files:
        data = _load_json(pf)
        golden = int(data["golden_total"])
        predicted = int(data["predicted_total"])
        deviation = abs(predicted - golden)
        golden_band = data.get("golden_band") or _band_for(golden)
        predicted_band = data.get("predicted_band") or _band_for(predicted)
        per_persona.append({
            "name": data.get("persona", pf.stem.replace(f"{condition}_", "")),
            "golden": golden,
            "predicted": predicted,
            "golden_band": golden_band,
            "predicted_band": predicted_band,
            "band_ok": golden_band == predicted_band,
            "deviation": deviation,
            "cr": round(1 - deviation / BDI_MAX, 3),
            "predicted_top4": data.get("top4"),
            "turns_replayed": data.get("turns_replayed"),
            "persona_turns": data.get("persona_turns"),
            "timing_s": data.get("timing_s"),
        })
    if not per_persona:
        return {"config": condition, "n_personas": 0, "per_persona": []}

    n = len(per_persona)
    mad = round(sum(p["deviation"] for p in per_persona) / n, 2)
    adodl = round(sum(p["cr"] for p in per_persona) / n, 3)
    dchr = round(sum(1 for p in per_persona if p["band_ok"]) / n, 3)
    by_severity: dict[str, list[bool]] = {}
    for p in per_persona:
        by_severity.setdefault(p["golden_band"], []).append(p["band_ok"])
    band_accuracy_by_severity = {
        band: round(sum(flags) / len(flags), 3) for band, flags in by_severity.items()
    }
    return {
        "config": condition,
        "n_personas": n,
        "dchr": dchr,
        "mad": mad,
        "adodl": adodl,
        "ashr_proxy": None,
        "boundary_accuracy": None,
        "band_accuracy_by_severity": band_accuracy_by_severity,
        "per_persona": per_persona,
        "_aggregation_note": (
            "Re-aggregated from per-persona JSON files. ashr_proxy and boundary_accuracy "
            "cannot be recomputed locally (golden top-4 symptom keys not stored in the "
            "per-persona files); see runs/tom_ablation/tom_ablation_summary.json for the "
            "original partial values."
        ),
    }


SUBMISSION_RUN_CONFIG = {
    "Run_1": {
        "correction_strategy": "band_aware",
        "band_deltas": {"minimal": -4, "mild": -4, "moderate": -5, "severe": -1},
        "max_turns": 8,
        "rationale": "Safety run — optimizes ADODL (closeness ratio); best on TalkDep.",
        "ablation_proxy": {
            "source": "runs/ablation_comparison/ablation_summary.json",
            "config_name": "A0_band_aware",
        },
    },
    "Run_2": {
        "correction_strategy": "flat_minus_2",
        "delta": -2,
        "max_turns": 8,
        "rationale": "Calibrated risk — balanced MAD / DCHR trade-off.",
        "ablation_proxy": {
            "source": "runs/ablation_comparison/ablation_summary.json",
            "config_name": "A0_flat_minus_2",
        },
    },
    "Run_3": {
        "correction_strategy": "flat_minus_3",
        "delta": -3,
        "max_turns": 8,
        "rationale": "Balanced hedge — conservative across all metrics.",
        "ablation_proxy": None,
    },
}

SUBMISSION_RUN_PIPELINE = (
    "Full pipeline for all three submission runs: specialized assessors (×4, Llama-3.3-70B, "
    "T=0.1) + linguistic features + Bayesian prior + justificator + SDC. ToM tracking on "
    "for orchestrator guidance; C1/C2 ToM corrections disabled."
)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalise_ablation_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        out = []
        for config_name, body in payload.items():
            if isinstance(body, dict):
                body = dict(body)
                body.setdefault("config", config_name)
                out.append(body)
        return out
    raise TypeError(f"Unsupported ablation payload type: {type(payload).__name__}")


def collect_ablations() -> dict:
    sections = {}
    for spec in ABLATION_SOURCES:
        path = spec["path"]
        if not path.exists():
            sections[spec["key"]] = {
                "source_file": str(path.relative_to(REPO_ROOT).as_posix()),
                "description": spec["description"],
                "error": "missing file",
            }
            continue
        configs = _normalise_ablation_payload(_load_json(path))
        sections[spec["key"]] = {
            "source_file": str(path.relative_to(REPO_ROOT).as_posix()),
            "description": spec["description"],
            "n_personas_expected": spec["n_personas_expected"],
            "configs": configs,
        }

    tom_configs = []
    for condition, p_dir in TOM_ABLATION["per_persona_dirs"].items():
        if p_dir.exists():
            tom_configs.append(_aggregate_tom_condition(condition, p_dir))
    sections[TOM_ABLATION["key"]] = {
        "source_files": [
            str(TOM_ABLATION["summary_file"].relative_to(REPO_ROOT).as_posix()),
            *(
                str(d.relative_to(REPO_ROOT).as_posix())
                for d in TOM_ABLATION["per_persona_dirs"].values()
            ),
        ],
        "description": TOM_ABLATION["description"],
        "n_personas_expected": 12,
        "configs": tom_configs,
    }
    return sections


_SUBMISSION_RE = re.compile(r"^task1_results_(\d{8})(?:_(.+))?$")


def collect_submissions() -> list[dict]:
    task1_root = RUNS_DIR / "task1"
    if not task1_root.exists():
        return []
    batches = []
    for child in sorted(task1_root.iterdir()):
        if not child.is_dir():
            continue
        match = _SUBMISSION_RE.match(child.name)
        if not match:
            continue
        date_str, suffix = match.group(1), match.group(2)
        try:
            iso_date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
        except ValueError:
            iso_date = None
        persona_dirs = sorted(p for p in child.iterdir() if p.is_dir() and p.name.startswith("persona"))
        personas = []
        for p_dir in persona_dirs:
            persona_id = p_dir.name.replace("persona", "").lstrip("0") or "0"
            persona_entry = {
                "persona_dir": p_dir.name,
                "persona_id": int(persona_id),
                "predictions_by_run": {},
            }
            for run_id in (1, 2, 3):
                res_file = p_dir / f"results_{run_id}.json"
                if not res_file.exists():
                    continue
                data = _load_json(res_file)
                if isinstance(data, list):
                    persona_entry["predictions_by_run"][f"Run_{run_id}"] = data
                else:
                    persona_entry["predictions_by_run"][f"Run_{run_id}"] = [data]
            personas.append(persona_entry)
        batches.append({
            "folder": str(child.relative_to(REPO_ROOT).as_posix()),
            "date_iso": iso_date,
            "suffix": suffix,
            "is_tom_variant": suffix == "tom" if suffix else False,
            "personas": personas,
        })
    return batches


def build_consolidated() -> dict:
    return {
        "metadata": {
            "task": "eRisk 2026 Task 1 — Depression Interview Simulation (BDI-II scoring of LLM personas)",
            "repository_path": str(REPO_ROOT.as_posix()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_doc": "docs/task1_solution_description.md",
            "ablation_evaluation_set": (
                "TalkDep — 12 clinically-grounded LLM personas with golden BDI-II totals and "
                "top-4 symptoms (Wang et al., CIKM 2025). Personas: Noah, Maya, Priya, Ethan, "
                "Gabriel, Alex, James, Laura, Linda, Elena, Marco, Maria."
            ),
            "submission_evaluation_set": (
                "20 organizer-provided LoRA personas (persona00..persona19), processed in "
                "incremental batches and submitted via the eRisk server. Golden labels are "
                "held by the organizers and are not available locally — submission entries "
                "below therefore contain predicted BDI-II totals and key-symptoms only."
            ),
            "metric_definitions": METRIC_DEFINITIONS,
            "config_legend": {
                "A0_baseline": "Single generic assessor, no linguistic features, no Bayesian prior, no justificator.",
                "A1_specialized": "A0 + 4 specialized assessors (somatic / affective / cognitive / behavioural).",
                "A2_linguistic": "A1 + linguistic meta-features (absolutist density, minimizing language, etc.).",
                "A3_prior": "A2 + Bayesian prior over item severities.",
                "A4_justificator": "A3 + justificator agent that audits per-item coherence.",
                "A5_temp_sweep_low": "A4 with assessor T=0.05.",
                "A6_temp_sweep_high": "A4 with assessor T=0.3.",
                "A7_no_prior": "Full pipeline (A4) minus the Bayesian prior (isolates prior contribution).",
                "band_aware": "Post-hoc correction with band-specific deltas (Minimal/Mild: -4; Moderate: -5; Severe: -1).",
                "flat_minus_2": "Post-hoc: subtract 2 points from total.",
                "flat_minus_3": "Post-hoc: subtract 3 points from total.",
                "minus5": "Post-hoc: subtract 5 points from total.",
                "progressive": "Progressive (severity-dependent) correction.",
                "proportional_085": "Multiplicative correction ×0.85.",
                "sdc": "Score Distribution Constraint — downgrade lowest-confidence score-3 items when minimizing-language signals appear.",
                "tom_on": "Theory of Mind tracking + coverage-gap guidance + C1/C2 corrections enabled.",
                "tom_off": "Full pipeline with all ToM components disabled.",
            },
        },
        "ablations": collect_ablations(),
        "official_submissions": {
            "pipeline": SUBMISSION_RUN_PIPELINE,
            "runs": SUBMISSION_RUN_CONFIG,
            "official_preliminary_results": OFFICIAL_PRELIMINARY_PDF,
            "scored_locally_vs_golden": score_official_submission(OFFICIAL_SUBMISSION_ROOT),
            "evaluation_note": (
                "Two views of submission performance are now available: (a) the local "
                "re-scoring against data/eRisk-2026/.../task1-llms/golden-data/patients_data.jsonl "
                "(field `scored_locally_vs_golden`), and (b) the organizer-reported values "
                "from the preliminary PDF (field `official_preliminary_results`). Per-batch "
                "incremental folders are kept under `batches` for traceability."
            ),
            "batches": collect_submissions(),
        },
    }


def main() -> None:
    consolidated = build_consolidated()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    n_ablations = sum(
        len(section.get("configs", []))
        for section in consolidated["ablations"].values()
        if isinstance(section, dict)
    )
    n_batches = len(consolidated["official_submissions"]["batches"])
    n_personas = sum(len(b["personas"]) for b in consolidated["official_submissions"]["batches"])
    print(f"  ablation sections : {len(consolidated['ablations'])}")
    print(f"  ablation configs  : {n_ablations}")
    print(f"  submission batches: {n_batches}  ({n_personas} persona artefacts)")


if __name__ == "__main__":
    main()
