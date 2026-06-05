"""Compute the symptom-routing oracle MAP for eRisk 2026 Task 3.

Implements `specs/task-3/analysis-oracle-map.md`:
  * Reads per-symptom AP for 5 runs x 2 qrels from runs/task3_all_results.json
    (option (b) in the spec; trec_eval / pytrec_eval not available locally).
  * Cross-checks one (run, qrels) pair by recomputing AP from the TREC run
    file and qrels CSV using a trec_eval-compatible AP definition
    (unjudged = non-relevant; standard MAP formula).
  * Computes oracle_MAP(q) = mean over 18 symptoms of max_r AP(r, s, q).
  * Emits analyses/oracle_map_results.json and analyses/oracle_map_summary.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_JSON = REPO / "runs" / "task3_all_results.json"
RUNS_DIR = REPO / "runs" / "task3" / "task3-adhd-ranking-results"
QRELS_DIR = (
    REPO
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task3-adhd-symptom-ranking"
    / "golden-data"
)
QRELS_FILES = {
    "majority": QRELS_DIR / "qrels_majority-final.csv",
    "unanimity": QRELS_DIR / "qrels_unanimity-final.csv",
}
TREC_FILES = {
    "HiPerT_full": RUNS_DIR / "INSALyon_HiPerT_full.trec",
    "LLM_cascade": RUNS_DIR / "INSALyon_LLM_cascade.trec",
    "Ensemble": RUNS_DIR / "INSALyon_Ensemble.trec",
    "BiEnc_baseline": RUNS_DIR / "INSALyon_BiEnc_baseline.trec",
    "DepTransfer": RUNS_DIR / "INSALyon_DepTransfer.trec",
}
RUN_LABELS = {
    "HiPerT_full": "CADRE full",
    "LLM_cascade": "LLM cascade",
    "Ensemble": "Ensemble",
    "BiEnc_baseline": "BiEnc baseline",
    "DepTransfer": "DepTransfer",
}
RUN_ORDER = ["LLM_cascade", "HiPerT_full", "Ensemble", "DepTransfer", "BiEnc_baseline"]
BEST_ACTUAL = {
    "majority": ("Ensemble", 0.159),
    "unanimity": ("LLM_cascade", 0.134),
}
N_SYMPTOMS = 18


def load_qrels(path: Path) -> dict[int, dict[str, int]]:
    qrels: dict[int, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q = int(row["query"])
            rel = 1 if row["relevant"].strip().lower() == "true" else 0
            qrels.setdefault(q, {})[row["doc_id"]] = rel
    return qrels


def load_run(path: Path) -> dict[int, list[tuple[float, str]]]:
    """Return query -> list of (score, doc_id), unsorted."""
    run: dict[int, list[tuple[float, str]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            q = int(parts[0])
            doc = parts[2]
            score = float(parts[4])
            run.setdefault(q, []).append((score, doc))
    return run


def trec_eval_sort(run_q: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """trec_eval ordering: score descending, ties broken by docid lex-descending."""
    return sorted(run_q, key=lambda r: (-r[0], [-ord(c) for c in r[1]]))


def ap_for_query(run_q: list[tuple[float, str]], qrels_q: dict[str, int]) -> float:
    """AP with unjudged = non-relevant, trec_eval semantics."""
    n_rel = sum(1 for v in qrels_q.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    cum_precision = 0.0
    for k, (_, doc) in enumerate(trec_eval_sort(run_q), 1):
        if qrels_q.get(doc, 0) > 0:
            hits += 1
            cum_precision += hits / k
    return cum_precision / n_rel


def crosscheck_one(run_name: str, qrels_name: str, cached_per_symptom: dict) -> dict:
    """Recompute per-symptom AP for one (run, qrels) and compare to cache."""
    qrels = load_qrels(QRELS_FILES[qrels_name])
    run = load_run(TREC_FILES[run_name])
    recomputed: dict[str, float] = {}
    diffs: list[dict] = []
    for s in range(1, N_SYMPTOMS + 1):
        ap = ap_for_query(run.get(s, []), qrels.get(s, {}))
        recomputed[str(s)] = round(ap, 4)
        cached = float(cached_per_symptom[str(s)]["AP"])
        if abs(round(ap, 4) - cached) > 5e-4:
            diffs.append({"symptom": s, "recomputed": round(ap, 4), "cached": cached})
    map_recomputed = round(sum(recomputed.values()) / N_SYMPTOMS, 4)
    return {
        "run": run_name,
        "qrels": qrels_name,
        "MAP_recomputed": map_recomputed,
        "n_symptoms_mismatched": len(diffs),
        "mismatches": diffs,
        "per_symptom_recomputed": recomputed,
    }


def main() -> None:
    cache = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    scored = cache["scored_locally_vs_qrels"]
    asrs_names = cache["metadata"]["asrs_symptoms"]

    short_names = {
        1: "Wrap-up details", 2: "Order", 3: "Remember appts",
        4: "Avoid effortful tasks", 5: "Fidget hands/feet",
        6: "Over-active drive", 7: "Careless mistakes",
        8: "Sustain attention", 9: "Concentrate on speech",
        10: "Misplace items", 11: "Distracted by noise",
        12: "Leave seat", 13: "Restless/fidgety",
        14: "Hard to unwind", 15: "Talk too much",
        16: "Finish sentences", 17: "Wait turn",
        18: "Interrupt others",
    }

    # ---- Cross-check (sanity for option (b) per spec) ----
    crosscheck_majority_biencer = crosscheck_one(
        "BiEnc_baseline", "majority",
        scored["BiEnc_baseline"]["majority"]["per_symptom"],
    )
    crosscheck_unanim_llm = crosscheck_one(
        "LLM_cascade", "unanimity",
        scored["LLM_cascade"]["unanimity"]["per_symptom"],
    )

    # ---- Oracle MAP ----
    oracle: dict[str, dict] = {}
    per_run_pick_counts: dict[str, dict[str, int]] = {}
    per_symptom_table: list[dict] = []
    item13_ties: dict[str, list[str]] = {}

    for qrels_name in ("majority", "unanimity"):
        per_symptom_pick: dict[int, dict] = {}
        pick_counts = {r: 0 for r in RUN_ORDER}
        sum_oracle_ap = 0.0
        for s in range(1, N_SYMPTOMS + 1):
            best_ap = -1.0
            best_run = None
            ap_by_run: dict[str, float] = {}
            for r in RUN_ORDER:
                ap = float(scored[r][qrels_name]["per_symptom"][str(s)]["AP"])
                ap_by_run[r] = ap
                if ap > best_ap:
                    best_ap = ap
                    best_run = r
            # Detect ties at the max (use a small relative+absolute tolerance to
            # treat cached values rounded to 4dp as tied when essentially equal)
            tol = 5e-5
            top = [r for r in RUN_ORDER if abs(ap_by_run[r] - best_ap) <= tol]
            tie_note = top if len(top) > 1 else None
            if s == 13 and tie_note is not None:
                item13_ties[qrels_name] = tie_note
            per_symptom_pick[s] = {
                "selected_run": best_run,
                "selected_AP": round(best_ap, 4),
                "all_AP": {r: round(v, 4) for r, v in ap_by_run.items()},
                "tie": tie_note,
            }
            pick_counts[best_run] += 1
            sum_oracle_ap += best_ap
        oracle[qrels_name] = {
            "oracle_MAP": round(sum_oracle_ap / N_SYMPTOMS, 4),
            "best_actual_run": BEST_ACTUAL[qrels_name][0],
            "best_actual_MAP": BEST_ACTUAL[qrels_name][1],
            "headroom": round(sum_oracle_ap / N_SYMPTOMS - BEST_ACTUAL[qrels_name][1], 4),
            "per_run_pick_counts": pick_counts,
            "per_symptom_pick": per_symptom_pick,
        }

    # Combined per-symptom selection table
    for s in range(1, N_SYMPTOMS + 1):
        per_symptom_table.append({
            "symptom_id": s,
            "short_name": short_names[s],
            "asrs_description": asrs_names[str(s)],
            "selected_run_majority": oracle["majority"]["per_symptom_pick"][s]["selected_run"],
            "selected_AP_majority": oracle["majority"]["per_symptom_pick"][s]["selected_AP"],
            "tie_majority": oracle["majority"]["per_symptom_pick"][s]["tie"],
            "selected_run_unanimity": oracle["unanimity"]["per_symptom_pick"][s]["selected_run"],
            "selected_AP_unanimity": oracle["unanimity"]["per_symptom_pick"][s]["selected_AP"],
            "tie_unanimity": oracle["unanimity"]["per_symptom_pick"][s]["tie"],
        })

    # Symptoms where all 5 runs have AP exactly 0 (or within tol) — pick is arbitrary
    zero_tie_symptoms = {}
    for q in ("majority", "unanimity"):
        zs = []
        for s in range(1, N_SYMPTOMS + 1):
            aps = [float(scored[r][q]["per_symptom"][str(s)]["AP"]) for r in RUN_ORDER]
            if max(aps) <= 5e-5:
                zs.append(s)
        zero_tie_symptoms[q] = zs

    # ---- Sanity checks ----
    sanity = {
        "all_zero_tie_symptoms": zero_tie_symptoms,
        "note_on_expected_majority_mismatch": (
            "Expected counts from numerical-claims.md (BiEnc 6, CADRE 5, LLM 4, "
            "Ensemble 2, DepTransfer 1) require breaking 2 of the 3 all-zero ties "
            "(items 7, 9, 10) to HiPerT_full. The clean trec_eval-style argmax used "
            "here breaks ties by RUN_ORDER iteration (LLM_cascade first), giving "
            "LLM_cascade those slots. The oracle MAP is invariant to this choice "
            "(max=0 either way); only the per-run pick counts differ."
        ),
        "oracle_ge_best_actual": {
            q: oracle[q]["oracle_MAP"] >= oracle[q]["best_actual_MAP"]
            for q in ("majority", "unanimity")
        },
        "pick_counts_sum_to_18": {
            q: sum(oracle[q]["per_run_pick_counts"].values()) == N_SYMPTOMS
            for q in ("majority", "unanimity")
        },
        "expected_majority_counts_v_actual": {
            "expected": {"BiEnc_baseline": 6, "HiPerT_full": 5,
                         "LLM_cascade": 4, "Ensemble": 2, "DepTransfer": 1},
            "actual": oracle["majority"]["per_run_pick_counts"],
        },
        "item13_tie_notes": item13_ties,
        "crosscheck_majority_BiEnc_baseline": {
            "MAP_recomputed_from_TREC": crosscheck_majority_biencer["MAP_recomputed"],
            "MAP_cached": scored["BiEnc_baseline"]["majority"]["aggregate"]["AP"],
            "n_symptoms_mismatched": crosscheck_majority_biencer["n_symptoms_mismatched"],
            "mismatches": crosscheck_majority_biencer["mismatches"],
        },
        "crosscheck_unanimity_LLM_cascade": {
            "MAP_recomputed_from_TREC": crosscheck_unanim_llm["MAP_recomputed"],
            "MAP_cached": scored["LLM_cascade"]["unanimity"]["aggregate"]["AP"],
            "n_symptoms_mismatched": crosscheck_unanim_llm["n_symptoms_mismatched"],
            "mismatches": crosscheck_unanim_llm["mismatches"],
        },
    }

    out = {
        "generated_from": str(RESULTS_JSON.relative_to(REPO)),
        "spec": "specs/task-3/analysis-oracle-map.md",
        "headline": {
            q: {
                "qrels": q,
                "oracle_MAP": oracle[q]["oracle_MAP"],
                "best_actual_MAP": oracle[q]["best_actual_MAP"],
                "best_actual_run": oracle[q]["best_actual_run"],
                "headroom": oracle[q]["headroom"],
            }
            for q in ("majority", "unanimity")
        },
        "per_run_pick_counts": {
            q: oracle[q]["per_run_pick_counts"] for q in ("majority", "unanimity")
        },
        "per_symptom_selection_table": per_symptom_table,
        "sanity": sanity,
    }
    out_json = REPO / "analyses" / "oracle_map_results.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- Markdown summary ----
    md = []
    md.append("# Task 3 Symptom-Routing Oracle MAP")
    md.append("")
    md.append("Computed per `specs/task-3/analysis-oracle-map.md`. Inputs: per-symptom "
              "AP from `runs/task3_all_results.json` (option (b)); cross-checked against "
              "TREC-file recomputation for two (run, qrels) pairs.")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append("| qrels | oracle_MAP | best_actual_run | best_actual_MAP | headroom |")
    md.append("|---|---:|---|---:|---:|")
    for q in ("majority", "unanimity"):
        h = out["headline"][q]
        md.append(
            f"| {q} | {h['oracle_MAP']:.4f} | {RUN_LABELS[h['best_actual_run']]} | "
            f"{h['best_actual_MAP']:.3f} | +{h['headroom']:.4f} |"
        )
    md.append("")
    md.append("## Per-symptom oracle selection")
    md.append("")
    md.append("| # | short name | majority pick | AP (maj) | unanimity pick | AP (unan) |")
    md.append("|---:|---|---|---:|---|---:|")
    for row in per_symptom_table:
        maj = RUN_LABELS[row["selected_run_majority"]]
        if row["tie_majority"]:
            maj += " *tie*"
        unan = RUN_LABELS[row["selected_run_unanimity"]]
        if row["tie_unanimity"]:
            unan += " *tie*"
        md.append(
            f"| {row['symptom_id']} | {row['short_name']} | {maj} | "
            f"{row['selected_AP_majority']:.4f} | {unan} | {row['selected_AP_unanimity']:.4f} |"
        )
    md.append("")
    md.append("## Per-run selection counts")
    md.append("")
    md.append("| run | majority | unanimity |")
    md.append("|---|---:|---:|")
    for r in RUN_ORDER:
        md.append(
            f"| {RUN_LABELS[r]} | "
            f"{oracle['majority']['per_run_pick_counts'][r]} | "
            f"{oracle['unanimity']['per_run_pick_counts'][r]} |"
        )
    md.append("")
    md.append("## Sanity checks")
    md.append("")
    md.append(f"- Oracle ≥ best actual: majority={sanity['oracle_ge_best_actual']['majority']}, "
              f"unanimity={sanity['oracle_ge_best_actual']['unanimity']}")
    md.append(f"- Pick counts sum to 18: majority={sanity['pick_counts_sum_to_18']['majority']}, "
              f"unanimity={sanity['pick_counts_sum_to_18']['unanimity']}")
    md.append(f"- Expected majority counts (from `numerical-claims.md`): "
              f"{sanity['expected_majority_counts_v_actual']['expected']}")
    md.append(f"- Actual majority counts: "
              f"{sanity['expected_majority_counts_v_actual']['actual']}")
    if item13_ties:
        md.append(f"- Item 13 ties recorded: {item13_ties}")
    else:
        md.append("- Item 13 ties: none recorded at machine precision.")
    cc1 = sanity["crosscheck_majority_BiEnc_baseline"]
    cc2 = sanity["crosscheck_unanimity_LLM_cascade"]
    md.append(
        f"- Cross-check BiEnc_baseline/majority: "
        f"recomputed MAP {cc1['MAP_recomputed_from_TREC']:.4f} vs cached "
        f"{cc1['MAP_cached']:.4f}; per-symptom mismatches: {cc1['n_symptoms_mismatched']}."
    )
    md.append(
        f"- Cross-check LLM_cascade/unanimity: "
        f"recomputed MAP {cc2['MAP_recomputed_from_TREC']:.4f} vs cached "
        f"{cc2['MAP_cached']:.4f}; per-symptom mismatches: {cc2['n_symptoms_mismatched']}."
    )
    out_md = REPO / "analyses" / "oracle_map_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(out["headline"], indent=2))
    print("Per-run pick counts (majority):", oracle["majority"]["per_run_pick_counts"])
    print("Per-run pick counts (unanimity):", oracle["unanimity"]["per_run_pick_counts"])
    print(f"Wrote {out_json.relative_to(REPO)} and {out_md.relative_to(REPO)}")
    print("Cross-check BiEnc/majority: recomputed=",
          cc1["MAP_recomputed_from_TREC"], "cached=", cc1["MAP_cached"],
          "mismatches=", cc1["n_symptoms_mismatched"])
    print("Cross-check LLM/unanimity: recomputed=",
          cc2["MAP_recomputed_from_TREC"], "cached=", cc2["MAP_cached"],
          "mismatches=", cc2["n_symptoms_mismatched"])


if __name__ == "__main__":
    main()
