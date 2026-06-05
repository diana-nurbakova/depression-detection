"""Compute per-run alert counts, TP latency, and per-subject latency for Task 2.

Outputs JSON with G2.1, G2.2, G2.3, G2.4 metrics for the DUET paper data-extraction request.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS_DIR = ROOT / "runs" / "task2" / "train" / "decisions"
GOLD_PATH = ROOT / "data" / "eRisk-2026" / "eRisk26-datasets-20260519T175618Z-3-001" / "eRisk26-datasets" / "task2-contextualized-depression" / "golden-data" / "risk_golden_truth_t2_2026.txt"


def load_gold(path: Path) -> dict[str, int]:
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        subj, label = parts[0], int(parts[1])
        gold[subj] = label
    return gold


def first_alert_per_subject(run_id: int) -> dict[str, int]:
    """Return {subject: first_round_with_decision==1} for the given run."""
    first_alert: dict[str, int] = {}
    files = sorted(DECISIONS_DIR.glob(f"run_{run_id}_round_*.json"))
    for f in files:
        # Parse round number from filename like run_0_round_0042.json
        round_num = int(f.stem.split("_round_")[1])
        with f.open(encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            subj = entry["nick"]
            if entry.get("decision", 0) == 1 and subj not in first_alert:
                first_alert[subj] = round_num
    return first_alert


def summarise_run(run_id: int, gold: dict[str, int]) -> dict:
    first_alert = first_alert_per_subject(run_id)
    total_alerts = len(first_alert)
    tps, fps = [], []
    tp_rounds, all_alert_rounds = [], []
    for subj, rnd in first_alert.items():
        all_alert_rounds.append(rnd)
        g = gold.get(subj)
        if g == 1:
            tps.append(subj)
            tp_rounds.append(rnd)
        elif g == 0:
            fps.append(subj)
    return {
        "run": f"R{run_id}",
        "total_alerts": total_alerts,
        "true_positives": len(tps),
        "false_positives": len(fps),
        "median_tp_first_alert_round": statistics.median(tp_rounds) if tp_rounds else None,
        "mean_tp_first_alert_round": statistics.fmean(tp_rounds) if tp_rounds else None,
        "median_alert_round_all_alerts": statistics.median(all_alert_rounds) if all_alert_rounds else None,
        "mean_alert_round_all_alerts": statistics.fmean(all_alert_rounds) if all_alert_rounds else None,
        "_first_alert": first_alert,
        "_tp_rounds": tp_rounds,
    }


def per_subject_rows(subjects: list[str], summaries: list[dict], gold: dict[str, int]) -> list[dict]:
    rows = []
    for subj in subjects:
        row = {"subject": subj, "gold": gold.get(subj)}
        for s in summaries:
            run = s["run"]
            row[f"{run}_alert_round"] = s["_first_alert"].get(subj)
        rows.append(row)
    return rows


def early_alert_cohort(r0_first: dict[str, int], r1_first: dict[str, int],
                       gold: dict[str, int], threshold: int = 5) -> dict:
    """Subjects where gold=1 AND R0 alerted AND R1 alerted ≥threshold rounds before R0."""
    cohort = []
    for subj, r0_round in r0_first.items():
        if gold.get(subj) != 1:
            continue
        r1_round = r1_first.get(subj)
        if r1_round is None:
            continue
        gap = r0_round - r1_round
        if gap >= threshold:
            cohort.append({
                "subject": subj,
                "r0_round": r0_round,
                "r1_round": r1_round,
                "gap": gap,
            })
    cohort.sort(key=lambda x: -x["gap"])
    gaps = [c["gap"] for c in cohort]
    return {
        "n_qualifying": len(cohort),
        "median_gap": statistics.median(gaps) if gaps else None,
        "mean_gap": statistics.fmean(gaps) if gaps else None,
        "members": cohort,
    }


def main() -> None:
    gold = load_gold(GOLD_PATH)
    print(f"Loaded gold: {len(gold)} subjects, "
          f"{sum(1 for v in gold.values() if v == 1)} positives, "
          f"{sum(1 for v in gold.values() if v == 0)} negatives")

    summaries = [summarise_run(r, gold) for r in range(5)]

    # G2.1, G2.2, G2.4 are derived from summaries.
    g21 = []
    g22 = {}
    g24 = {}
    for s in summaries:
        g21.append({
            "run": s["run"],
            "total_alerts": s["total_alerts"],
            "true_positives": s["true_positives"],
            "false_positives": s["false_positives"],
        })
        g22[s["run"]] = s["median_tp_first_alert_round"]
        g24[s["run"]] = {
            "mean_tp_first_alert_round": s["mean_tp_first_alert_round"],
            "mean_alert_round_all_alerts": s["mean_alert_round_all_alerts"],
        }

    # G2.3 — per-subject rows.
    target_subjects = ["subject_qMXSpL4", "subject_kcdgN0X", "subject_OvzSZuo", "subject_Pop2mTP"]
    g23_rows = per_subject_rows(target_subjects, summaries, gold)

    # Early-alert cohort.
    early_alert = early_alert_cohort(
        summaries[0]["_first_alert"],
        summaries[1]["_first_alert"],
        gold,
        threshold=5,
    )

    out = {
        "data_source": {
            "decisions_dir": str(DECISIONS_DIR),
            "gold": str(GOLD_PATH),
        },
        "G2_1_per_run_alert_counts": g21,
        "G2_2_per_run_median_tp_latency": g22,
        "G2_3_per_subject_latency": g23_rows,
        "G2_3_early_alert_cohort_R1_vs_R0_gap_ge_5": early_alert,
        "G2_4_per_run_mean_alert_round": g24,
    }

    out_path = ROOT / "docs" / "task2_data_extraction_local.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Echo highlights.
    print("\nG2.1 — Per-run alert counts:")
    for row in g21:
        print(f"  {row['run']}: total={row['total_alerts']}, TP={row['true_positives']}, FP={row['false_positives']}")
    print("\nG2.2 — Median TP first-alert round:")
    for run, val in g22.items():
        print(f"  {run}: {val}")
    print("\nG2.4 — Mean alert round (TP only / all alerts):")
    for run, val in g24.items():
        print(f"  {run}: mean_TP={val['mean_tp_first_alert_round']:.2f}, "
              f"mean_all={val['mean_alert_round_all_alerts']:.2f}")
    print(f"\nEarly-alert cohort (R1 ≥5 rounds before R0): "
          f"n={early_alert['n_qualifying']}, median gap={early_alert['median_gap']}")


if __name__ == "__main__":
    main()
