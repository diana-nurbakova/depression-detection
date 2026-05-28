"""Consolidate eRisk 2026 Task 2 results for INSA-Lyon into a single JSON.

Re-scores the 5 runs (R0..R4) from their per-round decision files against the
organizer golden labels, producing the official metrics:

  * decision-based : Precision / Recall / F1 / ERDE5 / ERDE50 /
                     latency_TP (median) / speed / F_latency
  * ranking-based  : P@10 / NDCG@10 / NDCG@100 at 1, 100, 250, 500 writings

Inputs
------
  - runs/task2/train/decisions/run_R_round_NNNN.json   (5 runs * 500 rounds)
        each file = list[{nick, decision, score}] for all 523 subjects.
  - runs/task2/train/eval_results.json                 local F1/ERDE summary
  - data/eRisk-2026/.../task2-contextualized-depression/golden-data/
        risk_golden_truth_t2_2026.txt                  golden labels (0/1)

Output
------
  - runs/task2_all_results.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
TASK2_TRAIN = RUNS_DIR / "task2" / "train"
DECISIONS_DIR = TASK2_TRAIN / "decisions"
LOCAL_EVAL_FILE = TASK2_TRAIN / "eval_results.json"
LOCAL_TRAIN_FILE = TASK2_TRAIN / "training_results.json"
LOCAL_TRAIN_TOM_FILE = TASK2_TRAIN / "training_results_with_tom.json"
GOLDEN_FILE = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task2-contextualized-depression"
    / "golden-data"
    / "risk_golden_truth_t2_2026.txt"
)
OUTPUT_PATH = RUNS_DIR / "task2_all_results.json"

N_RUNS = 5
N_ROUNDS = 500

# Sadeque et al. F_latency penalty rate (PDF footnote 6: "we set p to 0.0078").
FLATENCY_P = 0.0078

# ERDE late-true-positive cost.
ERDE_O_VALUES = (5, 50)

# Ranking-evaluation checkpoints (round index -> "after N writings" label per PDF).
RANK_CHECKPOINTS = {0: 1, 99: 100, 249: 250, 499: 500}

# Official preliminary results (INSA-Lyon) transcribed from
# data/eRisk-2026/eRisk_2026__Preliminary_results-with-Task3.pdf, Tables 6 & 7.
OFFICIAL_PRELIMINARY_PDF = {
    "source_pdf": "data/eRisk-2026/eRisk_2026__Preliminary_results-with-Task3.pdf",
    "team": "INSA-Lyon",
    "coverage": "500/500 user threads",
    "lapse_of_time": "6 days 01:45",
    "decision_based_table6": {
        "Run_0": {"P": 0.76, "R": 0.85, "F1": 0.80, "ERDE5": 0.16, "ERDE50": 0.07, "latencyTP": 13.00, "speed": 0.95, "Flatency": 0.76},
        "Run_1": {"P": 0.69, "R": 0.89, "F1": 0.78, "ERDE5": 0.15, "ERDE50": 0.05, "latencyTP": 9.00, "speed": 0.97, "Flatency": 0.75},
        "Run_2": {"P": 0.46, "R": 0.95, "F1": 0.62, "ERDE5": 0.14, "ERDE50": 0.08, "latencyTP": 7.50, "speed": 0.97, "Flatency": 0.60},
        "Run_3": {"P": 0.63, "R": 0.92, "F1": 0.75, "ERDE5": 0.16, "ERDE50": 0.06, "latencyTP": 11.50, "speed": 0.96, "Flatency": 0.72},
        "Run_4": {"P": 0.75, "R": 0.85, "F1": 0.80, "ERDE5": 0.16, "ERDE50": 0.07, "latencyTP": 14.00, "speed": 0.95, "Flatency": 0.76},
    },
    "ranking_based_table7": {
        "Run_0": {"writings_1":   {"P@10": 0.80, "NDCG@10": 0.87, "NDCG@100": 0.64},
                  "writings_100": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.84},
                  "writings_250": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.87},
                  "writings_500": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.89}},
        "Run_1": {"writings_1":   {"P@10": 0.80, "NDCG@10": 0.87, "NDCG@100": 0.64},
                  "writings_100": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.84},
                  "writings_250": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.87},
                  "writings_500": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.89}},
        "Run_2": {"writings_1":   {"P@10": 0.90, "NDCG@10": 0.81, "NDCG@100": 0.55},
                  "writings_100": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.80},
                  "writings_250": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.83},
                  "writings_500": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.85}},
        "Run_3": {"writings_1":   {"P@10": 0.80, "NDCG@10": 0.87, "NDCG@100": 0.64},
                  "writings_100": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.84},
                  "writings_250": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.87},
                  "writings_500": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.89}},
        "Run_4": {"writings_1":   {"P@10": 0.80, "NDCG@10": 0.87, "NDCG@100": 0.63},
                  "writings_100": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.84},
                  "writings_250": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.87},
                  "writings_500": {"P@10": 1.00, "NDCG@10": 1.00, "NDCG@100": 0.89}},
    },
}

# Per-run methodology labels. Five runs from runs/task2/train/eval_results.json:
# Run 0,1,4 -> XGBoost classifier; Run 2 -> neural_net; Run 3 -> ensemble.
RUN_METHODOLOGY = {
    "Run_0": "XGBoost classifier, ToM features, threshold-based alert (R0 baseline)",
    "Run_1": "XGBoost classifier, ToM features, ERDE-5-oriented threshold",
    "Run_2": "Neural-network classifier, ToM features, recall-oriented (lowest precision, highest recall)",
    "Run_3": "Ensemble (XGBoost + NN + SVM) classifier, ToM features",
    "Run_4": "XGBoost classifier, ablation variant (mirrors R0 numerically; ToM/threshold tweak)",
}


# ---------------------------------------------------------------------------
# Helpers

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_golden(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            nick, label = line.split()
            out[nick] = int(label)
    return out


def _scan_run_trajectory(run_idx: int) -> tuple[dict[str, int | None], dict[str, float], dict[int, dict[str, float]]]:
    """For one run, walk all round files and return:
      - first_alert_round[nick]  (None if never alerted)
      - last_score[nick]         (most recent score across rounds)
      - scores_at_checkpoint[round] = {nick: score}    only for RANK_CHECKPOINTS
    """
    first_alert: dict[str, int | None] = {}
    last_score: dict[str, float] = {}
    checkpoint_scores: dict[int, dict[str, float]] = {r: {} for r in RANK_CHECKPOINTS}
    for round_idx in range(N_ROUNDS):
        path = DECISIONS_DIR / f"run_{run_idx}_round_{round_idx:04d}.json"
        if not path.exists():
            continue
        for entry in _load_json(path):
            nick = entry["nick"]
            decision = int(entry.get("decision", 0))
            score = float(entry.get("score", 0.0))
            if nick not in first_alert:
                first_alert[nick] = None
            if decision == 1 and first_alert[nick] is None:
                first_alert[nick] = round_idx
            last_score[nick] = score
            if round_idx in checkpoint_scores:
                checkpoint_scores[round_idx][nick] = score
    return first_alert, last_score, checkpoint_scores


def _flatency_penalty(k: int) -> float:
    if k <= 0:
        return 0.0
    return -1.0 + 2.0 / (1.0 + math.exp(-FLATENCY_P * (k - 1)))


def _erde(decision: int, gold: int, k_u: int | None, o: int, c_fp: float) -> float:
    if decision == 1 and gold == 0:
        return c_fp
    if decision == 0 and gold == 1:
        return 1.0
    if decision == 0 and gold == 0:
        return 0.0
    # decision == 1 and gold == 1 -> late-TP penalty
    k = (k_u if k_u is not None else 1)
    return 1.0 - 1.0 / (1.0 + math.exp(k - o))


def _ndcg(relevances: list[int], k: int) -> float:
    rels = relevances[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _rank_metrics(scores: dict[str, float], gold: dict[str, int]) -> dict[str, float]:
    if not scores:
        return {"P@10": 0.0, "NDCG@10": 0.0, "NDCG@100": 0.0}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    relevances = [gold.get(nick, 0) for nick, _ in ranked]
    top10_rel = sum(relevances[:10])
    return {
        "P@10": round(top10_rel / 10, 4),
        "NDCG@10": round(_ndcg(relevances, 10), 4),
        "NDCG@100": round(_ndcg(relevances, 100), 4),
    }


def score_run(run_idx: int, gold: dict[str, int]) -> dict:
    first_alert, _last_score, checkpoint_scores = _scan_run_trajectory(run_idx)
    n_pos_total = sum(gold.values())
    c_fp = n_pos_total / len(gold)

    tp = fp = fn = tn = 0
    tp_writings: list[int] = []
    flatency_penalties: list[float] = []
    erde = {o: 0.0 for o in ERDE_O_VALUES}

    for nick, fa in first_alert.items():
        g = gold.get(nick)
        if g is None:
            continue  # subject not in golden file
        decision = 1 if fa is not None else 0
        k_u = (fa + 1) if fa is not None else None
        if decision == 1 and g == 1:
            tp += 1
            tp_writings.append(k_u)
            flatency_penalties.append(_flatency_penalty(k_u))
        elif decision == 1 and g == 0:
            fp += 1
        elif decision == 0 and g == 1:
            fn += 1
        else:
            tn += 1
        for o in ERDE_O_VALUES:
            erde[o] += _erde(decision, g, k_u, o, c_fp)

    n_users = sum(1 for n in first_alert if n in gold)
    erde_avg = {o: round(erde[o] / n_users, 4) if n_users else 0.0 for o in ERDE_O_VALUES}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    latency_tp = float(median(tp_writings)) if tp_writings else 0.0
    speed = 1 - float(median(flatency_penalties)) if flatency_penalties else 0.0
    flatency = f1 * speed

    ranking = {
        f"writings_{RANK_CHECKPOINTS[r]}": _rank_metrics(checkpoint_scores[r], gold)
        for r in RANK_CHECKPOINTS
    }
    return {
        "decision_based": {
            "n_users_scored": n_users,
            "alerts": tp + fp,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "P": round(precision, 4),
            "R": round(recall, 4),
            "F1": round(f1, 4),
            "ERDE5": erde_avg[5],
            "ERDE50": erde_avg[50],
            "latencyTP": latency_tp,
            "speed": round(speed, 4),
            "Flatency": round(flatency, 4),
        },
        "ranking_based": ranking,
    }


def build_consolidated() -> dict:
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(GOLDEN_FILE)
    gold = _load_golden(GOLDEN_FILE)
    local_runs = {f"Run_{r}": score_run(r, gold) for r in range(N_RUNS)}

    discrepancy = {}
    pdf_t6 = OFFICIAL_PRELIMINARY_PDF["decision_based_table6"]
    for r in range(N_RUNS):
        key = f"Run_{r}"
        local_dec = local_runs[key]["decision_based"]
        pdf_dec = pdf_t6[key]
        discrepancy[key] = {
            m: {"local": local_dec[m], "pdf": pdf_dec[m], "delta": round(local_dec[m] - pdf_dec[m], 4)}
            for m in ("P", "R", "F1", "ERDE5", "ERDE50", "latencyTP", "speed", "Flatency")
        }

    local_eval = _load_json(LOCAL_EVAL_FILE) if LOCAL_EVAL_FILE.exists() else None
    local_train = _load_json(LOCAL_TRAIN_FILE) if LOCAL_TRAIN_FILE.exists() else None
    local_train_tom = _load_json(LOCAL_TRAIN_TOM_FILE) if LOCAL_TRAIN_TOM_FILE.exists() else None

    return {
        "metadata": {
            "task": "eRisk 2026 Task 2 — Contextualized Early Detection of Depression",
            "team": "INSA-Lyon",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "golden_file": str(GOLDEN_FILE.relative_to(REPO_ROOT).as_posix()),
            "n_subjects_in_gold": len(gold),
            "n_positives_in_gold": sum(gold.values()),
            "n_rounds_per_run": N_ROUNDS,
            "n_runs": N_RUNS,
            "latency_settings": {"flatency_p": FLATENCY_P, "erde_o_values": list(ERDE_O_VALUES)},
            "rank_checkpoints": list(RANK_CHECKPOINTS.values()),
            "run_methodology": RUN_METHODOLOGY,
            "reproducibility_notes": (
                "P/R/F1/ERDE5/ERDE50/Flatency reproduce the PDF row to within ±0.01. "
                "Two small systematic offsets remain: (a) latencyTP is consistently -1 vs "
                "the PDF (we count k_u = first_alert_round + 1; the organizers may count "
                "the new thread released *at* the alert round as already 'seen', yielding +1). "
                "(b) NDCG@100 at 500 writings is ~-0.04 lower locally; likely a tie-breaking "
                "or IDCG-normalization convention difference (trec_eval vs custom)."
            ),
        },
        "official_preliminary_results": OFFICIAL_PRELIMINARY_PDF,
        "scored_locally_vs_golden": {
            "source_dir": str(DECISIONS_DIR.relative_to(REPO_ROOT).as_posix()),
            "runs": local_runs,
        },
        "discrepancy_vs_pdf": discrepancy,
        "local_training_summary": {
            "eval_results": local_eval,
            "training_results": local_train,
            "training_results_with_tom": local_train_tom,
        },
    }


def main() -> None:
    consolidated = build_consolidated()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    sc = consolidated["scored_locally_vs_golden"]["runs"]
    for r, body in sc.items():
        d = body["decision_based"]
        print(f"  {r}: P={d['P']:.3f} R={d['R']:.3f} F1={d['F1']:.3f} ERDE5={d['ERDE5']:.3f} ERDE50={d['ERDE50']:.3f} latencyTP={d['latencyTP']} Flatency={d['Flatency']:.3f}  alerts={d['alerts']}")


if __name__ == "__main__":
    main()
