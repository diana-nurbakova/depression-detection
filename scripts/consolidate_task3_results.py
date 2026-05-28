"""Consolidate eRisk 2026 Task 3 results for INSA-Lyon into a single JSON.

Scores the 5 submitted TREC runs against the organizer's qrels (majority and
unanimity), producing per-symptom and aggregate AP / R-Prec / P@10 / NDCG.

Inputs
------
  - runs/task3/task3-adhd-ranking-results/INSALyon_<RUN>.trec   (5 files)
        TREC format: ``qid Q0 docid rank score runtag``
  - data/eRisk-2026/.../task3-adhd-symptom-ranking/golden-data/
        qrels_majority-final.csv      (~5224 rows, 751 relevant)
        qrels_unanimity-final.csv     (~5224 rows, ~483 relevant)

Output
------
  - runs/task3_all_results.json
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
TREC_DIR = RUNS_DIR / "task3" / "task3-adhd-ranking-results"
QRELS_ROOT = (
    REPO_ROOT
    / "data"
    / "eRisk-2026"
    / "eRisk26-datasets-20260519T175618Z-3-001"
    / "eRisk26-datasets"
    / "task3-adhd-symptom-ranking"
    / "golden-data"
)
QRELS_FILES = {
    "majority": QRELS_ROOT / "qrels_majority-final.csv",
    "unanimity": QRELS_ROOT / "qrels_unanimity-final.csv",
}
OUTPUT_PATH = RUNS_DIR / "task3_all_results.json"

# ASRS-v1.1 symptom names matched to the 18 query IDs the team used during
# retrieval. Source: src/hipert/config.py / specs/asrs_four_layer_definitions.md.
ASRS_SYMPTOMS = {
    1:  "Trouble wrapping up the final details of a project",
    2:  "Difficulty getting things in order",
    3:  "Problems remembering appointments or obligations",
    4:  "Avoiding or delaying tasks requiring a lot of thought",
    5:  "Fidgeting or squirming with hands or feet",
    6:  "Feeling overly active and compelled to do things",
    7:  "Making careless mistakes on boring or difficult work",
    8:  "Difficulty keeping attention on boring or repetitive work",
    9:  "Difficulty concentrating on what people say",
    10: "Misplacing or having difficulty finding things",
    11: "Distracted by activity or noise around you",
    12: "Leaving your seat in meetings or situations where seated is expected",
    13: "Feeling restless or fidgety",
    14: "Difficulty unwinding and relaxing when you have time",
    15: "Talking too much in social situations",
    16: "Finishing the sentences of others before they finish themselves",
    17: "Difficulty waiting your turn",
    18: "Interrupting others when they are busy",
}

# 5 submitted run files -> short name used in PDF.
RUN_FILES = {
    "HiPerT_full":   TREC_DIR / "INSALyon_HiPerT_full.trec",
    "LLM_cascade":   TREC_DIR / "INSALyon_LLM_cascade.trec",
    "Ensemble":      TREC_DIR / "INSALyon_Ensemble.trec",
    "BiEnc_baseline":TREC_DIR / "INSALyon_BiEnc_baseline.trec",
    "DepTransfer":   TREC_DIR / "INSALyon_DepTransfer.trec",
}

# Official preliminary results from the PDF (Tables 10 & 11).
OFFICIAL_PRELIMINARY_PDF = {
    "source_pdf": "data/eRisk-2026/eRisk_2026__Preliminary_results-with-Task3.pdf",
    "team": "INSA-Lyon",
    "n_teams_in_field": 12,
    "n_runs_in_field": 45,
    "majority_table10": {
        "Ensemble":       {"AP": 0.159, "R-PREC": 0.206, "P@10": 0.317, "NDCG": 0.406, "rank_AP": 7},
        "LLM_cascade":    {"AP": 0.158, "R-PREC": 0.180, "P@10": 0.344, "NDCG": 0.402, "rank_AP": 8},
        "HiPerT_full":    {"AP": 0.137, "R-PREC": 0.169, "P@10": 0.294, "NDCG": 0.388, "rank_AP": 9},
        "BiEnc_baseline": {"AP": 0.117, "R-PREC": 0.160, "P@10": 0.311, "NDCG": 0.321, "rank_AP": 10},
        "DepTransfer":    {"AP": 0.054, "R-PREC": 0.087, "P@10": 0.100, "NDCG": 0.216, "rank_AP": 22},
    },
    "unanimity_table11": {
        "LLM_cascade":    {"AP": 0.134, "R-PREC": 0.167, "P@10": 0.272, "NDCG": 0.369, "rank_AP": 5},
        "Ensemble":       {"AP": 0.129, "R-PREC": 0.160, "P@10": 0.244, "NDCG": 0.363, "rank_AP": 7},
        "BiEnc_baseline": {"AP": 0.122, "R-PREC": 0.154, "P@10": 0.256, "NDCG": 0.313, "rank_AP": 8},
        "HiPerT_full":    {"AP": 0.107, "R-PREC": 0.131, "P@10": 0.172, "NDCG": 0.336, "rank_AP": 10},
        "DepTransfer":    {"AP": 0.054, "R-PREC": 0.065, "P@10": 0.078, "NDCG": 0.202, "rank_AP": 23},
    },
}

RUN_METHODOLOGY = {
    "HiPerT_full":    "Trained encoder ensemble (Stage A BDI-Sen + eRisk-25 T1; Stage B ADHD silver). Three backbones averaged.",
    "LLM_cascade":    "LLM scores per candidate + confidence + cosine tie-breaking on bi-encoder embeddings.",
    "Ensemble":       "Reciprocal-rank fusion (RRF) of HiPerT_full + LLM_cascade, falling back to BiEnc_baseline + LLM_cascade if HiPerT unavailable.",
    "BiEnc_baseline": "Cosine similarity from candidate retrieval (bi-encoder, no LLM step).",
    "DepTransfer":    "Stage-A-only encoder, BDI-II→ASRS mapping for 12/18 items (cross-disorder transfer baseline).",
}


# ---------------------------------------------------------------------------

def _load_qrels(path: Path) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qid = int(row["query"])
            docid = row["doc_id"]
            rel = 1 if row["relevant"].strip().lower() == "true" else 0
            out[qid][docid] = rel
    return out


def _load_trec(path: Path) -> dict[int, list[str]]:
    """Return {qid: [docid_in_rank_order]} using trec_eval tie-breaking.

    trec_eval ignores the ``rank`` column in the submission file and re-sorts by
    ``score`` descending, breaking ties by ``docid`` *descending* (reverse lex
    order). This matters for runs with heavy score ties (e.g. our LLM_cascade,
    where query 1 only has 11 unique scores across 1000 docs).
    """
    rows: dict[int, list[tuple[float, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid = int(parts[0])
            docid = parts[2]
            score = float(parts[4])
            rows[qid].append((score, docid))
    out: dict[int, list[str]] = {}
    for qid, lst in rows.items():
        # Stable two-pass sort: docid descending first, score descending second.
        # On a tie in score, the docid-descending order is preserved.
        lst.sort(key=lambda t: t[1], reverse=True)
        lst.sort(key=lambda t: t[0], reverse=True)
        out[qid] = [docid for _, docid in lst]
    return out


def _per_query_metrics(ranking: list[str], qrel: dict[str, int]) -> dict[str, float]:
    """Compute AP, R-Prec, P@10, NDCG (full-cutoff) for one query.

    Follows trec_eval conventions: unjudged docs are treated as non-relevant
    (rel=0) and contribute to denominators in P@K but not as hits. NDCG cutoff
    is the full ranking length (trec_eval ``ndcg`` measure, gain=relevance,
    discount=log2(rank+1) with rank starting at 1).
    """
    R = sum(qrel.values())
    if R == 0:
        return {"AP": 0.0, "R-PREC": 0.0, "P@10": 0.0, "NDCG": 0.0}

    n_hits = 0
    ap_sum = 0.0
    dcg = 0.0
    for k, docid in enumerate(ranking, start=1):
        rel = qrel.get(docid, 0)
        if rel == 1:
            n_hits += 1
            ap_sum += n_hits / k
        dcg += rel / math.log2(k + 1)
    ap = ap_sum / R
    p10 = sum(1 for d in ranking[:10] if qrel.get(d, 0) == 1) / 10
    r_prec_top = ranking[:R] if R <= len(ranking) else ranking
    r_prec = sum(1 for d in r_prec_top if qrel.get(d, 0) == 1) / R

    ideal_relevances = sorted(qrel.values(), reverse=True)
    cutoff = max(len(ranking), R)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_relevances[:cutoff]))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"AP": ap, "R-PREC": r_prec, "P@10": p10, "NDCG": ndcg}


def score_run(run_path: Path, qrels: dict[int, dict[str, int]]) -> dict:
    ranking = _load_trec(run_path)
    per_q: dict[int, dict[str, float]] = {}
    for qid in sorted(qrels):
        per_q[qid] = _per_query_metrics(ranking.get(qid, []), qrels[qid])
    n = len(per_q)
    aggregate = {
        m: round(sum(q[m] for q in per_q.values()) / n, 4)
        for m in ("AP", "R-PREC", "P@10", "NDCG")
    }
    per_q_rounded = {
        qid: {m: round(v, 4) for m, v in body.items()}
        for qid, body in per_q.items()
    }
    return {"aggregate": aggregate, "per_symptom": per_q_rounded}


def build_consolidated() -> dict:
    qrels_loaded = {name: _load_qrels(p) for name, p in QRELS_FILES.items()}
    qrels_stats = {
        name: {
            "n_queries": len(qrels_loaded[name]),
            "n_total_judged": sum(len(q) for q in qrels_loaded[name].values()),
            "n_relevant": sum(sum(q.values()) for q in qrels_loaded[name].values()),
            "relevant_per_query": {qid: sum(qrels_loaded[name][qid].values()) for qid in sorted(qrels_loaded[name])},
        }
        for name in qrels_loaded
    }

    scored: dict[str, dict] = {}
    for run_name, run_path in RUN_FILES.items():
        if not run_path.exists():
            scored[run_name] = {"error": "missing trec file"}
            continue
        scored[run_name] = {
            "trec_file": str(run_path.relative_to(REPO_ROOT).as_posix()),
            "majority": score_run(run_path, qrels_loaded["majority"]),
            "unanimity": score_run(run_path, qrels_loaded["unanimity"]),
        }

    discrepancy = {"majority": {}, "unanimity": {}}
    for variant in ("majority", "unanimity"):
        pdf_table = OFFICIAL_PRELIMINARY_PDF[f"{variant}_table10" if variant == "majority" else f"{variant}_table11"]
        for run_name, pdf_row in pdf_table.items():
            local = scored.get(run_name, {}).get(variant, {}).get("aggregate")
            if not local:
                continue
            discrepancy[variant][run_name] = {
                m: {"local": local[m], "pdf": pdf_row[m], "delta": round(local[m] - pdf_row[m], 4)}
                for m in ("AP", "R-PREC", "P@10", "NDCG")
            }

    return {
        "metadata": {
            "task": "eRisk 2026 Task 3 — ADHD Symptom Sentence Ranking (1st edition)",
            "team": "INSA-Lyon",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_symptoms": 18,
            "n_runs_submitted": 5,
            "qrels_files": {k: str(v.relative_to(REPO_ROOT).as_posix()) for k, v in QRELS_FILES.items()},
            "qrels_statistics": qrels_stats,
            "asrs_symptoms": ASRS_SYMPTOMS,
            "run_methodology": RUN_METHODOLOGY,
            "metric_definitions": {
                "AP": "Average Precision per query; unjudged docs treated as non-relevant.",
                "R-PREC": "Precision at rank R, where R is the number of relevant docs for that query.",
                "P@10": "Precision at top 10 docs returned.",
                "NDCG": "Normalized DCG over the full ranking (trec_eval default; gain=rel, log2(rank+1) discount).",
                "MAP": "Mean AP across the 18 ASRS symptoms.",
            },
        },
        "official_preliminary_results": OFFICIAL_PRELIMINARY_PDF,
        "scored_locally_vs_qrels": scored,
        "discrepancy_vs_pdf": discrepancy,
    }


def main() -> None:
    consolidated = build_consolidated()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT).as_posix()}")
    print()
    for variant in ("majority", "unanimity"):
        print(f"=== {variant.upper()} qrels ===")
        for run_name in RUN_FILES:
            agg = consolidated["scored_locally_vs_qrels"].get(run_name, {}).get(variant, {}).get("aggregate")
            if agg:
                print(f"  {run_name:>16}: AP={agg['AP']:.3f}  R-PREC={agg['R-PREC']:.3f}  P@10={agg['P@10']:.3f}  NDCG={agg['NDCG']:.3f}")


if __name__ == "__main__":
    main()
