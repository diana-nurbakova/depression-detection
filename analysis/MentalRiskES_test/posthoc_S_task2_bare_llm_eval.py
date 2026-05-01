"""Evaluator for Task 2 bare-LLM experiments (S/S2/R2/S3/S4).

Walks `output/mentalriskes_task2_bare_llm/<model>__<mode>/raw.jsonl` and
produces:

  W_t2_bare_summary.csv         per-(model, mode) accuracy, F1, distribution
  W_t2_bare_per_run.csv         vs submitted/replay accuracy and projected rank
  W_t2_bare_tercile.csv         per-(model, mode) accuracy by round tercile
  W_t2_bare_confusion.csv       per-(model, mode) gold x pred confusion
  W_t2_bare_R2_inversion.csv    only for R2 mode: where does gold land in our
                                3-way ranking (rank 1 / 2 / 3 share)

Comparators:
  - submitted Run 2 acc 0.247
  - replay Run 2 acc 0.255 (full 82 rounds)
  - random baseline 0.363
  - top team 0.393

Runs in seconds; no LLM calls. Re-run any time more `raw.jsonl` rows land.

Usage:
  python analysis/MentalRiskES_test/posthoc_S_task2_bare_llm_eval.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import f1_score, confusion_matrix

from utils import load_config, load_task2_gold, repo_path


BARE_ROOT_REL = "output/mentalriskes_task2_bare_llm"

# Comparators (from leaderboard / SUMMARY)
SUBMITTED_ACC = {0: 0.210, 1: 0.237, 2: 0.247}
REPLAY_FULL_ACC = {0: None, 1: 0.220, 2: 0.255}  # Run 0 only finished today; rerun W to fill
T2_RANDOM = 0.363
T2_TOP = 0.393


def opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "raw.jsonl").exists())


def load_records(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bare_root = repo_path(BARE_ROOT_REL)
    run_dirs = discover_runs(bare_root)
    if not run_dirs:
        print(f"No bare-LLM runs found under {bare_root}")
        return

    gold = load_task2_gold(cfg)
    # Flatten gold into {(round, sid): int}
    gold_flat = {(rnd, sid): opt_to_int(opt) for rnd, sess in gold.items() for sid, opt in sess.items()}
    n_total_full = len(gold_flat)

    summary_rows = []
    tercile_rows = []
    confusion_rows = []
    inversion_rows = []
    per_run_rows = []

    # Tercile boundaries (over the 82-round span)
    max_round = max(rnd for rnd, _ in gold_flat)
    third = max_round / 3

    def tercile(rnd: int) -> str:
        if rnd <= third: return "early"
        if rnd <= 2 * third: return "mid"
        return "late"

    for rd in run_dirs:
        name = rd.name  # e.g. google_gemma-3-27b-it__S
        if "__" not in name:
            continue
        model_short, _, mode = name.rpartition("__")

        records = load_records(rd / "raw.jsonl")
        if not records:
            continue

        gold_seq, pred_seq, rnd_seq = [], [], []
        for rec in records:
            key = (int(rec["round"]), rec["session"])
            if key not in gold_flat:
                continue
            gold_seq.append(gold_flat[key])
            pred_seq.append(int(rec["prediction"]))
            rnd_seq.append(int(rec["round"]))

        if not gold_seq:
            continue

        n = len(gold_seq)
        correct = sum(g == p for g, p in zip(gold_seq, pred_seq))
        acc = correct / n
        macro_f1 = f1_score(gold_seq, pred_seq, average="macro", zero_division=0)
        per_class_f1 = f1_score(gold_seq, pred_seq, average=None, labels=[1, 2, 3], zero_division=0)

        pred_counts = Counter(pred_seq)
        gold_counts = Counter(gold_seq)
        # Chi-squared vs uniform and vs gold
        observed = [pred_counts.get(k, 0) for k in (1, 2, 3)]
        expected_uniform = [n / 3] * 3
        expected_gold = [gold_counts.get(k, 0) for k in (1, 2, 3)]
        chi2_uniform, p_uniform = chisquare(observed, expected_uniform)
        chi2_gold, p_gold = chisquare(observed, expected_gold)

        # Tercile breakdown
        tercile_correct = {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
        for g, p, r in zip(gold_seq, pred_seq, rnd_seq):
            t = tercile(r)
            tercile_correct[t][1] += 1
            if g == p:
                tercile_correct[t][0] += 1
        for t, (c, tot) in tercile_correct.items():
            tercile_rows.append({
                "model": model_short, "mode": mode,
                "tercile": t, "n": tot,
                "acc": c / tot if tot else None,
            })

        # Confusion matrix
        cm = confusion_matrix(gold_seq, pred_seq, labels=[1, 2, 3])
        for i, gv in enumerate([1, 2, 3]):
            for j, pv in enumerate([1, 2, 3]):
                confusion_rows.append({
                    "model": model_short, "mode": mode,
                    "gold": gv, "pred": pv, "count": int(cm[i, j]),
                })

        summary_rows.append({
            "model": model_short, "mode": mode,
            "n": n,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "f1_class_1": per_class_f1[0],
            "f1_class_2": per_class_f1[1],
            "f1_class_3": per_class_f1[2],
            "pred_p1": pred_counts.get(1, 0) / n,
            "pred_p2": pred_counts.get(2, 0) / n,
            "pred_p3": pred_counts.get(3, 0) / n,
            "chi2_uniform": chi2_uniform,
            "p_uniform": p_uniform,
            "chi2_gold": chi2_gold,
            "p_gold": p_gold,
        })

        # Comparator rows
        for ridx, ref_acc in SUBMITTED_ACC.items():
            per_run_rows.append({
                "model": model_short, "mode": mode,
                "compared_to": f"submitted_run{ridx}",
                "ref_acc": ref_acc,
                "bare_acc": acc,
                "delta_pp": (acc - ref_acc) * 100,
            })
        for ridx, ref_acc in REPLAY_FULL_ACC.items():
            if ref_acc is None:
                continue
            per_run_rows.append({
                "model": model_short, "mode": mode,
                "compared_to": f"replay_full_run{ridx}",
                "ref_acc": ref_acc,
                "bare_acc": acc,
                "delta_pp": (acc - ref_acc) * 100,
            })
        per_run_rows.append({
            "model": model_short, "mode": mode,
            "compared_to": "random_baseline",
            "ref_acc": T2_RANDOM, "bare_acc": acc,
            "delta_pp": (acc - T2_RANDOM) * 100,
        })
        per_run_rows.append({
            "model": model_short, "mode": mode,
            "compared_to": "top_team",
            "ref_acc": T2_TOP, "bare_acc": acc,
            "delta_pp": (acc - T2_TOP) * 100,
        })

        # R2 inversion analysis: where does the gold land in our 3-way ranking
        if mode == "R2":
            r1, r2, r3 = 0, 0, 0
            n_with_ranking = 0
            for rec in records:
                key = (int(rec["round"]), rec["session"])
                if key not in gold_flat:
                    continue
                ranking = rec.get("ranking")
                if not ranking:
                    continue
                gold_opt = gold_flat[key]
                pos = ranking.index(gold_opt) + 1
                n_with_ranking += 1
                if pos == 1: r1 += 1
                elif pos == 2: r2 += 1
                elif pos == 3: r3 += 1
            inversion_rows.append({
                "model": model_short, "mode": mode, "n": n_with_ranking,
                "gold_at_rank_1_pct": r1 / n_with_ranking if n_with_ranking else None,
                "gold_at_rank_2_pct": r2 / n_with_ranking if n_with_ranking else None,
                "gold_at_rank_3_pct": r3 / n_with_ranking if n_with_ranking else None,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "accuracy"], ascending=[True, False])
    summary_df.to_csv(out_dir / "W_t2_bare_summary.csv", index=False)

    if tercile_rows:
        tercile_df = pd.DataFrame(tercile_rows)
        tercile_df.to_csv(out_dir / "W_t2_bare_tercile.csv", index=False)

    if confusion_rows:
        confusion_df = pd.DataFrame(confusion_rows)
        confusion_df.to_csv(out_dir / "W_t2_bare_confusion.csv", index=False)

    if per_run_rows:
        per_run_df = pd.DataFrame(per_run_rows)
        per_run_df.to_csv(out_dir / "W_t2_bare_per_run.csv", index=False)

    if inversion_rows:
        inv_df = pd.DataFrame(inversion_rows)
        inv_df.to_csv(out_dir / "W_t2_bare_R2_inversion.csv", index=False)

    print("=" * 70)
    print("Task 2 Bare-LLM Evaluation")
    print("=" * 70)
    print("\nW_t2_bare_summary.csv:")
    print(summary_df.to_string(index=False))

    if tercile_rows:
        print("\nW_t2_bare_tercile.csv:")
        print(pd.DataFrame(tercile_rows).pivot(index=["model", "mode"], columns="tercile", values="acc"))

    if inversion_rows:
        print("\nW_t2_bare_R2_inversion.csv:")
        print(pd.DataFrame(inversion_rows).to_string(index=False))


if __name__ == "__main__":
    main()
