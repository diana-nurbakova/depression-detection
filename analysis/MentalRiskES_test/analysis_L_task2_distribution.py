"""Analysis L — Task 2 gold/predicted label distribution and KL divergence.

Tests whether random / majority baselines outperform our system, identifies position bias.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from utils import (
    load_config,
    load_task2_gold,
    load_task2_predictions,
    repo_path,
)


def opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def run() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_task2_gold(cfg)
    # Build full gold label list (all 82 rounds where available)
    gold_labels_full = []
    for rnd, sess_gold in gold.items():
        for sid, opt in sess_gold.items():
            gold_labels_full.append(opt_to_int(opt))

    print("=== Analysis L: Task 2 Distribution / Baselines ===\n")
    full_dist = Counter(gold_labels_full)
    n_full = len(gold_labels_full)
    print(f"Full gold (rounds 1-82): n={n_full}")
    for k in (1, 2, 3):
        print(f"  P(gold={k}) = {full_dist.get(k, 0) / n_full:.4f}  ({full_dist.get(k, 0)})")

    # Compare to subset (rounds 1-30 where we have predictions)
    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        preds = load_task2_predictions(cfg, run_idx)
        gold_subset = []
        pred_subset = []
        for rnd, sess_gold in gold.items():
            if rnd not in preds:
                continue
            for sid, opt in sess_gold.items():
                if sid not in preds[rnd]:
                    continue
                gold_subset.append(opt_to_int(opt))
                pred_subset.append(preds[rnd][sid])

        gold_dist = {k: gold_subset.count(k) / len(gold_subset) for k in (1, 2, 3)}
        pred_dist = {k: pred_subset.count(k) / len(pred_subset) for k in (1, 2, 3)}
        kl = sum(pred_dist[k] * np.log(pred_dist[k] / gold_dist[k]) for k in (1, 2, 3) if pred_dist[k] > 0 and gold_dist[k] > 0)

        # Chi-squared: pred vs gold (treating gold as expected)
        observed = [pred_subset.count(k) for k in (1, 2, 3)]
        expected = [gold_dist[k] * len(pred_subset) for k in (1, 2, 3)]
        chi2, p_val = chisquare(observed, expected)

        rows.append({
            "run": run_idx,
            "n": len(pred_subset),
            "gold_p1": gold_dist[1], "gold_p2": gold_dist[2], "gold_p3": gold_dist[3],
            "pred_p1": pred_dist[1], "pred_p2": pred_dist[2], "pred_p3": pred_dist[3],
            "KL_pred_to_gold": kl,
            "chi2": chi2, "chi2_pval": p_val,
            "majority_class_baseline": max(gold_dist.values()),
            "actual_acc": sum(1 for g, p in zip(gold_subset, pred_subset) if g == p) / len(gold_subset),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "L_task2_distribution.csv", index=False)
    print("\nPer-run distribution comparison:")
    print(df.to_string(index=False))

    # Position bias: do we systematically over-pick option 2 or 3?
    print("\n--- Position bias check ---")
    print("Random baseline expects 1/3 each (0.333). Gold (rounds 1-30) deviation:")
    sub_gold = []
    for rnd, sess_gold in gold.items():
        if rnd > 30:
            continue
        for sid, opt in sess_gold.items():
            sub_gold.append(opt_to_int(opt))
    sub_dist = Counter(sub_gold)
    n_sub = len(sub_gold)
    for k in (1, 2, 3):
        print(f"  Gold P(={k}) = {sub_dist.get(k, 0) / n_sub:.4f}  ({sub_dist.get(k, 0)} / {n_sub})")


if __name__ == "__main__":
    run()
