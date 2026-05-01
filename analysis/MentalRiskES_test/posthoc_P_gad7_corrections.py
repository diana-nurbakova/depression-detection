"""Post-Hoc Analysis P — Principled GAD-7 Corrections.

Pre-submission ablation predicted POSITIVE GAD-7 bias and proposed corrections P1-P5.
Tests them on the test set (Run 2) and reports the impact.

Note: corrections P1-P4 were designed to subtract from GAD-7 predictions (to reduce
over-prediction). On the actual test set, GAD-7 shows UNDER-prediction on the rounds
we have predictions for (rounds 1-30). We therefore also test "inverted" corrections
that ADD points, to see whether the trial-derived correction direction would have been
wrong on the test set.
"""
from __future__ import annotations

import copy

import pandas as pd

from utils import (
    classify_band,
    load_config,
    load_task1_gold,
    mae,
    repo_path,
    task1_last_round_predictions,
    total,
)


def gad7_total(scores: list[int]) -> int:
    return sum(scores)


def correction_P1(scores: list[int]) -> list[int]:
    """Item 2: if pred=3, set to 2 (uncontrollable worry)."""
    s = list(scores)
    if s[1] == 3:
        s[1] = 2
    return s


def correction_P2(scores: list[int], threshold: int = 12) -> list[int]:
    """Subtract 1 evenly when pred_total >= threshold (clipped at 0)."""
    s = list(scores)
    if gad7_total(s) >= threshold:
        # subtract 1 from highest-scoring item
        idx = max(range(len(s)), key=lambda i: s[i])
        s[idx] = max(0, s[idx] - 1)
    return s


def correction_P3(scores: list[int], threshold: int = 15) -> list[int]:
    """Subtract 2 (one each from top-2 items) when pred_total >= 15."""
    s = list(scores)
    if gad7_total(s) >= threshold:
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        for i in order[:2]:
            s[i] = max(0, s[i] - 1)
    return s


def correction_P4(scores: list[int]) -> list[int]:
    """Boundary-zone: if pred_total in [15,17], subtract 2 from top items."""
    s = list(scores)
    if 15 <= gad7_total(s) <= 17:
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        for i in order[:2]:
            s[i] = max(0, s[i] - 1)
    return s


def correction_P5(scores: list[int]) -> list[int]:
    """Cap items at 2 (no item allowed at 3 unless majority of items are >=2)."""
    s = list(scores)
    high_count = sum(1 for v in s if v >= 2)
    if high_count < 4:  # only allow 3s when most items already >=2
        s = [min(2, v) for v in s]
    return s


# Inverted corrections (since test shows under-prediction)
def correction_INV_total(scores: list[int], threshold: int = 5) -> list[int]:
    """ADD 1 to lowest-scoring item when pred_total < threshold (under-prediction fix)."""
    s = list(scores)
    if gad7_total(s) < threshold:
        idx = min(range(len(s)), key=lambda i: s[i])
        s[idx] = min(3, s[idx] + 1)
    return s


CORRECTIONS = {
    "baseline": lambda s: list(s),
    "P1": correction_P1,
    "P2": correction_P2,
    "P3": correction_P3,
    "P4": correction_P4,
    "P5": correction_P5,
    "P1+P2": lambda s: correction_P2(correction_P1(s)),
    "P1+P3": lambda s: correction_P3(correction_P1(s)),
    "P1+P4": lambda s: correction_P4(correction_P1(s)),
    # Test-set-derived inverted correction: shift UP when low (under-prediction)
    "INV_low": correction_INV_total,
}


def run() -> None:
    cfg = load_config()
    gold = load_task1_gold(cfg)
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        preds = task1_last_round_predictions(cfg, run_idx)
        common = sorted(set(gold.keys()) & set(preds.keys()))

        for corr_name, corr_fn in CORRECTIONS.items():
            mae_items_list = []
            mae_total_list = []
            band_correct_list = []
            for sid in common:
                g = gold[sid]["GAD-7"]
                p_orig = preds[sid]["GAD-7"]
                p_corr = corr_fn(p_orig)
                mae_items_list.append(mae(p_corr, g))
                mae_total_list.append(abs(total(p_corr) - total(g)))
                gband = classify_band(total(g), "GAD-7", cfg)
                pband = classify_band(total(p_corr), "GAD-7", cfg)
                band_correct_list.append(int(gband == pband))

            rows.append({
                "run": run_idx,
                "correction": corr_name,
                "n": len(common),
                "GAD7_MAE_items": sum(mae_items_list) / len(mae_items_list),
                "GAD7_MAE_total": sum(mae_total_list) / len(mae_total_list),
                "GAD7_band_acc": sum(band_correct_list) / len(band_correct_list),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "P_gad7_corrections.csv", index=False)

    print("=== Post-Hoc Analysis P: GAD-7 Corrections ===\n")
    for run_idx in sorted(df["run"].unique()):
        sub = df[df["run"] == run_idx].sort_values("GAD7_MAE_items").reset_index(drop=True)
        print(f"--- Run {run_idx} (sorted by MAE_items ascending) ---")
        print(sub[["correction", "GAD7_MAE_items", "GAD7_MAE_total", "GAD7_band_acc"]].to_string(index=False))
        print()


if __name__ == "__main__":
    run()
