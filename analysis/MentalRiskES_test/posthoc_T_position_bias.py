"""Post-Hoc Analysis T_bias — Task 2 Position and Length Bias.

Tests whether our system over-picks specific candidate positions and whether the
chosen response differs in length/complexity from the gold response.
"""
from __future__ import annotations

import statistics
from collections import Counter

import pandas as pd
from scipy.stats import chisquare

from utils import (
    load_config,
    load_task2_gold,
    load_task2_predictions,
    load_task2_test,
    repo_path,
)


def opt_to_int(opt: str) -> int:
    return int(opt.replace("option_", ""))


def run() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_task2_gold(cfg)
    test_data = load_task2_test(cfg)

    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        preds = load_task2_predictions(cfg, run_idx)
        for rnd in sorted(gold.keys()):
            if rnd not in preds or rnd not in test_data:
                continue
            for sid, opt in gold[rnd].items():
                if sid not in preds[rnd] or sid not in test_data[rnd]:
                    continue
                gint = opt_to_int(opt)
                pint = preds[rnd][sid]
                d = test_data[rnd][sid]
                lengths = [len(d.get(f"option_{k}", "").split()) for k in (1, 2, 3)]
                rows.append({
                    "run": run_idx,
                    "round": rnd,
                    "session": sid,
                    "gold": gint,
                    "pred": pint,
                    "correct": int(gint == pint),
                    "len_opt1": lengths[0],
                    "len_opt2": lengths[1],
                    "len_opt3": lengths[2],
                    "gold_len": lengths[gint - 1],
                    "pred_len": lengths[pint - 1],
                    "len_diff_pred_minus_gold": lengths[pint - 1] - lengths[gint - 1],
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "T_bias_long.csv", index=False)

    print("=== Post-Hoc Analysis T_bias: Position / Length Bias ===\n")

    # Position distribution per run + chi2 vs uniform & vs gold
    for run_idx in sorted(df["run"].unique()):
        sub = df[df["run"] == run_idx]
        pred_counts = [sub[sub["pred"] == k].shape[0] for k in (1, 2, 3)]
        gold_counts = [sub[sub["gold"] == k].shape[0] for k in (1, 2, 3)]
        n = len(sub)
        chi2_uniform, p_uniform = chisquare(pred_counts, [n / 3] * 3)
        chi2_gold, p_gold = chisquare(pred_counts, gold_counts)
        print(f"--- Run {run_idx} (n={n}) ---")
        print(f"  pred dist: opt1={pred_counts[0]}, opt2={pred_counts[1]}, opt3={pred_counts[2]}")
        print(f"  gold dist: opt1={gold_counts[0]}, opt2={gold_counts[1]}, opt3={gold_counts[2]}")
        print(f"  chi2 vs uniform: {chi2_uniform:.2f}  p={p_uniform:.4g}")
        print(f"  chi2 vs gold:    {chi2_gold:.2f}  p={p_gold:.4g}")

        # Length analysis when wrong
        wrong = sub[sub["correct"] == 0]
        if len(wrong):
            avg_diff = wrong["len_diff_pred_minus_gold"].mean()
            avg_pct_longer = (wrong["len_diff_pred_minus_gold"] > 0).mean()
            print(f"  When wrong: pred is {avg_diff:+.1f} words {'longer' if avg_diff > 0 else 'shorter'} than gold ({avg_pct_longer:.1%} of errors are longer)")
        print()

    # Length comparison overall
    print("--- Mean option length by position ---")
    print(f"  opt1: {df['len_opt1'].mean():.1f} words")
    print(f"  opt2: {df['len_opt2'].mean():.1f} words")
    print(f"  opt3: {df['len_opt3'].mean():.1f} words")

    # Conditional analysis (Run 2): when the system picked opt2, how often was opt2 the longest?
    print("\n--- Run 2: when we picked option K, was K the longest? ---")
    r2 = df[df["run"] == 2].copy()
    for k in (1, 2, 3):
        sub = r2[r2["pred"] == k]
        if len(sub) == 0:
            continue
        longest_count = 0
        for _, row in sub.iterrows():
            lengths = [row["len_opt1"], row["len_opt2"], row["len_opt3"]]
            if lengths[k - 1] == max(lengths):
                longest_count += 1
        print(f"  pred={k}: chosen {len(sub)} times, was longest in {longest_count} ({longest_count / len(sub):.1%})")


if __name__ == "__main__":
    run()
