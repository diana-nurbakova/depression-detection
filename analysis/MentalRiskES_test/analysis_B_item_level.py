"""Analysis B — Item-level error profiles.

For each item of each instrument, computes mean signed error, MAE, and exact-match rate.
Tests trial-ablation hypotheses (GAD-7 item 2 over-prediction, CompACT-10 VA bias).
"""
from __future__ import annotations

import pandas as pd

from utils import (
    load_config,
    load_task1_gold,
    repo_path,
    task1_last_round_predictions,
)


def run() -> pd.DataFrame:
    cfg = load_config()
    gold = load_task1_gold(cfg)
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        preds = task1_last_round_predictions(cfg, run_idx)
        common = sorted(set(gold.keys()) & set(preds.keys()))
        for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
            n_items = cfg["instruments"][instr]["n_items"]
            for i in range(n_items):
                diffs = []
                preds_i = []
                gold_i = []
                for sid in common:
                    g = gold[sid][instr][i]
                    p = preds[sid][instr][i]
                    diffs.append(p - g)
                    preds_i.append(p)
                    gold_i.append(g)
                if not diffs:
                    continue
                mean_signed = sum(diffs) / len(diffs)
                mae_i = sum(abs(d) for d in diffs) / len(diffs)
                exact = sum(1 for d in diffs if d == 0) / len(diffs)
                rows.append(
                    {
                        "run": run_idx,
                        "instrument": instr,
                        "item_idx_0based": i,
                        "item_label": cfg["instruments"][instr].get("item_labels", [str(i)] * n_items)[i] if "item_labels" in cfg["instruments"][instr] else f"item_{i+1}",
                        "n": len(diffs),
                        "mean_signed_error": mean_signed,
                        "MAE": mae_i,
                        "exact_match_rate": exact,
                        "mean_gold": sum(gold_i) / len(gold_i),
                        "mean_pred": sum(preds_i) / len(preds_i),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "B_item_level_errors.csv", index=False)

    print("=== Analysis B: Item-Level Error Profiles ===\n")
    for run_idx in sorted(df["run"].unique()):
        sub = df[df["run"] == run_idx]
        print(f"--- Run {run_idx} ---")
        print(sub[["instrument", "item_label", "mean_signed_error", "MAE", "exact_match_rate", "mean_gold", "mean_pred"]].to_string(index=False))
        print()

    # CompACT-10 subscale decomposition (Run 2)
    print("\n--- Run 2 CompACT-10 subscale decomposition ---")
    sub_compact = df[(df["run"] == 2) & (df["instrument"] == "CompACT-10")].copy()
    subscales = cfg["instruments"]["CompACT-10"]["subscales"]
    rows_sub = []
    for name, items in subscales.items():
        s = sub_compact[sub_compact["item_idx_0based"].isin(items)]
        rows_sub.append({
            "subscale": name,
            "items_1based": [i + 1 for i in items],
            "mean_signed_error": s["mean_signed_error"].mean(),
            "MAE": s["MAE"].mean(),
            "mean_gold": s["mean_gold"].mean(),
            "mean_pred": s["mean_pred"].mean(),
        })
    sub_df = pd.DataFrame(rows_sub)
    sub_df.to_csv(out_dir / "B_compact10_subscale_run2.csv", index=False)
    print(sub_df.to_string(index=False))

    # Hypothesis tests (Run 2)
    print("\n--- Hypothesis tests (Run 2) ---")
    r2 = df[df["run"] == 2]
    gad7_item2 = r2[(r2["instrument"] == "GAD-7") & (r2["item_idx_0based"] == 1)].iloc[0]
    print(f"H1: GAD-7 item 2 (uncontrollable worry) signed bias = {gad7_item2['mean_signed_error']:+.3f}  (hypothesis: positive)")

    va_bias = sub_compact[sub_compact["item_idx_0based"].isin(subscales["VA"])]["mean_signed_error"].mean()
    print(f"H2: CompACT-10 VA mean signed bias = {va_bias:+.3f}  (hypothesis: ≥ +0.5)")

    ba_bias = sub_compact[sub_compact["item_idx_0based"].isin(subscales["BA"])]["mean_signed_error"].mean()
    print(f"H3: CompACT-10 BA mean signed bias = {ba_bias:+.3f}  (hypothesis: ≈ 0)")

    phq9_item9 = r2[(r2["instrument"] == "PHQ-9") & (r2["item_idx_0based"] == 8)].iloc[0]
    print(f"H4: PHQ-9 item 9 (suicidality) exact-match rate = {phq9_item9['exact_match_rate']:.3f}  (hypothesis: highest)")
    phq9_only = r2[r2["instrument"] == "PHQ-9"].reset_index(drop=True)
    phq9_top = phq9_only.iloc[phq9_only["exact_match_rate"].idxmax()]
    print(f"     PHQ-9 highest exact-match item: {phq9_top['item_label']} ({phq9_top['exact_match_rate']:.3f})")

    return df


if __name__ == "__main__":
    run()
