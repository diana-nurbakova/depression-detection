"""Analysis E — Run Comparison and Ablation Validation.

Compare Run 0 / 1 / 2 on Task 1 metrics (per-instrument MAE, total bias, band acc),
oracle ensemble headroom.
"""
from __future__ import annotations

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


def run() -> None:
    cfg = load_config()
    gold = load_task1_gold(cfg)
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all run predictions
    run_preds = {meta["idx"]: task1_last_round_predictions(cfg, meta["idx"]) for meta in cfg["team"]["runs"]}

    common = sorted(set(gold.keys()) & set.intersection(*[set(p.keys()) for p in run_preds.values()]))

    # Per-patient per-instrument MAE_items per run (table for win/loss/tie)
    rows = []
    for sid in common:
        for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
            g = gold[sid][instr]
            row = {"session": sid, "instrument": instr}
            for ridx, preds in run_preds.items():
                p = preds[sid][instr]
                row[f"run{ridx}_mae"] = mae(p, g)
                row[f"run{ridx}_total_bias"] = total(p) - total(g)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "E_per_patient_per_run.csv", index=False)

    print("=== Analysis E: Run Comparison ===\n")
    # Per-run aggregate
    for instr in ("PHQ-9", "GAD-7", "CompACT-10", "ALL"):
        print(f"\n--- {instr} ---")
        if instr == "ALL":
            sub = df
        else:
            sub = df[df["instrument"] == instr]
        for ridx in run_preds.keys():
            mae_mean = sub[f"run{ridx}_mae"].mean()
            bias_mean = sub[f"run{ridx}_total_bias"].mean()
            print(f"  Run {ridx}: MAE_items={mae_mean:.4f}, mean_total_bias={bias_mean:+.2f}")

    # Win counts (lowest MAE per row)
    df["best_run_mae"] = df[[f"run{r}_mae" for r in run_preds]].idxmin(axis=1)
    win_counts = df.groupby(["instrument", "best_run_mae"]).size().unstack(fill_value=0)
    print("\n--- Win counts (lowest MAE per session-instrument) ---")
    print(win_counts.to_string())

    # Oracle ensemble: per session-instrument, take min MAE across runs
    df["oracle_mae"] = df[[f"run{r}_mae" for r in run_preds]].min(axis=1)
    oracle_overall = df["oracle_mae"].mean()
    best_run_overall = min(df[f"run{r}_mae"].mean() for r in run_preds)
    print(f"\nBest single-run mean MAE_items: {best_run_overall:.4f}")
    print(f"Oracle ensemble mean MAE_items:  {oracle_overall:.4f}")
    print(f"Headroom (oracle vs best single): {best_run_overall - oracle_overall:+.4f}")

    # Per-instrument oracle
    print("\n--- Per-instrument oracle gap ---")
    for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
        sub = df[df["instrument"] == instr]
        best = min(sub[f"run{r}_mae"].mean() for r in run_preds)
        oracle = sub["oracle_mae"].mean()
        print(f"  {instr}: best={best:.4f}, oracle={oracle:.4f}, gap={best - oracle:+.4f}")


if __name__ == "__main__":
    run()
