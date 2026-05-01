"""Analysis A — Per-patient error decomposition.

For each patient and run, compute per-instrument MAE, signed bias, and band correctness.
Stratifies by gold severity band, identifies hardest patients.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import (
    classify_band,
    load_config,
    load_task1_gold,
    mae,
    repo_path,
    signed_bias,
    task1_last_round_per_session,
    task1_last_round_predictions,
    total,
)


def run() -> pd.DataFrame:
    cfg = load_config()
    gold = load_task1_gold(cfg)
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        run_label = run_meta["label"]
        preds = task1_last_round_predictions(cfg, run_idx)
        last_rounds = task1_last_round_per_session(cfg, run_idx)

        common = sorted(set(gold.keys()) & set(preds.keys()))
        for sid in common:
            for instr in ("PHQ-9", "GAD-7", "CompACT-10"):
                g = gold[sid][instr]
                p = preds[sid][instr]
                gtot = total(g)
                ptot = total(p)
                gband = classify_band(gtot, instr, cfg)
                pband = classify_band(ptot, instr, cfg)
                rows.append(
                    {
                        "run": run_idx,
                        "run_label": run_label,
                        "session": sid,
                        "instrument": instr,
                        "gold_total": gtot,
                        "pred_total": ptot,
                        "total_bias": ptot - gtot,
                        "MAE_items": mae(p, g),
                        "signed_bias_per_item": signed_bias(p, g),
                        "gold_band": gband,
                        "pred_band": pband,
                        "band_correct": int(gband == pband),
                        "last_pred_round": last_rounds.get(sid, 0),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "A_per_patient_errors.csv", index=False)

    # Aggregate per-run summary
    agg = (
        df.groupby(["run", "instrument"])
        .agg(
            mae_items=("MAE_items", "mean"),
            mean_total_bias=("total_bias", "mean"),
            mae_total=("total_bias", lambda x: x.abs().mean()),
            band_acc=("band_correct", "mean"),
            n_sessions=("session", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "A_per_run_summary.csv", index=False)

    # Stratified by gold severity band (Run 2 only — primary submission)
    primary_run = 2
    sub = df[df["run"] == primary_run]
    strat = (
        sub.groupby(["instrument", "gold_band"])
        .agg(
            mae_items=("MAE_items", "mean"),
            mean_total_bias=("total_bias", "mean"),
            band_acc=("band_correct", "mean"),
            n=("session", "nunique"),
        )
        .reset_index()
    )
    strat.to_csv(out_dir / "A_run2_band_stratified.csv", index=False)

    print("=== Analysis A: Per-Patient Error Decomposition ===")
    print(f"Sessions analyzed: {df['session'].nunique()} (gold has 17, test has 10)\n")
    print("Per-run aggregate:")
    print(agg.to_string(index=False))
    print("\nRun 2 — stratified by gold severity band:")
    print(strat.to_string(index=False))

    # Hardest patients (Run 2)
    hard = sub.groupby("session")["MAE_items"].mean().sort_values(ascending=False)
    print("\nHardest sessions (Run 2, mean MAE_items across instruments):")
    print(hard.to_string())

    return df


if __name__ == "__main__":
    run()
