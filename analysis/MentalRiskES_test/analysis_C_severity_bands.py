"""Analysis C — Severity Band Classification Accuracy.

PHQ-9 and GAD-7 band classification. Confusion matrices, directional misclassification,
boundary-zone vs mid-band split.
"""
from __future__ import annotations

import pandas as pd

from utils import (
    band_index,
    classify_band,
    load_config,
    load_task1_gold,
    repo_path,
    task1_last_round_predictions,
    total,
)


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
        for instr in ("PHQ-9", "GAD-7"):
            sem = cfg["instruments"][instr]["sem"]
            cutoffs = cfg["instruments"][instr]["band_cutoffs"]
            boundaries = [hi for _, hi, _ in cutoffs[:-1]]  # boundary totals
            for sid in common:
                gtot = total(gold[sid][instr])
                ptot = total(preds[sid][instr])
                gband = classify_band(gtot, instr, cfg)
                pband = classify_band(ptot, instr, cfg)
                gi = band_index(gband, instr, cfg)
                pi = band_index(pband, instr, cfg)
                # boundary zone: gold within 1 SEM of any boundary
                boundary_zone = any(abs(gtot - b) <= sem for b in boundaries)
                rows.append({
                    "run": run_idx,
                    "session": sid,
                    "instrument": instr,
                    "gold_total": gtot,
                    "pred_total": ptot,
                    "gold_band": gband,
                    "pred_band": pband,
                    "gold_band_idx": gi,
                    "pred_band_idx": pi,
                    "exact_match": int(gband == pband),
                    "adjacent_match": int(abs(gi - pi) <= 1),
                    "direction": "over" if pi > gi else ("under" if pi < gi else "match"),
                    "boundary_zone": boundary_zone,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "C_band_classification.csv", index=False)

    print("=== Analysis C: Severity Band Classification ===\n")
    for run_idx in sorted(df["run"].unique()):
        sub = df[df["run"] == run_idx]
        print(f"--- Run {run_idx} ---")
        agg = sub.groupby("instrument").agg(
            n=("session", "nunique"),
            band_acc=("exact_match", "mean"),
            adjacent_acc=("adjacent_match", "mean"),
            over_rate=("direction", lambda x: (x == "over").mean()),
            under_rate=("direction", lambda x: (x == "under").mean()),
        ).reset_index()
        print(agg.to_string(index=False))
        print()

    # Confusion matrices for Run 2 (primary)
    print("\n--- Run 2 confusion matrices ---")
    for instr in ("PHQ-9", "GAD-7"):
        sub = df[(df["run"] == 2) & (df["instrument"] == instr)]
        cm = pd.crosstab(sub["gold_band"], sub["pred_band"], dropna=False)
        # Order bands consistently
        bands = [name for _, _, name in cfg["instruments"][instr]["band_cutoffs"]]
        cm = cm.reindex(index=bands, columns=bands, fill_value=0)
        print(f"\n{instr}:")
        print(cm.to_string())
        cm.to_csv(out_dir / f"C_run2_{instr}_confusion.csv")

    # Boundary-zone vs mid-band (Run 2)
    print("\n--- Run 2 boundary-zone vs mid-band accuracy ---")
    r2 = df[df["run"] == 2]
    bz = r2.groupby(["instrument", "boundary_zone"]).agg(
        n=("session", "nunique"),
        band_acc=("exact_match", "mean"),
    ).reset_index()
    print(bz.to_string(index=False))
    bz.to_csv(out_dir / "C_run2_boundary_zone.csv", index=False)

    # H: >50% of GAD-7 band misclassifications occur at moderate->severe boundary
    print("\n--- Hypothesis: GAD-7 misclassifications concentrated at moderate->severe ---")
    gad_misclass = r2[(r2["instrument"] == "GAD-7") & (r2["exact_match"] == 0)]
    if len(gad_misclass) == 0:
        print("No GAD-7 misclassifications in Run 2.")
    else:
        mod_to_severe = gad_misclass[(gad_misclass["gold_band"] == "moderate") & (gad_misclass["pred_band"] == "severe")]
        pct = len(mod_to_severe) / len(gad_misclass)
        print(f"GAD-7 total misclassifications: {len(gad_misclass)}")
        print(f"  moderate→severe: {len(mod_to_severe)} ({pct:.1%})")


if __name__ == "__main__":
    run()
