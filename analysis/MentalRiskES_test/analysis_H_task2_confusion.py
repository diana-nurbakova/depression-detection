"""Analysis H — Task 2 response selection error taxonomy.

3x3 confusion matrix gold x predicted, per-class precision/recall/F1, accuracy by round position.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

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
    rows = []
    for run_meta in cfg["team"]["runs"]:
        run_idx = run_meta["idx"]
        preds = load_task2_predictions(cfg, run_idx)
        for rnd, sess_gold in gold.items():
            if rnd not in preds:
                continue  # we only have round 1-30 predictions
            for sid, opt in sess_gold.items():
                if sid not in preds[rnd]:
                    continue
                gint = opt_to_int(opt)
                pint = preds[rnd][sid]
                rows.append({
                    "run": run_idx,
                    "round": rnd,
                    "session": sid,
                    "gold": gint,
                    "pred": pint,
                    "correct": int(gint == pint),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "H_task2_predictions_long.csv", index=False)

    print("=== Analysis H: Task 2 Confusion / Errors ===")
    print(f"Predictions analyzed: {len(df)} per-round samples (10 sessions x 30 rounds)\n")

    for run_idx in sorted(df["run"].unique()):
        sub = df[df["run"] == run_idx]
        print(f"--- Run {run_idx} ---")
        print(f"Accuracy (rounds 1-30 only): {sub['correct'].mean():.4f}  (n={len(sub)})")
        cm = confusion_matrix(sub["gold"], sub["pred"], labels=[1, 2, 3])
        cm_df = pd.DataFrame(cm, index=[f"gold_{i}" for i in [1, 2, 3]], columns=[f"pred_{i}" for i in [1, 2, 3]])
        print("Confusion matrix:")
        print(cm_df.to_string())
        print(classification_report(sub["gold"], sub["pred"], labels=[1, 2, 3], zero_division=0))
        cm_df.to_csv(out_dir / f"H_run{run_idx}_confusion.csv")
        print()

    # Round position analysis (Run 2)
    print("--- Run 2 accuracy by round tercile ---")
    r2 = df[df["run"] == 2].copy()
    max_round = r2["round"].max()
    def tercile(r):
        if r <= max_round / 3: return "early"
        if r <= 2 * max_round / 3: return "mid"
        return "late"
    r2["tercile"] = r2["round"].apply(tercile)
    pos_acc = r2.groupby("tercile").agg(n=("correct", "size"), acc=("correct", "mean")).reindex(["early", "mid", "late"])
    print(pos_acc.to_string())
    pos_acc.to_csv(out_dir / "H_run2_round_position.csv")

    # Dominant error type (Run 2)
    print("\n--- Run 2 dominant error pattern ---")
    err = r2[r2["correct"] == 0]
    err_counts = err.groupby(["gold", "pred"]).size().reset_index(name="count")
    err_counts["pct_of_errors"] = err_counts["count"] / err_counts["count"].sum()
    err_counts = err_counts.sort_values("count", ascending=False)
    print(err_counts.to_string(index=False))


if __name__ == "__main__":
    run()
