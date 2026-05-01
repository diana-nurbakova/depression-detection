"""Analysis Q — Per-instrument leaderboard ranking from official Excel.

Establishes INSALyon's strengths (CompACT-10, PHQ-9 Macro) and weaknesses (GAD-7).
"""
from __future__ import annotations

import pandas as pd

from utils import load_config, repo_path


def run() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx = repo_path(cfg["paths"]["leaderboard_xlsx"])
    task1 = pd.read_excel(xlsx, sheet_name="Task1")

    # Take each team's BEST run per metric
    metrics = {
        "MAE_GAD7": "GAD-7",
        "MAE_PHQ9": "PHQ-9",
        "MAE_CompACT10": "CompACT-10",
        "Macro_MAE_GAD7": "GAD-7 Macro",
        "Macro_MAE_PHQ9": "PHQ-9 Macro",
        "Macro_MAE_CompACT10": "CompACT-10 Macro",
        "MAE_Combined": "Combined",
    }
    out_rows = []
    for col, name in metrics.items():
        # best (lowest) per team
        best = task1.loc[task1.groupby("Team")[col].idxmin(), ["Team", "Run", col]].copy()
        best = best.sort_values(col).reset_index(drop=True)
        best["rank"] = best.index + 1
        for _, r in best.iterrows():
            out_rows.append({
                "metric": name,
                "team": r["Team"],
                "best_run": r["Run"],
                "value": r[col],
                "rank": r["rank"],
            })
    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "Q_team_metric_ranks.csv", index=False)

    print("=== Analysis Q: Per-Instrument Leaderboard Ranking ===\n")
    # Pivot for INSALyon and top contenders
    target_team = cfg["team"]["name"]
    print(f"--- {target_team} per-metric position ---")
    insalyon = df[df["team"] == target_team]
    print(insalyon[["metric", "best_run", "value", "rank"]].to_string(index=False))

    # Top 3 per metric
    print("\n--- Top 3 per metric ---")
    for name in metrics.values():
        top = df[df["metric"] == name].head(3)
        rank_str = ", ".join(f"{r['team']}/Run{int(r['best_run'])}={r['value']:.4f}" for _, r in top.iterrows())
        print(f"  {name}: {rank_str}")

    # Compute "balanced rank" per team
    print("\n--- Balanced rank (avg of per-instrument best ranks excluding Combined) ---")
    component_metrics = [name for name in metrics.values() if name != "Combined"]
    balanced = (
        df[df["metric"].isin(component_metrics)]
        .groupby("team")["rank"]
        .mean()
        .sort_values()
        .reset_index(name="avg_rank")
    )
    print(balanced.head(20).to_string(index=False))
    balanced.to_csv(out_dir / "Q_balanced_rank.csv", index=False)


if __name__ == "__main__":
    run()
