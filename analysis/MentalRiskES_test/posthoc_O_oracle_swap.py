"""Post-Hoc Analysis O — Aggregate-Level Oracle Component Swap.

Per-patient predictions for other teams aren't released, so we estimate the
hypothetical hybrid scores at the aggregate level: take INSALyon's PHQ-9 and
CompACT-10 metrics and substitute the Gemma baseline's GAD-7.
"""
from __future__ import annotations

import pandas as pd

from utils import load_config, repo_path


def run() -> None:
    cfg = load_config()
    out_dir = repo_path(cfg["paths"]["output_dir"])
    xlsx = repo_path(cfg["paths"]["leaderboard_xlsx"])
    df = pd.read_excel(xlsx, sheet_name="Task1")

    insalyon = df[df["Team"] == cfg["team"]["name"]].copy().sort_values("Rank")
    gemma = df[df["Team"].str.contains("Gemma", na=False)].iloc[0]
    fbkillers = df[df["Team"] == "FBKillers"].sort_values("MAE_GAD7").iloc[0]
    verbanex = df[df["Team"] == "VerbaNex AI"].sort_values("MAE_GAD7").iloc[0]

    print("=== Post-Hoc Analysis O: Oracle Component Swap (Aggregate-Level) ===\n")

    # Build hybrid for each INSALyon run
    rows = []
    for _, our_row in insalyon.iterrows():
        for donor_name, donor in (("Gemma", gemma), ("FBKillers_best", fbkillers), ("VerbaNex_best", verbanex)):
            hybrid = (donor["MAE_GAD7"] + our_row["MAE_PHQ9"] + our_row["MAE_CompACT10"]) / 3
            macro_hybrid = (donor["Macro_MAE_GAD7"] + our_row["Macro_MAE_PHQ9"] + our_row["Macro_MAE_CompACT10"]) / 3

            # Project rank: how many teams have lower MAE_Combined?
            n_below = (df["MAE_Combined"] < hybrid).sum()
            projected_rank = n_below + 1

            rows.append({
                "our_run": int(our_row["Run"]),
                "our_rank": int(our_row["Rank"]),
                "our_MAE_combined": our_row["MAE_Combined"],
                "donor": donor_name,
                "donor_GAD7": donor["MAE_GAD7"],
                "hybrid_MAE_combined": hybrid,
                "hybrid_macro_MAE": macro_hybrid,
                "delta_combined": hybrid - our_row["MAE_Combined"],
                "projected_rank_in_official_table": int(projected_rank),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "O_oracle_swap.csv", index=False)
    print(out.to_string(index=False))

    # Reverse swap: what if we kept GAD-7 but borrowed others' PHQ-9 and CompACT-10?
    print("\n--- Reverse swap: how much does our GAD-7 hurt others? (just a sanity check) ---")
    for _, our_row in insalyon.iterrows():
        # FBKillers gets our GAD-7 instead
        hybrid = (our_row["MAE_GAD7"] + fbkillers["MAE_PHQ9"] + fbkillers["MAE_CompACT10"]) / 3
        print(f"  Run {int(our_row['Run'])}: FBKillers + INSALyon GAD-7 = {hybrid:.4f} (FBKillers original combined: {fbkillers['MAE_Combined']:.4f})")


if __name__ == "__main__":
    run()
