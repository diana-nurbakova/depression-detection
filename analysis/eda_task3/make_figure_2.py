#!/usr/bin/env python3
"""
Regenerate Figure 2 of the CADRE eRisk 2026 Task 3 working notes:
relevant docs per symptom under majority and unanimity qrels,
grouped into three panels by ASRS bifactor structure (Stanton et al. 2018).

Bifactor structure:
  - Inattention:        items 1, 2, 3, 4, 7, 8, 9, 10, 11
  - Motor H/I:          items 5, 6, 12, 13, 14
  - Verbal H/I:         items 15, 16, 17, 18

Input:  /mnt/user-data/uploads/eda_task3.json
Output: /mnt/user-data/outputs/erisk-t3-paper/figures/qrels_per_symptom_bifactor.png
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INPUT  = "/mnt/user-data/uploads/eda_task3.json"
OUTPUT = "/mnt/user-data/outputs/erisk-t3-paper/figures/qrels_per_symptom_bifactor.png"

BIFACTOR = {
    "Inattention":        [1, 2, 3, 4, 7, 8, 9, 10, 11],
    "Motor H/I":          [5, 6, 12, 13, 14],
    "Verbal H/I":         [15, 16, 17, 18],
}

# Short labels (matching the labels on the original Figure 2)
SHORT_LABELS = {
    1: "wrap-up details",   2: "getting in order",  3: "remember appts",
    4: "avoid thought tasks", 5: "fidget hands/feet", 6: "overly active",
    7: "careless mistakes", 8: "keep attention",    9: "concentrate on speech",
    10: "misplacing things", 11: "distracted noise", 12: "leaving seat",
    13: "restless/fidgety", 14: "unwinding",        15: "talking too much",
    16: "finish sentences", 17: "waiting turn",     18: "interrupting",
}


def main():
    with open(INPUT) as f:
        eda = json.load(f)

    maj = eda["qrels"]["majority"]["per_symptom"]
    una = eda["qrels"]["unanimity"]["per_symptom"]

    # Compute relative panel widths proportional to item count so bars are uniform
    panel_widths = [len(items) for items in BIFACTOR.values()]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4.8),
        gridspec_kw={"width_ratios": panel_widths, "wspace": 0.12},
        sharey=True,
    )

    # Compute global y-max so all panels share the same scale
    y_max = max(max(int(maj[str(s)]["relevant"]) for s in range(1, 19)),
                max(int(una[str(s)]["relevant"]) for s in range(1, 19))) + 5

    bar_width = 0.38

    # Style: solid mid-grey for majority, hatched white-with-dark-border for unanimity.
    # B&W safe; no colour-coding needed.
    majority_style = dict(color="#6c8ebf", edgecolor="black", linewidth=0.6,
                          label="majority (\u22652/3)")
    unanimity_style = dict(facecolor="#c46966", edgecolor="black", linewidth=0.6,
                           label="unanimity (3/3)")

    for ax, (factor_name, items) in zip(axes, BIFACTOR.items()):
        x = list(range(len(items)))
        maj_vals = [int(maj[str(i)]["relevant"]) for i in items]
        una_vals = [int(una[str(i)]["relevant"]) for i in items]

        ax.bar([xi - bar_width / 2 for xi in x], maj_vals,
               width=bar_width, **majority_style)
        ax.bar([xi + bar_width / 2 for xi in x], una_vals,
               width=bar_width, **unanimity_style)

        ax.set_xticks(x)
        # Two-line tick labels: item number on top, short label below
        labels = [f"{i}\n{SHORT_LABELS[i]}" for i in items]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        ax.set_title(factor_name, fontsize=12, fontweight="bold", pad=6)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("# relevant docs", fontsize=11)

    # Single legend below the x-axis labels (with extra figure padding)
    legend_handles = [
        Patch(facecolor="#6c8ebf", edgecolor="black", label="majority (\u22652/3)"),
        Patch(facecolor="#c46966", edgecolor="black", label="unanimity (3/3)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=10, ncol=2)

    plt.subplots_adjust(bottom=0.22)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
