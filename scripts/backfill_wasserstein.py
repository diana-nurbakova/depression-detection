"""Retroactively compute Wasserstein metrics + 3-layer transport analysis for existing tom_on results.

Reads E_profiles/I_profiles from saved JSONs, recomputes all metrics
using the POT library, and updates the files in place.

Usage:
    uv run python scripts/backfill_wasserstein.py [--dir runs/tom_ablation/tom_on]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from erisk_task1.tom import (
    compute_transport_analysis,
    get_cost_matrix,
    wasserstein_balanced,
    wasserstein_transport_plan,
)


def backfill_file(fpath: Path, ground_truth: dict[str, list]) -> bool:
    """Recompute Wasserstein metrics for a single result file. Returns True if updated."""
    if fpath.stat().st_size == 0:
        print(f"  SKIP {fpath.name}: empty file (needs rerun)")
        return False
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    tom = data.get("tom_summary")
    if not tom:
        print(f"  SKIP {fpath.name}: no tom_summary")
        return False

    E_profiles = {int(k): np.array(v) for k, v in tom.get("E_profiles", {}).items()}
    I_profiles = {int(k): np.array(v) for k, v in tom.get("I_profiles", {}).items()}

    if not E_profiles:
        print(f"  SKIP {fpath.name}: no E_profiles")
        return False

    persona = data["persona"]
    gt_vec = ground_truth.get(persona)
    gt_array = np.array(gt_vec, dtype=np.float64) if gt_vec else None

    cost_mat = get_cost_matrix()

    W_self: dict[str, dict[str, float]] = {}
    W_align: dict[str, float] = {}
    W_accuracy: dict[str, float] = {}
    transport_plans: dict[str, list] = {}
    transport_analysis: dict[str, dict] = {}

    sorted_turns = sorted(E_profiles.keys())

    for t in sorted_turns:
        E_t = E_profiles[t]

        # W_self: profile shift over last k turns
        self_k = {}
        for k in (1, 2, 5):
            t_prev = t - k
            if t_prev in E_profiles:
                d = wasserstein_balanced(E_profiles[t_prev], E_t, cost_mat)
                if d is not None:
                    self_k[str(k)] = round(d, 4)
        if self_k:
            W_self[str(t)] = self_k

        # W_align: interviewer vs persona
        if t in I_profiles:
            d = wasserstein_balanced(I_profiles[t], E_t, cost_mat)
            if d is not None:
                W_align[str(t)] = round(d, 4)

        # W_accuracy + transport plan: assessment vs ground truth
        if gt_array is not None:
            result = wasserstein_transport_plan(E_t, gt_array, cost_mat)
            if result is not None:
                d, gamma = result
                W_accuracy[str(t)] = round(d, 4)
                transport_plans[str(t)] = np.round(gamma, 6).tolist()

            # 3-layer transport analysis
            analysis = compute_transport_analysis(E_t, gt_array, cost_mat)
            if any(v is not None for v in analysis.values()):
                transport_analysis[str(t)] = analysis

    # Update tom_summary
    tom["W_self"] = W_self
    tom["W_align"] = W_align
    tom["W_accuracy"] = W_accuracy
    tom["transport_plans"] = transport_plans
    tom["transport_analysis"] = transport_analysis
    tom["pot_available"] = True

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    n_turns = len(sorted_turns)
    n_analyzed = len(transport_analysis)
    print(f"  OK {fpath.name}: {n_turns} turns, "
          f"W_self={len(W_self)}, W_align={len(W_align)}, "
          f"W_accuracy={len(W_accuracy)}, 3-layer={n_analyzed}")
    return True


def main():
    tom_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/tom_ablation/tom_on")

    if not tom_dir.exists():
        print(f"Directory not found: {tom_dir}")
        sys.exit(1)

    # Load ground truth
    gt_path = Path("data/talkdep_conversations/ground_truth.json")
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth for {len(ground_truth)} personas")
    else:
        ground_truth = {}
        print("WARNING: No ground truth file found, W_accuracy will be empty")

    # Process all result files
    files = sorted(tom_dir.glob("tom_on_*.json"))
    print(f"Found {len(files)} result files in {tom_dir}\n")

    updated = 0
    for fpath in files:
        if backfill_file(fpath, ground_truth):
            updated += 1

    print(f"\nDone: {updated}/{len(files)} files updated")


if __name__ == "__main__":
    main()
