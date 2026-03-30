"""Evaluation utilities for MentalRiskES Task 1.

Uses shared metrics from mentalriskes.metrics. Adds task1-specific
convenience functions for trial evaluation and trajectory analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..metrics import (
    evaluate_task1_full,
    format_task1_report,
    mae,
    pearson_r,
    rmse,
)

logger = logging.getLogger(__name__)

# Manual annotations from spec (Appendix C) — trial patient at round 19
TRIAL_GOLD = {
    "PHQ-9": [1, 2, 1, 2, 1, 2, 2, 2, 0],
    "GAD-7": [3, 2, 2, 2, 2, 1, 2],
    "CompACT-10": [3, 3, 4, 3, 3, 3, 4, 3, 3, 4],
}


def evaluate_prediction(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
    gold: dict[str, list[int]] | None = None,
) -> dict:
    """Evaluate a single prediction using the full metric suite."""
    gold = gold or TRIAL_GOLD
    predicted = {"PHQ-9": phq9, "GAD-7": gad7, "CompACT-10": compact10}
    return evaluate_task1_full(predicted, gold)


def print_evaluation_report(results: dict) -> str:
    """Format evaluation results as a readable report."""
    return format_task1_report(results)


def evaluate_trial_run(predictions_log_path: str | Path) -> dict:
    """
    Evaluate all predictions from a trial run log file (JSONL).

    Finds the last prediction for each session and evaluates against gold.
    """
    path = Path(predictions_log_path)
    if not path.exists():
        logger.warning("Predictions log not found: %s", path)
        return {}

    # Find last prediction per session
    last_preds: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sid = entry["session_id"]
            last_preds[sid] = entry

    results = {}
    for sid, entry in last_preds.items():
        results[sid] = evaluate_prediction(
            entry["phq9"], entry["gad7"], entry["compact10"],
        )

    return results


def evaluate_trajectory(
    predictions_log_path: str | Path,
    gold: dict[str, list[int]] | None = None,
) -> dict[str, list[dict]]:
    """
    Evaluate how predictions evolve across rounds (convergence analysis).

    Returns:
        {session_id: [{round, rmse_phq9, rmse_gad7, rmse_compact10, ...}]}
    """
    gold = gold or TRIAL_GOLD
    path = Path(predictions_log_path)
    if not path.exists():
        return {}

    trajectories: dict[str, list[dict]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sid = entry["session_id"]
            r = entry["round"]

            round_metrics = {"round": r}
            for instrument, key in [("PHQ-9", "phq9"), ("GAD-7", "gad7"), ("CompACT-10", "compact10")]:
                g = gold.get(instrument)
                p = entry.get(key, [])
                if g and p:
                    round_metrics[f"rmse_{instrument}"] = rmse(p, g)
                    round_metrics[f"mae_{instrument}"] = mae(p, g)
                    round_metrics[f"total_{instrument}"] = sum(p)

            trajectories.setdefault(sid, []).append(round_metrics)

    return trajectories
