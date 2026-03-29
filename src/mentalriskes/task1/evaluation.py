"""Evaluation utilities for MentalRiskES Task 1.

Compares predictions against manual annotations (trial data) or gold labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Manual annotations from spec (Appendix C) — trial patient at round 19
TRIAL_GOLD = {
    "PHQ-9": [1, 2, 1, 2, 1, 2, 2, 2, 0],
    "GAD-7": [3, 2, 2, 2, 2, 1, 2],
    "CompACT-10": [3, 3, 4, 3, 3, 3, 4, 3, 3, 4],
}


def item_mae(predicted: list[int], gold: list[int]) -> float:
    """Mean Absolute Error per item."""
    return float(np.mean(np.abs(np.array(predicted) - np.array(gold))))


def per_item_errors(predicted: list[int], gold: list[int]) -> list[int]:
    """Absolute error per item."""
    return [abs(p - g) for p, g in zip(predicted, gold)]


def total_score_error(predicted: list[int], gold: list[int]) -> int:
    """Absolute error of total scores."""
    return abs(sum(predicted) - sum(gold))


def severity_band_phq9(total: int) -> str:
    if total <= 4:
        return "minimal"
    elif total <= 9:
        return "mild"
    elif total <= 14:
        return "moderate"
    elif total <= 19:
        return "moderately_severe"
    return "severe"


def severity_band_gad7(total: int) -> str:
    if total <= 4:
        return "minimal"
    elif total <= 9:
        return "mild"
    elif total <= 14:
        return "moderate"
    return "severe"


def evaluate_prediction(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
    gold: dict[str, list[int]] | None = None,
) -> dict:
    """
    Evaluate a prediction against gold standard.

    If gold is None, uses TRIAL_GOLD (manual annotations for trial patient at round 19).
    """
    gold = gold or TRIAL_GOLD

    results = {}
    for instrument, predicted in [("PHQ-9", phq9), ("GAD-7", gad7), ("CompACT-10", compact10)]:
        g = gold.get(instrument)
        if g is None:
            continue

        results[instrument] = {
            "predicted": predicted,
            "gold": g,
            "predicted_total": sum(predicted),
            "gold_total": sum(g),
            "item_mae": round(item_mae(predicted, g), 3),
            "per_item_errors": per_item_errors(predicted, g),
            "total_error": total_score_error(predicted, g),
        }

        if instrument == "PHQ-9":
            results[instrument]["predicted_band"] = severity_band_phq9(sum(predicted))
            results[instrument]["gold_band"] = severity_band_phq9(sum(g))
            results[instrument]["band_correct"] = (
                results[instrument]["predicted_band"] == results[instrument]["gold_band"]
            )
        elif instrument == "GAD-7":
            results[instrument]["predicted_band"] = severity_band_gad7(sum(predicted))
            results[instrument]["gold_band"] = severity_band_gad7(sum(g))
            results[instrument]["band_correct"] = (
                results[instrument]["predicted_band"] == results[instrument]["gold_band"]
            )

    return results


def print_evaluation_report(results: dict) -> str:
    """Format evaluation results as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("MentalRiskES Task 1 — Evaluation Report")
    lines.append("=" * 60)

    for instrument in ["PHQ-9", "GAD-7", "CompACT-10"]:
        r = results.get(instrument)
        if r is None:
            continue

        lines.append(f"\n--- {instrument} ---")
        lines.append(f"  Predicted: {r['predicted']} (total={r['predicted_total']})")
        lines.append(f"  Gold:      {r['gold']} (total={r['gold_total']})")
        lines.append(f"  Item MAE:  {r['item_mae']}")
        lines.append(f"  Per-item:  {r['per_item_errors']}")
        lines.append(f"  Total err: {r['total_error']}")

        if "predicted_band" in r:
            match = "MATCH" if r["band_correct"] else "MISMATCH"
            lines.append(f"  Band: {r['predicted_band']} vs {r['gold_band']} [{match}]")

    report = "\n".join(lines)
    return report


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
