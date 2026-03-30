"""Shared evaluation metrics for MentalRiskES 2026.

Implements the metric families from the organizer evaluation scripts
(sinai-uja/MentalRiskES-IberLEF):

  - Regression: RMSE, Pearson correlation, MAE
  - Classification: accuracy, macro-F1, Cohen's kappa (weighted)
  - Early detection: ERDE_o, latency-weighted F1
  - Ranking: Precision@K
  - Efficiency: emissions aggregation

These are used by both Task 1 (item-level ordinal prediction) and
Task 2 (therapist response selection).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Regression metrics (Task 1: item-level and total-score)
# ============================================================================

def rmse(predicted: list[float], gold: list[float]) -> float:
    """Root Mean Squared Error."""
    p, g = np.asarray(predicted, dtype=float), np.asarray(gold, dtype=float)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def mae(predicted: list[float], gold: list[float]) -> float:
    """Mean Absolute Error."""
    p, g = np.asarray(predicted, dtype=float), np.asarray(gold, dtype=float)
    return float(np.mean(np.abs(p - g)))


def pearson_r(predicted: list[float], gold: list[float]) -> float:
    """Pearson correlation coefficient.

    Returns 0.0 if either series has zero variance (constant predictions).
    """
    p, g = np.asarray(predicted, dtype=float), np.asarray(gold, dtype=float)
    if len(p) < 2 or np.std(p) == 0 or np.std(g) == 0:
        return 0.0
    return float(np.corrcoef(p, g)[0, 1])


def per_item_rmse(
    predicted_items: list[list[int]],
    gold_items: list[list[int]],
) -> list[float]:
    """RMSE computed per item across multiple samples.

    Args:
        predicted_items: list of predictions, each a list of item scores.
        gold_items: list of gold, each a list of item scores.

    Returns:
        list of RMSE values, one per item.
    """
    pred = np.array(predicted_items, dtype=float)
    gold = np.array(gold_items, dtype=float)
    return [float(np.sqrt(np.mean((pred[:, i] - gold[:, i]) ** 2)))
            for i in range(pred.shape[1])]


def per_dimension_metrics(
    predicted_items: list[list[int]],
    gold_items: list[list[int]],
    dimension_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-dimension RMSE and Pearson (matches ClassMultiRegressionEvaluation).

    Each "dimension" is one item position across all samples.

    Returns:
        {dimension_name: {rmse: float, pearson: float, mae: float}}
    """
    pred = np.array(predicted_items, dtype=float)
    gold = np.array(gold_items, dtype=float)
    n_dims = pred.shape[1]

    if dimension_names is None:
        dimension_names = [f"item_{i+1}" for i in range(n_dims)]

    results = {}
    for i, name in enumerate(dimension_names):
        p_col = pred[:, i].tolist()
        g_col = gold[:, i].tolist()
        results[name] = {
            "rmse": rmse(p_col, g_col),
            "pearson": pearson_r(p_col, g_col),
            "mae": mae(p_col, g_col),
        }

    # Averages
    results["_mean"] = {
        "rmse": float(np.mean([r["rmse"] for r in results.values() if not r == results.get("_mean")])),
        "pearson": float(np.mean([r["pearson"] for r in results.values() if not r == results.get("_mean")])),
        "mae": float(np.mean([r["mae"] for r in results.values() if not r == results.get("_mean")])),
    }

    return results


# ============================================================================
# Classification metrics (severity bands, Task 2 option selection)
# ============================================================================

def accuracy(predicted: list, gold: list) -> float:
    """Simple accuracy."""
    if not predicted:
        return 0.0
    return sum(1 for p, g in zip(predicted, gold) if p == g) / len(predicted)


def macro_f1(predicted: list, gold: list) -> float:
    """Macro-averaged F1 score."""
    labels = sorted(set(gold) | set(predicted))
    f1s = []
    for label in labels:
        tp = sum(1 for p, g in zip(predicted, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def cohens_kappa_weighted(
    predicted: list[int],
    gold: list[int],
    weights: str = "linear",
) -> float:
    """Weighted Cohen's kappa for ordinal data.

    Args:
        predicted: predicted values.
        gold: gold values.
        weights: "linear" or "quadratic".

    Returns:
        Weighted kappa coefficient.
    """
    if not predicted:
        return 0.0

    labels = sorted(set(predicted) | set(gold))
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(predicted)

    # Confusion matrix
    matrix = np.zeros((n_labels, n_labels))
    for p, g in zip(predicted, gold):
        matrix[label_to_idx[p]][label_to_idx[g]] += 1

    # Weight matrix
    w = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if weights == "linear":
                w[i][j] = abs(i - j) / (n_labels - 1) if n_labels > 1 else 0
            else:  # quadratic
                w[i][j] = (i - j) ** 2 / (n_labels - 1) ** 2 if n_labels > 1 else 0

    # Observed and expected
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    observed_w = np.sum(w * matrix) / n
    expected_w = np.sum(w * expected) / n

    if expected_w == 0:
        return 1.0
    return float(1 - observed_w / expected_w)


# ============================================================================
# Early Risk Detection Error (ERDE)
# ============================================================================

def erde(
    predictions_by_round: dict[int, float],
    gold_positive: bool,
    o: int = 5,
) -> float:
    """Early Risk Detection Error.

    Penalizes late correct detections with a sigmoid cost function.

    Args:
        predictions_by_round: {round: predicted_severity_score} across rounds.
        gold_positive: whether this user is truly at risk.
        o: ERDE parameter (5 or 30 typically). Controls the penalty ramp.

    Returns:
        ERDE cost (lower is better).
    """
    if not gold_positive:
        # True negative: cost is 0 if we never flag, 1 if we do
        # For regression tasks, we treat "flagging" as exceeding a threshold
        return 0.0

    # For positive cases: find earliest round where we detect
    # Cost function: cost(k) = 1 - 1/(1 + exp(k - o))
    # where k is the round of detection (1-indexed)
    for round_k in sorted(predictions_by_round.keys()):
        score = predictions_by_round[round_k]
        if score > 0:  # detected
            cost = 1 - 1 / (1 + math.exp(round_k - o))
            return cost

    # Never detected: cost = 1
    return 1.0


def latency_weighted_f1(
    session_detections: list[dict],
    o: int = 5,
) -> float:
    """Latency-weighted F1 as used by MentalRiskES organizers.

    Args:
        session_detections: list of {gold_positive, detection_round, detected}.
        o: ERDE parameter.

    Returns:
        Latency-weighted F1 score.
    """
    # Simplified: compute F1 then penalize by average latency
    tp = sum(1 for s in session_detections if s["gold_positive"] and s["detected"])
    fp = sum(1 for s in session_detections if not s["gold_positive"] and s["detected"])
    fn = sum(1 for s in session_detections if s["gold_positive"] and not s["detected"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Average latency penalty for true positives
    if tp > 0:
        latency_costs = []
        for s in session_detections:
            if s["gold_positive"] and s["detected"]:
                k = s["detection_round"]
                cost = 1 / (1 + math.exp(k - o))  # inverted sigmoid: earlier = higher
                latency_costs.append(cost)
        avg_speed = float(np.mean(latency_costs))
        return f1 * avg_speed

    return 0.0


# ============================================================================
# Ranking: Precision@K
# ============================================================================

def precision_at_k(
    predicted_scores: list[float],
    gold_scores: list[float],
    k: int = 10,
) -> float:
    """Precision@K: fraction of top-K predicted that are in true top-K.

    Used for identifying the most severe cases.
    """
    if len(predicted_scores) < k:
        k = len(predicted_scores)
    if k == 0:
        return 0.0

    pred_top_k = set(np.argsort(predicted_scores)[-k:])
    gold_top_k = set(np.argsort(gold_scores)[-k:])

    overlap = len(pred_top_k & gold_top_k)
    return overlap / k


# ============================================================================
# Multi-instrument evaluation (Task 1 specific)
# ============================================================================

# Instrument item names for per-dimension reporting
INSTRUMENT_ITEMS = {
    "PHQ-9": [
        "anhedonia", "mood", "sleep", "fatigue", "appetite",
        "self_worth", "concentration", "psychomotor", "suicidality",
    ],
    "GAD-7": [
        "nervousness", "worry_control", "excessive_worry",
        "relaxation", "restlessness", "irritability", "fear",
    ],
    "CompACT-10": [
        "rushing", "coherent_living", "thought_suppression",
        "values_aligned", "avoidance", "inattentive",
        "persistence", "emotional_suppression", "autopilot", "perseverance",
    ],
}

# CompACT-10 triflex subscale groupings
COMPACT10_SUBSCALES = {
    "openness": [2, 4, 7],     # items 3, 5, 8 (0-indexed)
    "awareness": [0, 5, 8],    # items 1, 6, 9
    "valued_action": [1, 3, 6, 9],  # items 2, 4, 7, 10
}


def evaluate_task1_full(
    predicted: dict[str, list[int]],
    gold: dict[str, list[int]],
) -> dict:
    """Full evaluation suite for a single Task 1 prediction.

    Computes all metrics the organizers are likely to use.

    Args:
        predicted: {"PHQ-9": [9 ints], "GAD-7": [7 ints], "CompACT-10": [10 ints]}
        gold: same format.

    Returns:
        Comprehensive evaluation dict.
    """
    results = {}

    for instrument in ["PHQ-9", "GAD-7", "CompACT-10"]:
        p = predicted.get(instrument, [])
        g = gold.get(instrument, [])
        if not p or not g:
            continue

        inst_result = {
            "rmse": rmse(p, g),
            "mae": mae(p, g),
            "pearson": pearson_r(p, g),
            "total_predicted": sum(p),
            "total_gold": sum(g),
            "total_rmse": rmse([sum(p)], [sum(g)]),
            "per_item_errors": [abs(pi - gi) for pi, gi in zip(p, g)],
        }

        # Severity band (PHQ-9, GAD-7)
        if instrument == "PHQ-9":
            inst_result["predicted_band"] = _phq9_band(sum(p))
            inst_result["gold_band"] = _phq9_band(sum(g))
            inst_result["band_correct"] = inst_result["predicted_band"] == inst_result["gold_band"]
        elif instrument == "GAD-7":
            inst_result["predicted_band"] = _gad7_band(sum(p))
            inst_result["gold_band"] = _gad7_band(sum(g))
            inst_result["band_correct"] = inst_result["predicted_band"] == inst_result["gold_band"]

        # Weighted kappa (ordinal agreement)
        inst_result["kappa_linear"] = cohens_kappa_weighted(p, g, weights="linear")
        inst_result["kappa_quadratic"] = cohens_kappa_weighted(p, g, weights="quadratic")

        # CompACT-10 subscale breakdown
        if instrument == "CompACT-10":
            for sub_name, indices in COMPACT10_SUBSCALES.items():
                p_sub = [p[i] for i in indices]
                g_sub = [g[i] for i in indices]
                inst_result[f"subscale_{sub_name}_rmse"] = rmse(p_sub, g_sub)
                inst_result[f"subscale_{sub_name}_mae"] = mae(p_sub, g_sub)

        results[instrument] = inst_result

    # Cross-instrument averages (what ClassMultiRegressionEvaluation computes)
    all_rmse = [r["rmse"] for r in results.values()]
    all_pearson = [r["pearson"] for r in results.values()]
    results["_overall"] = {
        "mean_rmse": float(np.mean(all_rmse)) if all_rmse else 0.0,
        "mean_pearson": float(np.mean(all_pearson)) if all_pearson else 0.0,
    }

    return results


def format_task1_report(results: dict) -> str:
    """Format full Task 1 evaluation as a readable report."""
    lines = [
        "=" * 65,
        "MentalRiskES Task 1 — Full Evaluation Report",
        "=" * 65,
    ]

    for instrument in ["PHQ-9", "GAD-7", "CompACT-10"]:
        r = results.get(instrument)
        if not r:
            continue

        lines.append(f"\n--- {instrument} ---")
        lines.append(f"  RMSE:     {r['rmse']:.3f}")
        lines.append(f"  MAE:      {r['mae']:.3f}")
        lines.append(f"  Pearson:  {r['pearson']:.3f}")
        lines.append(f"  Kappa(L): {r['kappa_linear']:.3f}  Kappa(Q): {r['kappa_quadratic']:.3f}")
        lines.append(f"  Total:    {r['total_predicted']} vs {r['total_gold']} (err={abs(r['total_predicted']-r['total_gold'])})")
        lines.append(f"  Per-item: {r['per_item_errors']}")

        if "predicted_band" in r:
            match = "MATCH" if r["band_correct"] else "MISMATCH"
            lines.append(f"  Band:     {r['predicted_band']} vs {r['gold_band']} [{match}]")

        if instrument == "CompACT-10":
            for sub in ["openness", "awareness", "valued_action"]:
                lines.append(f"  {sub:16s} RMSE={r[f'subscale_{sub}_rmse']:.3f}  MAE={r[f'subscale_{sub}_mae']:.3f}")

    ov = results.get("_overall", {})
    if ov:
        lines.append(f"\n--- Overall ---")
        lines.append(f"  Mean RMSE:    {ov['mean_rmse']:.3f}")
        lines.append(f"  Mean Pearson: {ov['mean_pearson']:.3f}")

    return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================

def _phq9_band(total: int) -> str:
    if total <= 4: return "minimal"
    if total <= 9: return "mild"
    if total <= 14: return "moderate"
    if total <= 19: return "moderately_severe"
    return "severe"


def _gad7_band(total: int) -> str:
    if total <= 4: return "minimal"
    if total <= 9: return "mild"
    if total <= 14: return "moderate"
    return "severe"
