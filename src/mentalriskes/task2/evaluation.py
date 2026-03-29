"""Evaluation metrics for Task 2: accuracy and Cohen's kappa."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cohens_kappa(predictions: list[int], labels: list[int], k: int = 3) -> float:
    """Compute Cohen's kappa for k-class classification.

    Args:
        predictions: predicted option numbers (1-indexed).
        labels: gold option numbers (1-indexed).
        k: number of classes.

    Returns:
        Cohen's kappa coefficient.
    """
    n = len(predictions)
    if n == 0:
        return 0.0

    # Build confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for pred, gold in zip(predictions, labels):
        matrix[pred - 1][gold - 1] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    pe = 0.0
    for i in range(k):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(k))
        pe += (row_sum * col_sum) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def accuracy(predictions: list[int], labels: list[int]) -> float:
    """Compute accuracy."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)


# Therapeutic phases for the trial data (from spec section 2.3)
TRIAL_PHASES: dict[int, str] = {
    1: "crisis",
    2: "committed_action", 3: "committed_action",
    4: "acceptance", 5: "acceptance",
    6: "defusion", 7: "defusion", 8: "defusion",
    9: "activation", 10: "activation", 11: "activation", 12: "activation",
    13: "integration", 14: "integration", 15: "integration",
    16: "self_as_context", 17: "self_as_context",
    18: "closing", 19: "closing",
}


def per_phase_accuracy(
    predictions: dict[int, int],
    labels: dict[int, int],
) -> dict[str, dict]:
    """Compute accuracy breakdown by therapeutic phase.

    Args:
        predictions: {round_id: chosen_option}
        labels: {round_id: correct_option}

    Returns:
        {phase: {correct: N, total: N, accuracy: float, rounds: [...]}}
    """
    phase_results: dict[str, dict] = {}

    for round_id in sorted(labels.keys()):
        if round_id not in predictions:
            continue
        phase = TRIAL_PHASES.get(round_id, "unknown")
        if phase not in phase_results:
            phase_results[phase] = {"correct": 0, "total": 0, "rounds": []}

        pred = predictions[round_id]
        gold = labels[round_id]
        is_correct = pred == gold
        phase_results[phase]["total"] += 1
        if is_correct:
            phase_results[phase]["correct"] += 1
        phase_results[phase]["rounds"].append({
            "round": round_id, "pred": pred, "gold": gold, "correct": is_correct
        })

    for phase, data in phase_results.items():
        data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0.0

    return phase_results


def bootstrap_ci(
    predictions: list[int],
    labels: list[int],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for accuracy."""
    import random

    n = len(predictions)
    if n == 0:
        return (0.0, 0.0)

    accs = []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        correct = sum(1 for i in indices if predictions[i] == labels[i])
        accs.append(correct / n)

    accs.sort()
    lo = accs[int((1 - confidence) / 2 * n_bootstrap)]
    hi = accs[int((1 + confidence) / 2 * n_bootstrap)]
    return (lo, hi)


def evaluate_result(
    result_path: Path,
    labels: dict[int, int],
) -> dict:
    """Evaluate a pipeline result JSONL against gold labels.

    Returns:
        Evaluation dict with accuracy, kappa, per-phase, and CIs.
    """
    # Load predictions from JSONL
    predictions: dict[int, int] = {}
    config_info = {}

    with open(result_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") == "config":
                config_info = entry
            elif entry.get("type") == "round":
                predictions[entry["round_id"]] = entry["chosen_option"]

    # Align predictions with labels
    common_rounds = sorted(set(predictions.keys()) & set(labels.keys()))
    pred_list = [predictions[r] for r in common_rounds]
    label_list = [labels[r] for r in common_rounds]

    acc = accuracy(pred_list, label_list)
    kappa = cohens_kappa(pred_list, label_list)
    ci = bootstrap_ci(pred_list, label_list)
    phase_acc = per_phase_accuracy(predictions, labels)

    return {
        "config_id": config_info.get("config_id", result_path.stem),
        "n_rounds": len(common_rounds),
        "accuracy": acc,
        "cohens_kappa": kappa,
        "bootstrap_ci_95": ci,
        "per_phase": phase_acc,
        "predictions": predictions,
        "total_elapsed_ms": config_info.get("total_elapsed_ms", 0),
        "llm_stats": config_info.get("llm_stats", {}),
    }


def format_evaluation_report(eval_result: dict) -> str:
    """Format evaluation result as a human-readable report."""
    lines = [
        f"=== {eval_result['config_id']} ===",
        f"Rounds evaluated: {eval_result['n_rounds']}",
        f"Accuracy: {eval_result['accuracy']:.1%} ({int(eval_result['accuracy'] * eval_result['n_rounds'])}/{eval_result['n_rounds']})",
        f"Cohen's kappa: {eval_result['cohens_kappa']:.3f}",
        f"95% CI: [{eval_result['bootstrap_ci_95'][0]:.1%}, {eval_result['bootstrap_ci_95'][1]:.1%}]",
        "",
        "Per-phase accuracy:",
    ]

    for phase, data in eval_result["per_phase"].items():
        lines.append(f"  {phase}: {data['accuracy']:.0%} ({data['correct']}/{data['total']})")
        for r in data["rounds"]:
            mark = "OK" if r["correct"] else "XX"
            lines.append(f"    R{r['round']:2d}: pred={r['pred']} gold={r['gold']} {mark}")

    if eval_result.get("total_elapsed_ms"):
        lines.append(f"\nTotal time: {eval_result['total_elapsed_ms']/1000:.1f}s")
    if eval_result.get("llm_stats"):
        stats = eval_result["llm_stats"]
        lines.append(f"LLM calls: {stats.get('call_count', '?')}, tokens: {stats.get('total_tokens', '?')}")

    return "\n".join(lines)
