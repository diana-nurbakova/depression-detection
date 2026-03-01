"""Per-symptom reliability audit.

Generates audit reports after scoring: escalation rates, score
distributions, Llama-GPT agreement rates.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_audit_report(
    silver_labels_dir: Path,
    symptom_ids: list[int] | None = None,
) -> dict:
    """Generate a comprehensive audit report across symptoms.

    Args:
        silver_labels_dir: Directory containing symptom_N.jsonl files.
        symptom_ids: Symptoms to audit. If None, all found.

    Returns:
        Audit report dictionary.
    """
    if symptom_ids is None:
        symptom_ids = list(range(1, 19))

    report: dict = {"symptoms": {}, "overall": {}}
    all_labels = []
    all_escalated = 0
    all_total = 0

    for symptom_id in symptom_ids:
        jsonl_path = silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not jsonl_path.exists():
            continue

        results = _load_results(jsonl_path)
        if not results:
            continue

        symptom_report = _audit_symptom(symptom_id, results)
        report["symptoms"][symptom_id] = symptom_report

        all_labels.extend(r.get("final_label", 0) for r in results)
        all_escalated += symptom_report["escalated_count"]
        all_total += symptom_report["total_count"]

    # Overall statistics
    if all_total > 0:
        overall_dist = Counter(all_labels)
        report["overall"] = {
            "total_scored": all_total,
            "total_escalated": all_escalated,
            "escalation_rate": round(all_escalated / all_total, 4),
            "score_distribution": {
                score: round(overall_dist.get(score, 0) / all_total, 4)
                for score in range(4)
            },
        }

    return report


def _audit_symptom(symptom_id: int, results: list[dict]) -> dict:
    """Generate audit for a single symptom."""
    total = len(results)
    labels = [r.get("final_label", 0) for r in results]
    escalated = [r for r in results if r.get("escalated", False)]

    # Score distribution
    dist = Counter(labels)

    # Escalation analysis
    escalation_triggers: Counter = Counter()
    agreements = 0
    disagreements = 0

    for r in escalated:
        for trigger in r.get("escalation_triggers", []):
            # Extract rule number
            if "Rule" in trigger:
                rule = trigger.split(":")[0].strip()
                escalation_triggers[rule] += 1

        if r.get("gpt_output") is not None:
            llama_score = r.get("llama_output", {}).get("score", -1)
            gpt_score = r.get("gpt_output", {}).get("score", -1)
            if llama_score == gpt_score:
                agreements += 1
            else:
                disagreements += 1

    return {
        "total_count": total,
        "score_distribution": {
            score: round(dist.get(score, 0) / total, 4) if total else 0
            for score in range(4)
        },
        "mean_score": round(sum(labels) / total, 3) if total else 0,
        "escalated_count": len(escalated),
        "escalation_rate": round(len(escalated) / total, 4) if total else 0,
        "escalation_triggers": dict(escalation_triggers),
        "llama_gpt_agreement": {
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": (
                round(agreements / (agreements + disagreements), 4)
                if (agreements + disagreements) > 0 else None
            ),
        },
        "confidence_weight_mean": round(
            sum(r.get("confidence_weight", 0) for r in results) / total, 4,
        ) if total else 0,
    }


def _load_results(filepath: Path) -> list[dict]:
    """Load results from a JSONL file."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def print_audit_report(report: dict) -> None:
    """Print a human-readable audit report."""
    print("\n" + "=" * 60)
    print("HIPERT-ADHD SCORING AUDIT REPORT")
    print("=" * 60)

    if "overall" in report and report["overall"]:
        overall = report["overall"]
        print(f"\nOverall: {overall['total_scored']} sentences scored")
        print(f"Escalation rate: {overall['escalation_rate']:.1%}")
        print("Score distribution:")
        for score in range(4):
            pct = overall["score_distribution"].get(score, 0)
            bar = "#" * int(pct * 50)
            print(f"  Score {score}: {pct:6.1%} {bar}")

    print("\nPer-Symptom Breakdown:")
    print("-" * 60)

    for symptom_id in sorted(report.get("symptoms", {}).keys()):
        s = report["symptoms"][symptom_id]
        print(
            f"  Item {symptom_id:2d}: "
            f"{s['total_count']:5d} scored, "
            f"esc={s['escalation_rate']:.0%}, "
            f"mean={s['mean_score']:.2f}, "
            f"dist=[{s['score_distribution'][0]:.0%}/"
            f"{s['score_distribution'][1]:.0%}/"
            f"{s['score_distribution'][2]:.0%}/"
            f"{s['score_distribution'][3]:.0%}]",
        )
        if s["llama_gpt_agreement"]["agreement_rate"] is not None:
            print(
                f"          Llama-GPT agreement: "
                f"{s['llama_gpt_agreement']['agreement_rate']:.0%}",
            )

    print("=" * 60)
