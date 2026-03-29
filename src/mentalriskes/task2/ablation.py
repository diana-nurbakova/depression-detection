"""Ablation runner: systematically compares pipeline configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import LLMConfig
from ..llm_client import LLMClient
from .data import TRIAL_INFERRED_LABELS, load_trial_rounds
from .evaluation import evaluate_result, format_evaluation_report
from .pipeline import PipelineConfig, Task2Pipeline

logger = logging.getLogger(__name__)


# Ablation schedule from spec section 6.2
def get_ablation_configs(
    local_model: str = "llama3.3:70b",
    api_model: str = "claude-sonnet-4-20250514",
) -> list[PipelineConfig]:
    """Return the 11 ablation configurations from the spec."""
    configs = []

    # Pass 1 — Core comparison (4 runs)
    configs.append(PipelineConfig(name="P1_C1", model=local_model, framing="FUNC", pipeline="B", lang="es", lookback_window=3))
    configs.append(PipelineConfig(name="P1_C2", model=api_model, framing="FUNC", pipeline="B", lang="es", lookback_window=3))
    configs.append(PipelineConfig(name="P1_C3", model=local_model, framing="FUNC", pipeline="B", lang="en", lookback_window=3))
    configs.append(PipelineConfig(name="P1_C4", model=api_model, framing="FUNC", pipeline="B", lang="en", lookback_window=3))

    # Pass 2 — Framing variants (3 runs) — uses placeholder; replace with best from P1
    configs.append(PipelineConfig(name="P2_C5", model=local_model, framing="HYB", pipeline="B", lang="es", lookback_window=3))
    configs.append(PipelineConfig(name="P2_C6", model=local_model, framing="TOM-B", pipeline="B", lang="es", lookback_window=3))
    configs.append(PipelineConfig(name="P2_C7", model=local_model, framing="TOM-C", pipeline="B", lang="es", lookback_window=3))

    # Pass 3 — Structure + lookback (3 runs)
    configs.append(PipelineConfig(name="P3_C8", model=local_model, framing="FUNC", pipeline="B+", lang="es", lookback_window=3))
    configs.append(PipelineConfig(name="P3_C9", model=local_model, framing="FUNC", pipeline="B", lang="es", lookback_window=1))
    configs.append(PipelineConfig(name="P3_C10", model=local_model, framing="FUNC", pipeline="B", lang="es", lookback_window=5))

    # Pass 4 — Permutation voting (1 run)
    configs.append(PipelineConfig(name="P4_C11", model=local_model, framing="FUNC", pipeline="B", lang="es", lookback_window=3, permutation_voting=True))

    return configs


def run_ablation(
    configs: list[PipelineConfig],
    trial_dir: Path,
    output_dir: Path,
    llm_config: LLMConfig,
    labels: dict[int, int] | None = None,
) -> list[dict]:
    """Run ablation across all configs and return evaluation results.

    Args:
        configs: list of pipeline configurations to test.
        trial_dir: path to trial data directory.
        output_dir: directory to save results.
        llm_config: base LLM configuration (model overridden per config).
        labels: gold labels for evaluation. Defaults to TRIAL_INFERRED_LABELS.

    Returns:
        List of evaluation dicts, sorted by accuracy descending.
    """
    if labels is None:
        labels = TRIAL_INFERRED_LABELS

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        logger.info("=== Ablation %d/%d: %s ===", i + 1, len(configs), cfg.config_id)

        llm = LLMClient.from_config(llm_config, model_override=cfg.model)
        pipeline = Task2Pipeline(llm=llm, config=cfg)
        result = pipeline.run_trial(trial_dir)
        result_path = pipeline.save_result(result, output_dir)

        eval_result = evaluate_result(result_path, labels)
        results.append(eval_result)

        report = format_evaluation_report(eval_result)
        logger.info("\n%s\n", report)

    # Sort by accuracy descending
    results.sort(key=lambda r: r["accuracy"], reverse=True)

    # Save summary
    summary_path = output_dir / "ablation_summary.json"
    summary = []
    for r in results:
        summary.append({
            "config_id": r["config_id"],
            "accuracy": r["accuracy"],
            "cohens_kappa": r["cohens_kappa"],
            "bootstrap_ci_95": r["bootstrap_ci_95"],
            "n_rounds": r["n_rounds"],
            "total_elapsed_ms": r.get("total_elapsed_ms", 0),
        })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Ablation summary saved to %s", summary_path)
    return results


def format_ablation_summary(results: list[dict]) -> str:
    """Format ablation results as a comparison table."""
    lines = [
        "=" * 80,
        "ABLATION SUMMARY",
        "=" * 80,
        f"{'Config':<45} {'Acc':>6} {'Kappa':>7} {'CI 95%':>15} {'Time':>8}",
        "-" * 80,
    ]
    for r in results:
        ci = r.get("bootstrap_ci_95", (0, 0))
        time_s = r.get("total_elapsed_ms", 0) / 1000
        lines.append(
            f"{r['config_id']:<45} {r['accuracy']:>5.1%} {r['cohens_kappa']:>7.3f} "
            f"[{ci[0]:.1%},{ci[1]:.1%}] {time_s:>7.1f}s"
        )
    lines.append("=" * 80)
    return "\n".join(lines)
