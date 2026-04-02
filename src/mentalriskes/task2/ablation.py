"""Ablation runner: systematically compares pipeline configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import LLMConfig
from ..llm_client import LLMClient, create_llm_client
from .data import TRIAL_INFERRED_LABELS, discover_sessions, load_session_labels, load_trial_rounds
from .evaluation import accuracy, bootstrap_ci, cohens_kappa, evaluate_result, format_evaluation_report
from .pipeline import PipelineConfig, Task2Pipeline

logger = logging.getLogger(__name__)


def _create_client(llm_config: LLMConfig, model_override: str | None = None):
    """Create the right LLM client based on provider (LLMClient or HFInferenceClient)."""
    return create_llm_client(llm_config, model_override)


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
    fallback_config: LLMConfig | None = None,
) -> list[dict]:
    """Run ablation across all configs and return evaluation results.

    Args:
        configs: list of pipeline configurations to test.
        trial_dir: path to trial data directory.
        output_dir: directory to save results.
        llm_config: base LLM configuration (model overridden per config).
        labels: gold labels for evaluation. Defaults to TRIAL_INFERRED_LABELS.
        fallback_config: optional fallback LLM config (e.g. TogetherAI).

    Returns:
        List of evaluation dicts, sorted by accuracy descending.
    """
    if labels is None:
        labels = TRIAL_INFERRED_LABELS

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        logger.info("=== Ablation %d/%d: %s ===", i + 1, len(configs), cfg.config_id)

        # Only override model if the config model is compatible with the provider
        # (e.g. don't try "llama3.3:70b" on TogetherAI or "claude-*" on Ollama)
        model = cfg.model
        if llm_config.provider == "openai" and ":" in model:
            # Ollama-style model name on OpenAI-compatible provider — use provider default
            model = None
        elif llm_config.provider == "ollama" and "/" in model:
            # HF-style model name on Ollama — skip
            model = None
        elif llm_config.provider == "huggingface" and ":" in model:
            # Ollama-style model name on HF — skip
            model = None
        llm = _create_client(llm_config, model_override=model)
        if fallback_config is not None:
            fallback_llm = _create_client(fallback_config)
            llm.with_fallback(fallback_llm)
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


def run_multi_session_ablation(
    configs: list[PipelineConfig],
    simulated_dir: Path,
    output_dir: Path,
    llm_config: LLMConfig,
    fallback_config: LLMConfig | None = None,
) -> list[dict]:
    """Run ablation across all configs on multiple simulated sessions.

    For each config, runs the pipeline on every session directory found in
    simulated_dir, then aggregates predictions across all sessions for a
    single accuracy/kappa score per config.

    Args:
        configs: list of pipeline configurations to test.
        simulated_dir: root directory containing session subdirectories
            (each with round_*.json and labels.json).
        output_dir: directory to save results.
        llm_config: base LLM configuration (model overridden per config).
        fallback_config: optional fallback LLM config.

    Returns:
        List of evaluation dicts, sorted by accuracy descending.
    """
    import time

    sessions = discover_sessions(simulated_dir)
    if not sessions:
        logger.error("No simulated sessions found in %s", simulated_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        logger.info("=== Ablation %d/%d: %s (%d sessions) ===", i + 1, len(configs), cfg.config_id, len(sessions))
        t0 = time.monotonic()

        all_preds: list[int] = []
        all_labels: list[int] = []
        session_results: list[dict] = []

        for session_dir in sessions:
            session_id = session_dir.name
            labels = load_session_labels(session_dir)

            # Build fresh pipeline per session (reset state)
            model = cfg.model
            if llm_config.provider == "openai" and ":" in model:
                model = None
            elif llm_config.provider == "ollama" and "/" in model:
                model = None
            elif llm_config.provider == "huggingface" and ":" in model:
                model = None
            llm = _create_client(llm_config, model_override=model)
            if fallback_config is not None:
                fallback_llm = _create_client(fallback_config)
                llm.with_fallback(fallback_llm)

            pipeline = Task2Pipeline(llm=llm, config=cfg)
            result = pipeline.run_trial(session_dir)
            result_path = pipeline.save_result(result, output_dir / cfg.config_id)

            # Collect per-session predictions aligned with labels
            preds_dict = {r.round_id: r.selection.chosen_option for r in result.rounds}
            common_rounds = sorted(set(preds_dict.keys()) & set(labels.keys()))
            sess_preds = [preds_dict[r] for r in common_rounds]
            sess_labels = [labels[r] for r in common_rounds]

            sess_acc = accuracy(sess_preds, sess_labels)
            session_results.append({
                "session_id": session_id,
                "n_rounds": len(common_rounds),
                "accuracy": sess_acc,
                "predictions": {r: preds_dict[r] for r in common_rounds},
                "labels": {r: labels[r] for r in common_rounds},
            })

            all_preds.extend(sess_preds)
            all_labels.extend(sess_labels)

            logger.info("  Session %s: %d rounds, acc=%.1f%%", session_id, len(common_rounds), sess_acc * 100)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Aggregate metrics across all sessions
        agg_acc = accuracy(all_preds, all_labels)
        agg_kappa = cohens_kappa(all_preds, all_labels)
        agg_ci = bootstrap_ci(all_preds, all_labels)

        eval_result = {
            "config_id": cfg.config_id,
            "n_sessions": len(sessions),
            "n_rounds": len(all_preds),
            "accuracy": agg_acc,
            "cohens_kappa": agg_kappa,
            "bootstrap_ci_95": agg_ci,
            "sessions": session_results,
            "total_elapsed_ms": elapsed_ms,
        }
        results.append(eval_result)

        logger.info(
            "  AGGREGATE: %d rounds, acc=%.1f%%, kappa=%.3f, CI=[%.1f%%, %.1f%%]",
            len(all_preds), agg_acc * 100, agg_kappa, agg_ci[0] * 100, agg_ci[1] * 100,
        )

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
            "n_sessions": r["n_sessions"],
            "n_rounds": r["n_rounds"],
            "total_elapsed_ms": r.get("total_elapsed_ms", 0),
            "per_session": [
                {"session_id": s["session_id"], "accuracy": s["accuracy"], "n_rounds": s["n_rounds"]}
                for s in r["sessions"]
            ],
        })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Multi-session ablation summary saved to %s", summary_path)
    return results


def format_multi_session_summary(results: list[dict]) -> str:
    """Format multi-session ablation results as a comparison table."""
    lines = [
        "=" * 90,
        "MULTI-SESSION ABLATION SUMMARY",
        "=" * 90,
        f"{'Config':<40} {'Sess':>4} {'Rnds':>5} {'Acc':>6} {'Kappa':>7} {'CI 95%':>15} {'Time':>8}",
        "-" * 90,
    ]
    for r in results:
        ci = r.get("bootstrap_ci_95", (0, 0))
        time_s = r.get("total_elapsed_ms", 0) / 1000
        lines.append(
            f"{r['config_id']:<40} {r['n_sessions']:>4} {r['n_rounds']:>5} "
            f"{r['accuracy']:>5.1%} {r['cohens_kappa']:>7.3f} "
            f"[{ci[0]:.1%},{ci[1]:.1%}] {time_s:>7.1f}s"
        )

    # Per-session breakdown for top config
    if results:
        lines.append("")
        lines.append(f"Per-session breakdown (top config: {results[0]['config_id']}):")
        lines.append(f"  {'Session':<35} {'Rounds':>6} {'Accuracy':>8}")
        lines.append("  " + "-" * 52)
        for s in results[0].get("sessions", []):
            lines.append(f"  {s['session_id']:<35} {s['n_rounds']:>6} {s['accuracy']:>7.1%}")

    lines.append("=" * 90)
    return "\n".join(lines)


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
