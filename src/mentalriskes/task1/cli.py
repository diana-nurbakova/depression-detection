"""CLI for MentalRiskES Task 1 pipeline.

Entry point: `mentalriskes <command>`
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from ..config import load_config
from .pipeline import Pipeline, setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "config_path", default="config/mentalriskes.yaml",
              help="Path to configuration YAML file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """MentalRiskES 2026 Task 1: Symptom Detection in Therapeutic Conversations."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--run", "run_name", default=None, help="Run only a specific run config (by name).")
@click.pass_context
def trial(ctx: click.Context, run_name: str | None) -> None:
    """Run pipeline on local trial data for calibration and testing."""
    config = load_config(ctx.obj["config_path"])
    level = "DEBUG" if ctx.obj["verbose"] else config.pipeline.log_level
    log_file = str(config.data.log_dir / "trial.log")
    setup_logging(level, log_file)

    if run_name:
        config.runs = [r for r in config.runs if r.name == run_name]
        if not config.runs:
            click.echo(f"Run '{run_name}' not found in config.", err=True)
            sys.exit(1)

    pipeline = Pipeline(config)
    results = pipeline.run_trial()

    # Print summary
    for rname, preds in results.items():
        click.echo(f"\n{'='*50}")
        click.echo(f"Run: {rname} — {len(preds)} predictions")

        if preds:
            last = preds[-1]
            click.echo(f"  Last round ({last.round_number}):")
            click.echo(f"    PHQ-9:      {last.phq9} (total={sum(last.phq9)})")
            click.echo(f"    GAD-7:      {last.gad7} (total={sum(last.gad7)})")
            click.echo(f"    CompACT-10: {last.compact10} (total={sum(last.compact10)})")

            if last.consistency_warnings:
                click.echo(f"    Warnings: {len(last.consistency_warnings)}")
                for w in last.consistency_warnings:
                    click.echo(f"      - {w['rule']}: {w['message']}")


@cli.command()
@click.pass_context
def server(ctx: click.Context) -> None:
    """Run pipeline against competition server (GET/POST loop)."""
    config = load_config(ctx.obj["config_path"])
    level = "DEBUG" if ctx.obj["verbose"] else config.pipeline.log_level
    log_file = str(config.data.log_dir / "server.log")
    setup_logging(level, log_file)

    if not config.server.base_url or not config.server.token:
        click.echo("Error: MENTALRISKES_BASE_URL and MENTALRISKES_TOKEN env vars required.", err=True)
        sys.exit(1)

    pipeline = Pipeline(config)
    pipeline.run_server()


@cli.command()
@click.option("--run", "run_name", default=None, help="Evaluate a specific run.")
@click.pass_context
def evaluate(ctx: click.Context, run_name: str | None) -> None:
    """Evaluate trial predictions against manual annotations."""
    config = load_config(ctx.obj["config_path"])
    setup_logging(config.pipeline.log_level)

    from .evaluation import evaluate_trial_run, print_evaluation_report

    log_dir = config.data.log_dir
    pattern = f"predictions_{run_name}.jsonl" if run_name else "predictions_*.jsonl"

    for log_path in sorted(log_dir.glob(pattern)):
        click.echo(f"\nEvaluating: {log_path.name}")
        results = evaluate_trial_run(log_path)

        for sid, eval_result in results.items():
            click.echo(f"\nSession: {sid}")
            report = print_evaluation_report(eval_result)
            click.echo(report)


@cli.command()
@click.option("--output", "-o", default=None, help="Output path for extracted examples.")
@click.pass_context
def extract_primate(ctx: click.Context, output: str | None) -> None:
    """Extract PHQ-9 few-shot examples from the PRIMATE dataset."""
    config = load_config(ctx.obj["config_path"])
    setup_logging(config.pipeline.log_level)

    from .data import load_primate_dataset
    from .primate import compute_cooccurrence, compute_prevalence, extract_primate_examples

    posts = load_primate_dataset(config.data.primate_path)

    click.echo(f"Loaded {len(posts)} posts")
    click.echo("Extracting examples...")

    examples = extract_primate_examples(posts)
    prevalence = compute_prevalence(posts)
    cooccurrence = compute_cooccurrence(posts)

    output_dir = Path(output) if output else config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save examples
    examples_path = output_dir / "phq9_examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    click.echo(f"Examples saved to {examples_path}")

    # Save prevalence
    prev_path = output_dir / "phq9_prevalence.json"
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(prevalence, f, indent=2)
    click.echo(f"Prevalence saved to {prev_path}")

    # Save co-occurrence
    cooc_path = output_dir / "phq9_cooccurrence.json"
    from .primate import PHQ9_ITEMS
    symptom_names = [PHQ9_ITEMS[i]["name"] for i in range(9)]
    with open(cooc_path, "w", encoding="utf-8") as f:
        json.dump({"symptom_names": symptom_names, "matrix": cooccurrence}, f, indent=2)

    # Print summary
    click.echo("\nSymptom Prevalence:")
    for name, data in prevalence.items():
        click.echo(f"  {name:<20} {data['count']:>5} ({data['prevalence']:.3f})")

    # Print example counts
    click.echo("\nExamples per symptom:")
    for key, data in examples.items():
        click.echo(f"  {key:<30} pos={len(data['positive_examples'])} neg={len(data['negative_examples'])}")


@cli.command()
@click.option("--configs", default="A0,A1,A3,A5",
              help="Comma-separated config names to run (A0-A5). Default: A0,A1,A3,A5.")
@click.option("--output-dir", default="runs/mentalriskes_ablation",
              help="Directory to save per-config results and report.")
@click.option("--posthoc", "raw_log", default=None,
              help="Path to existing predictions JSONL: apply B/C post-hoc (no LLM re-run).")
@click.option("--level-b/--no-level-b", default=True,
              help="Apply Level B constraints in post-hoc mode.")
@click.option("--level-c/--no-level-c", default=False,
              help="Apply Level C agent in post-hoc mode (requires Ollama).")
@click.pass_context
def ablation(
    ctx: click.Context,
    configs: str,
    output_dir: str,
    raw_log: str | None,
    level_b: bool,
    level_c: bool,
) -> None:
    """Run calibration ablation study (A0–A5) on trial data.

    Two modes:\n
    1. Full LLM ablation (default): runs each config end-to-end, requires Ollama.\n
       Example: mentalriskes ablation --configs A0,A1,A3,A5\n
    2. Post-hoc mode: applies Level B/C to an existing prediction log.\n
       Example: mentalriskes ablation --posthoc output/mentalriskes/logs/predictions_run0_A5.jsonl
    """
    config = load_config(ctx.obj["config_path"])
    level = "DEBUG" if ctx.obj["verbose"] else config.pipeline.log_level
    setup_logging(level)

    from .ablation import (
        ABLATION_CONFIGS,
        format_ablation_comparison,
        format_posthoc_report,
        posthoc_calibration_ablation,
        run_ablation_study,
    )
    from .evaluation import TRIAL_GOLD

    if raw_log:
        # Post-hoc mode: apply B/C to existing prediction log
        click.echo(f"Post-hoc calibration ablation on: {raw_log}")
        click.echo(f"  Level B: {level_b} | Level C: {level_c}")

        client = None
        if level_c:
            from ..llm_client import create_llm_client
            client = create_llm_client(config.llm)

        result = posthoc_calibration_ablation(
            raw_log, gold=TRIAL_GOLD,
            apply_level_b=level_b, apply_level_c=level_c,
            client=client,
        )
        report = format_posthoc_report(result, TRIAL_GOLD)
        click.echo(report)

        # Save
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = f"{'B' if level_b else ''}{'C' if level_c else ''}"
        out_path = out / f"posthoc_{suffix}_{Path(raw_log).stem}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        click.echo(f"\nSaved to: {out_path}")
    else:
        # Full LLM ablation
        config_names = [c.strip() for c in configs.split(",") if c.strip()]
        unknown = [c for c in config_names if c not in ABLATION_CONFIGS]
        if unknown:
            click.echo(f"Unknown configs: {unknown}. Valid: {list(ABLATION_CONFIGS.keys())}", err=True)
            sys.exit(1)

        click.echo(f"Running ablation configs: {config_names}")
        click.echo(f"Trial data: {config.data.trial_dir}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"LLM: {config.llm.model} via {config.llm.provider}")
        click.echo("")

        results = run_ablation_study(
            configs=config_names,
            trial_dir=config.data.trial_dir,
            pipeline_cfg=config,
            output_dir=output_dir,
        )

        if results:
            report = format_ablation_comparison(results)
            click.echo(report)


@cli.command("sim-ablation")
@click.option("--simulated-dir", required=True,
              help="Root directory with simulated persona sessions (each sub-dir has round_N.json + metadata.json).")
@click.option("--configs", default="A0,A1,A3,A5",
              help="Comma-separated ablation config names. Default: A0,A1,A3,A5.")
@click.option("--output-dir", default="runs/mentalriskes_simulated_ablation",
              help="Directory to save results and report.")
@click.pass_context
def sim_ablation(
    ctx: click.Context,
    simulated_dir: str,
    configs: str,
    output_dir: str,
) -> None:
    """Run calibration ablation on simulated persona sessions.

    Validates Level A/B/C calibration tiers across multiple synthetic patient
    profiles. Gold labels are derived from simulator metadata
    (phq9_total, gad7_total, compact10_profile).

    Typical usage after generating sessions with mentalriskes-dataprep simulate:\\n
      mentalriskes sim-ablation --simulated-dir output/mentalriskes/data_prep/simulated/task1

    Use --configs A0,A1,A3,A5 for the standard ablation or A0,A1,A2,A3,A4,A5
    for the full 6-config study.
    """
    config = load_config(ctx.obj["config_path"])
    level = "DEBUG" if ctx.obj["verbose"] else config.pipeline.log_level
    setup_logging(level)

    from .ablation import ABLATION_CONFIGS, run_ablation_on_simulated

    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in ABLATION_CONFIGS]
    if unknown:
        click.echo(f"Unknown configs: {unknown}. Valid: {list(ABLATION_CONFIGS.keys())}", err=True)
        import sys
        sys.exit(1)

    click.echo(f"Simulated ablation on: {simulated_dir}")
    click.echo(f"Configs: {config_names}")
    click.echo(f"LLM: {config.llm.model} via {config.llm.provider}")
    click.echo(f"Output: {output_dir}")
    click.echo("")

    results = run_ablation_on_simulated(
        simulated_dir=simulated_dir,
        configs=config_names,
        pipeline_cfg=config,
        output_dir=output_dir,
    )

    click.echo(f"\nCompleted: {len(results)} sessions processed.")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show current configuration and resource status."""
    config = load_config(ctx.obj["config_path"])

    click.echo("MentalRiskES 2026 Task 1 Configuration")
    click.echo("=" * 50)

    click.echo(f"\nServer:")
    click.echo(f"  URL:       {'[set]' if config.server.base_url else '[NOT SET]'}")
    click.echo(f"  Token:     {'[set]' if config.server.token else '[NOT SET]'}")
    click.echo(f"  Mode:      {'trial' if config.server.use_trial else 'test'}")

    click.echo(f"\nLLM:")
    click.echo(f"  Provider:  {config.llm.provider}")
    click.echo(f"  URL:       {'[set]' if config.llm.base_url else '[NOT SET]'}")
    click.echo(f"  Model:     {config.llm.model}")

    click.echo(f"\nRuns ({len(config.runs)}):")
    for r in config.runs:
        click.echo(f"  {r.name}: {r.description}")
        click.echo(f"    model={r.model} calibration={r.calibration} few_shot={r.few_shot}")
        click.echo(f"    prompt_anchors={r.prompt_anchors} level_b={r.level_b} level_c={r.level_c}")

    click.echo(f"\nData:")
    click.echo(f"  Trial:     {config.data.trial_dir} ({'exists' if config.data.trial_dir.exists() else 'MISSING'})")
    click.echo(f"  PRIMATE:   {config.data.primate_path} ({'exists' if config.data.primate_path.exists() else 'MISSING'})")
    click.echo(f"  Output:    {config.data.output_dir}")

    click.echo(f"\nResources:")
    for name, path in [
        ("Calibration", config.resources.calibration_config),
        ("ACT vocab", config.resources.act_vocabulary),
        ("Hexaflex", config.resources.hexaflex_quotes),
    ]:
        status = "exists" if path.exists() else "MISSING"
        click.echo(f"  {name:<12} {path} ({status})")
