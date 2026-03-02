"""CLI entry point for the HiPerT-ADHD pipeline.

Usage:
    uv run hipert parse --stats-only
    uv run hipert retrieve --symptoms 5,12
    uv run hipert score --symptoms 5 --limit 5
    uv run hipert score --resume
    uv run hipert output --top-n 1000
    uv run hipert audit
    uv run hipert annotate-prep --symptoms 5
    uv run hipert run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from hipert.config import load_config


def _parse_symptom_ids(symptoms_str: str | None) -> list[int] | None:
    """Parse comma-separated symptom IDs string."""
    if symptoms_str is None:
        return None
    return [int(s.strip()) for s in symptoms_str.split(",") if s.strip()]


@click.group()
@click.option(
    "--config", "config_path",
    default="config/pipeline.yaml",
    help="Path to pipeline config YAML.",
)
@click.option(
    "--symptoms-config", "symptoms_path",
    default="config/symptoms.yaml",
    help="Path to symptoms config YAML.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, config_path: str, symptoms_path: str, verbose: bool) -> None:
    """HiPerT-ADHD: ADHD Symptom Sentence Ranking Pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["symptoms_path"] = symptoms_path
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--stats-only", is_flag=True, help="Print corpus stats only.")
@click.pass_context
def parse(ctx: click.Context, stats_only: bool) -> None:
    """Parse TREC corpus and display statistics."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    if ctx.obj["verbose"]:
        config = config  # log_level already set

    from hipert.pipeline.runner import PipelineRunner
    runner = PipelineRunner(config)

    try:
        stats = runner.run_parse(stats_only=stats_only)
        click.echo("\nCorpus Statistics:")
        click.echo(json.dumps(stats, indent=2))
    finally:
        runner.close()


@cli.command()
@click.option(
    "--symptoms", type=str, default=None,
    help="Comma-separated symptom IDs (default: all).",
)
@click.option(
    "--top-k", type=int, default=None,
    help="Override retrieval top-K per symptom.",
)
@click.pass_context
def retrieve(ctx: click.Context, symptoms: str | None, top_k: int | None) -> None:
    """Run bi-encoder retrieval and candidate selection."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    symptom_ids = _parse_symptom_ids(symptoms)

    from hipert.pipeline.runner import PipelineRunner
    runner = PipelineRunner(config)

    try:
        results = runner.run_retrieve(symptom_ids=symptom_ids, top_k=top_k)
        click.echo("\nRetrieval Results:")
        for sid, candidates in sorted(results.items()):
            click.echo(f"  Symptom {sid}: {len(candidates)} candidates")
    finally:
        runner.close()


@cli.command()
@click.option(
    "--symptoms", type=str, default=None,
    help="Comma-separated symptom IDs (default: all).",
)
@click.option(
    "--limit", type=int, default=None,
    help="Max sentences to score per symptom (for testing).",
)
@click.option("--dry-run", is_flag=True, help="Build prompts without calling LLM.")
@click.option(
    "--resume/--no-resume", default=True,
    help="Resume from checkpoint (default: resume).",
)
@click.pass_context
def score(
    ctx: click.Context,
    symptoms: str | None,
    limit: int | None,
    dry_run: bool,
    resume: bool,
) -> None:
    """Run LLM scoring cascade on candidates."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    symptom_ids = _parse_symptom_ids(symptoms)

    from hipert.pipeline.runner import PipelineRunner
    runner = PipelineRunner(config)

    try:
        runner.run_score(
            symptom_ids=symptom_ids,
            limit=limit,
            dry_run=dry_run,
            resume=resume,
        )
    finally:
        runner.close()


@cli.command()
@click.option(
    "--top-n", type=int, default=1000,
    help="Top-N sentences per symptom in output.",
)
@click.pass_context
def output(ctx: click.Context, top_n: int) -> None:
    """Generate TREC-format rankings from scored results."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    from hipert.pipeline.runner import PipelineRunner
    runner = PipelineRunner(config)

    try:
        runner.run_output(top_n=top_n)
        click.echo(f"Rankings written to {config.output_dir / 'rankings'}")
    finally:
        runner.close()


@cli.command()
@click.option(
    "--symptoms", type=str, default=None,
    help="Comma-separated symptom IDs (default: all).",
)
@click.pass_context
def audit(ctx: click.Context, symptoms: str | None) -> None:
    """Generate quality audit report."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    symptom_ids = _parse_symptom_ids(symptoms)

    from hipert.quality.audit import generate_audit_report, print_audit_report

    report = generate_audit_report(
        silver_labels_dir=config.output_dir / "silver_labels",
        symptom_ids=symptom_ids,
    )
    print_audit_report(report)

    # Also save to JSON
    report_path = config.output_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    click.echo(f"\nFull report saved to {report_path}")


@cli.command("annotate-prep")
@click.option(
    "--output-dir", default="candidates",
    help="Output directory for candidate TSV files.",
)
@click.option(
    "--annotations-dir", default="annotations",
    help="Output directory for annotation template JSON files.",
)
@click.option(
    "--symptoms", type=str, default=None,
    help="Comma-separated symptom IDs (default: all).",
)
@click.option(
    "--top-k", type=int, default=50,
    help="Top candidates to retrieve per symptom (default: 50).",
)
@click.option(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility.",
)
@click.pass_context
def annotate_prep(
    ctx: click.Context,
    output_dir: str,
    annotations_dir: str,
    symptoms: str | None,
    top_k: int,
    seed: int,
) -> None:
    """Generate annotation candidates and templates for few-shot examples."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symptom_ids = _parse_symptom_ids(symptoms)

    from scripts.retrieve_candidates import run_candidate_retrieval

    result = run_candidate_retrieval(
        config_path=ctx.obj["config_path"],
        symptoms_config=ctx.obj["symptoms_path"],
        symptom_ids=symptom_ids,
        output_dir=output_dir,
        annotations_dir=annotations_dir,
        top_k=top_k,
        seed=seed,
    )

    click.echo("\nAnnotation Prep Results:")
    for sid, counts in sorted(result["symptoms"].items()):
        click.echo(
            f"  Symptom {sid}: {counts['retrieval']} retrieval "
            f"+ {counts['score0']} score-0 = {counts['total']} total"
        )
    click.echo(f"  Shared score-0 pool: {result['score0_pool_size']} sentences")
    click.echo(f"\n  Candidates: {output_dir}/symptom_*_candidates.tsv")
    click.echo(f"  Templates:  {annotations_dir}/symptom_*_examples.json")


@cli.command()
@click.option(
    "--symptoms", type=str, default=None,
    help="Comma-separated symptom IDs (default: all).",
)
@click.option(
    "--limit", type=int, default=None,
    help="Max sentences to score per symptom.",
)
@click.pass_context
def run(ctx: click.Context, symptoms: str | None, limit: int | None) -> None:
    """Run the full pipeline end-to-end."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    symptom_ids = _parse_symptom_ids(symptoms)

    from hipert.pipeline.runner import PipelineRunner
    runner = PipelineRunner(config)

    try:
        click.echo("Step 1/4: Parsing corpus...")
        stats = runner.run_parse(stats_only=True)
        click.echo(f"  {stats['total_sentences']} sentences in {stats['total_files']} files")

        click.echo("Step 2/4: Retrieving candidates...")
        results = runner.run_retrieve(symptom_ids=symptom_ids)
        for sid, candidates in sorted(results.items()):
            click.echo(f"  Symptom {sid}: {len(candidates)} candidates")

        click.echo("Step 3/4: Scoring with LLM cascade...")
        runner.run_score(
            symptom_ids=symptom_ids,
            limit=limit,
            resume=True,
        )

        click.echo("Step 4/4: Generating TREC output...")
        runner.run_output()

        click.echo("\nPipeline complete!")
    finally:
        runner.close()


if __name__ == "__main__":
    cli()
