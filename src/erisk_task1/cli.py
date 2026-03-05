"""CLI entry point for the Task 1 pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .config import load_config


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure logging for the pipeline."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


@click.group()
def cli():
    """eRisk 2026 Task 1 — Conversational Depression Detection."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config/task1.yaml", help="Config YAML path")
@click.option("--run-id", type=int, default=1, help="Run ID (1, 2, or 3)")
@click.option("--personas", default=None, help="Comma-separated persona IDs (e.g., 4,5)")
@click.option("--output", default="runs/task1", help="Base output directory")
@click.option("--log-level", default="INFO", help="Log level")
def run(config_path: str, run_id: int, personas: str | None, output: str, log_level: str):
    """Run the full pipeline for specified personas.

    Example:
      uv run python -m erisk_task1.cli run --personas 4,5 --run-id 1
      uv run python -m erisk_task1.cli run --config config/task1_run2.yaml --personas 4,5 --run-id 2
    """
    config = load_config(config_path)
    config.run_id = run_id
    config.logging.output_dir = output

    if personas:
        config.persona_ids = [int(p.strip()) for p in personas.split(",")]

    # Setup logging with file output
    log_file = str(Path(output) / f"pipeline_run{run_id}.log")
    setup_logging(log_level, log_file)

    click.echo(f"Task 1 Pipeline — Run {run_id}")
    click.echo(f"Personas: {config.persona_ids}")
    click.echo(f"Output: {output}")
    click.echo(f"Assessor: {config.assessor.model} via {config.assessor.provider}")
    click.echo(f"Interviewer: {config.interviewer.model} via {config.interviewer.provider}")
    click.echo()

    from .pipeline import run_pipeline
    results = run_pipeline(config)

    click.echo(f"\nCompleted {len(results)} personas for run {run_id}")
    for r in results:
        click.echo(
            f"  persona{int(r.persona_id):02d}: "
            f"BDI={r.final_total} ({r.final_band.value}) "
            f"turns={len(r.conversation)//2}"
        )

    click.echo(f"\nPer-persona outputs in: {output}/persona{{ID}}/")
    click.echo(f"  interactions_{run_id}.json + results_{run_id}.json")


@cli.command()
@click.option("--output", default="runs/task1", help="Base output directory")
@click.option("--run-id", type=int, required=True, help="Run ID to merge")
@click.option("--personas", default=None, help="Comma-separated persona IDs (default: all found)")
def merge(output: str, run_id: int, personas: str | None):
    """Merge per-persona submission files into single files.

    Produces:
      runs/task1/interactions_{run_id}.json
      runs/task1/results_{run_id}.json

    Example:
      uv run python -m erisk_task1.cli merge --run-id 1 --personas 4,5
    """
    setup_logging("INFO")
    base_dir = Path(output)

    if personas:
        persona_ids = [int(p.strip()) for p in personas.split(",")]
    else:
        # Auto-discover persona dirs
        persona_ids = []
        for d in sorted(base_dir.glob("persona*")):
            if d.is_dir() and (d / f"results_{run_id}.json").exists():
                try:
                    pid = int(d.name.replace("persona", ""))
                    persona_ids.append(pid)
                except ValueError:
                    pass

    if not persona_ids:
        click.echo(f"No persona results found for run {run_id} in {base_dir}")
        return

    click.echo(f"Merging run {run_id} for personas: {persona_ids}")

    from .pipeline import merge_submission_files
    merge_submission_files(base_dir, persona_ids, run_id)

    click.echo(f"Merged files:")
    click.echo(f"  {base_dir}/interactions_{run_id}.json")
    click.echo(f"  {base_dir}/results_{run_id}.json")


@cli.command()
@click.option("--config", "config_path", default="config/task1.yaml", help="Config YAML path")
@click.option("--personas", default=None, help="Comma-separated persona IDs")
@click.option("--log-level", default="INFO", help="Log level")
def assess(config_path: str, personas: str | None, log_level: str):
    """Run assessors only on an existing conversation (for testing)."""
    setup_logging(log_level)
    config = load_config(config_path)

    if personas:
        config.persona_ids = [int(p.strip()) for p in personas.split(",")]

    from .llm_client import make_clients
    clients = make_clients(config)
    click.echo(f"Assessor client: {clients['assessor'].provider}/{clients['assessor'].model}")
    click.echo("Ready to assess. Use 'run' command for full pipeline.")


@cli.command()
@click.option("--base-model", default="all-mpnet-base-v2", help="Base sentence transformer model")
@click.option("--epochs", default=10, type=int, help="Training epochs")
@click.option("--batch-size", default=64, type=int, help="Batch size")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--loss", default="weighted_bce", type=click.Choice(["weighted_bce", "focal"]), help="Loss function")
@click.option("--output", default="models/symptom_transformer", help="Output directory")
@click.option("--device", default="cpu", help="Device (cpu, cuda, cuda:0)")
@click.option("--log-level", default="INFO", help="Log level")
def train(base_model: str, epochs: int, batch_size: int, lr: float, loss: str, output: str, device: str, log_level: str):
    """Train the symptom sentence transformer (Tier 2 model).

    Trains a multi-label classifier head on DepreSym + ReDSM5 + BDI-Sen
    datasets. Outputs to models/symptom_transformer/.

    Example:
      uv run python -m erisk_task1.cli train --epochs 10 --device cuda
      uv run python -m erisk_task1.cli train --loss focal --lr 1e-4
    """
    setup_logging(log_level)

    from .sentence_transformer import SentenceTransformerConfig, train_symptom_transformer

    config = SentenceTransformerConfig(
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        loss=loss,
        output_dir=output,
        device=device,
    )

    click.echo(f"Training symptom transformer")
    click.echo(f"  Base model: {base_model}")
    click.echo(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    click.echo(f"  Loss: {loss}, Device: {device}")
    click.echo(f"  Output: {output}")
    click.echo()

    model_dir = train_symptom_transformer(config)
    click.echo(f"\nTraining complete. Model saved to: {model_dir}")

    # Print summary stats
    import json
    stats_path = model_dir / "training_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        click.echo(f"Best macro F1: {stats['best_macro_f1']:.4f}")
        click.echo("Per-symptom F1:")
        for name, f1 in stats["per_symptom_f1"].items():
            click.echo(f"  {name}: {f1:.3f}")


@cli.command()
@click.argument("text")
def features(text: str):
    """Extract linguistic features from a text sample."""
    from .linguistic import extract_features
    feats = extract_features(text)
    click.echo(f"Word count: {feats.word_count}")
    click.echo(f"Absolutist: {feats.absolutist_count} ({feats.absolutist_ratio:.4f})")
    click.echo(f"  Words: {feats.absolutist_words_found}")
    click.echo(f"1st person singular ratio: {feats.first_person_singular_ratio:.4f}")
    click.echo(f"Negative emotion: {feats.negative_emotion_count}")
    click.echo(f"Positive emotion: {feats.positive_emotion_count}")
    click.echo(f"Hedging: {feats.hedging_count}")
    click.echo(f"Coping: {feats.coping_count}")
    click.echo(f"Discrepancy: {feats.discrepancy_count}")


if __name__ == "__main__":
    cli()
