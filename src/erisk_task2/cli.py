"""CLI entry point for eRisk 2026 Task 2 pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from erisk_task2.config import load_config


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.option("--config", "-c", default="config/task2.yaml", help="Config YAML path")
@click.pass_context
def cli(ctx, config):
    """eRisk 2026 Task 2: Contextualized Early Detection of Depression."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    setup_logging(ctx.obj["config"].logging.log_level)


@cli.command()
@click.option("--data-dir", required=True, help="Path to training data directory (all_combined/)")
@click.option("--labels", required=True, help="Path to ground truth labels file")
@click.option("--output-dir", default="./runs/task2/train", help="Output directory")
@click.pass_context
def train(ctx, data_dir, labels, output_dir):
    """Train classifiers on training data with cross-validation."""
    from erisk_task2.pipeline import train_pipeline
    cfg = ctx.obj["config"]
    cfg.training_data_dir = data_dir
    cfg.labels_path = labels
    cfg.logging.output_dir = output_dir
    # Add file logging to output directory
    log_path = Path(output_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    train_pipeline(cfg)


@cli.command()
@click.option("--official/--unofficial", default=False, help="Use official server")
@click.pass_context
def run(ctx, official):
    """Run live competition pipeline against eRisk server."""
    from erisk_task2.pipeline import run_pipeline
    cfg = ctx.obj["config"]
    if official:
        cfg.server.base_url = "https://erisk.irlab.org/challenge-t2-official"
    run_pipeline(cfg)


@cli.command()
@click.option("--data-dir", required=True, help="Path to training data directory")
@click.option("--labels", required=True, help="Path to ground truth labels file")
@click.pass_context
def evaluate(ctx, data_dir, labels):
    """Run offline evaluation on training data (simulate round-by-round)."""
    from erisk_task2.pipeline import evaluate_pipeline
    cfg = ctx.obj["config"]
    cfg.training_data_dir = data_dir
    cfg.labels_path = labels
    evaluate_pipeline(cfg)


if __name__ == "__main__":
    cli()
