"""CLI entry point for the HiPerT-ADHD pipeline.

Usage:
    uv run hipert parse --stats-only
    uv run hipert retrieve --symptoms 5,12
    uv run hipert score --symptoms 5 --limit 5
    uv run hipert score --resume
    uv run hipert output --top-n 1000
    uv run hipert output --run 2 --top-n 1000
    uv run hipert output --run all
    uv run hipert runs
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
@click.option(
    "--workers", type=int, default=None,
    help="Parallel symptom workers (default: config num_workers).",
)
@click.pass_context
def score(
    ctx: click.Context,
    symptoms: str | None,
    limit: int | None,
    dry_run: bool,
    resume: bool,
    workers: int | None,
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
            max_workers=workers,
        )
    finally:
        runner.close()


@cli.command()
@click.option(
    "--top-n", type=int, default=1000,
    help="Top-N sentences per symptom in output.",
)
@click.option(
    "--run", "run_id", type=str, default=None,
    help="Run ID (1-5) or 'all' to generate specific run(s). "
         "Without this flag, uses legacy silver-label output.",
)
@click.pass_context
def output(ctx: click.Context, top_n: int, run_id: str | None) -> None:
    """Generate TREC-format rankings from scored results."""
    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    if run_id is not None:
        # New run-based output
        import logging

        from hipert.data.trec_writer import write_trec_from_rankings
        from hipert.runs import generate_run, list_runs
        from hipert.runs.registry import SYSTEM_NAMES

        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        rankings_dir = config.output_dir / "rankings"

        if run_id.lower() == "all":
            run_ids = [1, 2, 3, 4, 5]
        else:
            run_ids = [int(r.strip()) for r in run_id.split(",")]

        for rid in run_ids:
            system_name = SYSTEM_NAMES.get(rid, f"HiPerTRun{rid}")
            try:
                rankings = generate_run(rid, config)
                output_path = rankings_dir / f"{system_name}.trec"
                write_trec_from_rankings(
                    rankings, output_path, system_name, top_n=top_n,
                )
                total = sum(len(v) for v in rankings.values())
                click.echo(
                    f"  Run {rid} ({system_name}): {total} lines -> {output_path}"
                )
            except Exception as e:
                click.echo(f"  Run {rid}: SKIPPED ({e})", err=True)

        click.echo(f"\nRankings written to {rankings_dir}")
    else:
        # Legacy output from silver labels
        from hipert.pipeline.runner import PipelineRunner
        runner = PipelineRunner(config)

        try:
            runner.run_output(top_n=top_n)
            click.echo(f"Rankings written to {config.output_dir / 'rankings'}")
        finally:
            runner.close()


@cli.command()
def runs() -> None:
    """List available submission runs and their status."""
    from hipert.runs import list_runs

    available_runs = list_runs()

    click.echo("\neRisk 2026 Task 3 — Submission Runs:\n")
    for r in available_runs:
        status = "READY" if r["available"] else "STUB"
        click.echo(
            f"  Run {r['id']}: [{status:5s}] {r['system_name']}"
            f"\n         {r['description']}"
        )
    click.echo()


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


@cli.command("train")
@click.option(
    "--stage", type=click.Choice(["a", "b", "ab"]), default="ab",
    help="Training stage: a (depression), b (ADHD), ab (both).",
)
@click.option(
    "--backbone", type=click.Choice(["mpnet", "mental-roberta", "clinical-bert", "all"]),
    default="mpnet",
    help="Backbone model to train (default: mpnet).",
)
@click.option("--epochs", type=int, default=None, help="Override max epochs.")
@click.option("--batch-size", type=int, default=32, help="Training batch size.")
@click.option("--lr", type=float, default=2e-5, help="Learning rate.")
@click.option(
    "--resume-from", type=str, default=None,
    help="Path to checkpoint to resume from.",
)
@click.pass_context
def train(
    ctx: click.Context,
    stage: str,
    backbone: str,
    epochs: int | None,
    batch_size: int,
    lr: float,
    resume_from: str | None,
) -> None:
    """Train encoder models (Stage A: depression, Stage B: ADHD)."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    checkpoint_dir = config.output_dir / "training_checkpoints"

    backbones = (
        ["mpnet", "mental-roberta", "clinical-bert"]
        if backbone == "all" else [backbone]
    )

    resume_path = Path(resume_from) if resume_from else None

    for bb in backbones:
        click.echo(f"\n{'='*60}")
        click.echo(f"Training backbone: {bb}")
        click.echo(f"{'='*60}")

        if stage in ("a", "ab"):
            from hipert.training.stage_a import train_stage_a

            if config.bdisen_dir is None:
                click.echo("ERROR: bdisen_dir not configured", err=True)
                return

            stage_a_ckpt = train_stage_a(
                bdisen_dir=config.bdisen_dir,
                erisk2025_dir=config.erisk2025_dir,
                erisk2025_trec_dir=config.erisk2025_trec_dir,
                backbone_name=bb,
                checkpoint_dir=checkpoint_dir,
                max_epochs_a1=epochs or 10,
                batch_size=batch_size,
                learning_rate=lr,
                resume_from=resume_path,
            )
            click.echo(f"  Stage A best: {stage_a_ckpt}")

        if stage in ("b", "ab"):
            from hipert.training.stage_b import train_stage_b

            silver_dir = config.output_dir / "silver_labels"
            if stage == "b":
                # Need explicit Stage A checkpoint
                if resume_from:
                    stage_a_ckpt = Path(resume_from)
                else:
                    stage_a_ckpt = (
                        checkpoint_dir / "stage_a2" / bb
                        / f"stage_a2_{bb}_best.pt"
                    )
                    if not stage_a_ckpt.exists():
                        stage_a_ckpt = (
                            checkpoint_dir / "stage_a1" / bb
                            / f"stage_a1_{bb}_best.pt"
                        )
            # stage_a_ckpt set from Stage A run above when stage=="ab"

            stage_b_ckpt = train_stage_b(
                silver_labels_dir=silver_dir,
                stage_a_checkpoint=stage_a_ckpt,
                backbone_name=bb,
                checkpoint_dir=checkpoint_dir,
                max_epochs=epochs or 15,
                batch_size=batch_size,
                learning_rate=lr * 0.5,
            )
            click.echo(f"  Stage B best: {stage_b_ckpt}")

    click.echo("\nTraining complete!")


@cli.command("infer")
@click.option(
    "--backbone", type=click.Choice(["mpnet", "mental-roberta", "clinical-bert", "all"]),
    default="all",
    help="Backbone model(s) for inference.",
)
@click.option(
    "--stage", type=click.Choice(["stage_a", "stage_b"]), default="stage_b",
    help="Which training stage checkpoint to use.",
)
@click.option("--top-n", type=int, default=1000, help="Top-N per symptom.")
@click.option(
    "--output-subdir", default="encoder_scores",
    help="Subdirectory under output/ for scores.",
)
@click.pass_context
def infer(
    ctx: click.Context,
    backbone: str,
    stage: str,
    top_n: int,
    output_subdir: str,
) -> None:
    """Run encoder inference to produce scored rankings."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])
    checkpoint_dir = config.output_dir / "training_checkpoints"
    candidates_dir = config.output_dir / "candidates"
    output_dir = config.output_dir / output_subdir

    if backbone == "all":
        from hipert.training.inference import run_ensemble_inference

        run_ensemble_inference(
            checkpoint_dir=checkpoint_dir,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            stage=stage,
            top_n=top_n,
        )
    else:
        from hipert.training.inference import run_inference

        run_inference(
            checkpoint_dir=checkpoint_dir,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            backbone_name=backbone,
            stage=stage,
            top_n=top_n,
        )

    click.echo(f"Encoder scores written to {output_dir}")


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


# ---------------------------------------------------------------------------
# v2 cross-encoder commands
# ---------------------------------------------------------------------------


@cli.command("extract-v2")
@click.pass_context
def extract_v2(ctx: click.Context) -> None:
    """Extract training data from LLM cascade outputs for v2 cross-encoder."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    from hipert.training.extract_training_data import (
        extract_training_data,
        save_training_data,
    )

    data = extract_training_data(
        silver_labels_dir=config.output_dir / "silver_labels",
        candidates_dir=config.output_dir / "candidates",
    )

    output_path = config.output_dir / "training_v2" / "training_data.jsonl"
    save_training_data(data, output_path)

    click.echo(f"Extracted {len(data)} training examples to {output_path}")


@cli.command("train-v2")
@click.option(
    "--head-type", type=click.Choice(["coral", "listmle", "both"]),
    default="coral", help="Loss function (default: coral).",
)
@click.option(
    "--backbone", type=click.Choice(["mpnet", "mental-roberta", "clinical-bert", "all"]),
    default="mpnet", help="Backbone model (default: mpnet).",
)
@click.option("--epochs", type=int, default=20, help="Max epochs per fold.")
@click.option("--batch-size", type=int, default=64, help="Training batch size.")
@click.option("--lr", type=float, default=2e-5, help="Learning rate.")
@click.option("--folds", type=int, default=5, help="Number of CV folds (1-5).")
@click.option(
    "--threshold-weights", type=str, default="1.0,1.5,2.0",
    help="CORAL threshold weights (comma-separated).",
)
@click.option(
    "--use-confidence/--no-confidence", default=True,
    help="Use confidence weighting for CORAL.",
)
@click.pass_context
def train_v2(
    ctx: click.Context,
    head_type: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    folds: int,
    threshold_weights: str,
    use_confidence: bool,
) -> None:
    """Train cross-encoder v2 with CORAL or ListMLE loss."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    from hipert.training.extract_training_data import load_training_data
    from hipert.training.trainer_v2 import TrainerV2, TrainerV2Config, select_best_variant

    # Load training data
    data_path = config.output_dir / "training_v2" / "training_data.jsonl"
    if not data_path.exists():
        click.echo(f"ERROR: Training data not found at {data_path}", err=True)
        click.echo("Run 'hipert extract-v2' first.", err=True)
        sys.exit(1)

    all_data = load_training_data(data_path)
    click.echo(f"Loaded {len(all_data)} training examples")

    # Parse threshold weights
    tw = [float(x) for x in threshold_weights.split(",")]

    head_types = ["coral", "listmle"] if head_type == "both" else [head_type]
    backbones = (
        ["mpnet", "mental-roberta", "clinical-bert"]
        if backbone == "all" else [backbone]
    )

    summaries: dict[str, dict] = {}

    for ht in head_types:
        for bb in backbones:
            click.echo(f"\n{'='*60}")
            click.echo(f"Training: {ht} / {bb}")
            click.echo(f"{'='*60}")

            trainer_config = TrainerV2Config(
                backbone_name=bb,
                head_type=ht,
                learning_rate=lr if ht == "coral" else lr * 0.5,
                max_epochs=epochs if ht == "coral" else epochs + 10,
                batch_size=batch_size,
                threshold_weights=tw if ht == "coral" else None,
                use_confidence_weighting=use_confidence and ht == "coral",
                checkpoint_dir=config.output_dir / "training_v2",
                num_folds=folds,
                patience=3 if ht == "coral" else 5,
            )

            trainer = TrainerV2(trainer_config)
            summary = trainer.train_all_folds(all_data)
            summaries[f"{ht}/{bb}"] = summary

            click.echo(
                f"  Result: NDCG@10={summary['mean_ndcg@10']:.4f}±{summary['std_ndcg@10']:.4f} "
                f"P@10={summary['mean_p@10']:.4f} CV={summary['mean_cv']:.4f}"
            )

    # If both variants trained, select best
    if head_type == "both" and len(backbones) == 1:
        bb = backbones[0]
        coral_s = summaries.get(f"coral/{bb}", {})
        listmle_s = summaries.get(f"listmle/{bb}", {})
        if coral_s and listmle_s:
            best = select_best_variant(coral_s, listmle_s)
            click.echo(f"\nSelected variant: {best}")

    click.echo("\nv2 training complete!")


@cli.command("infer-v2")
@click.option(
    "--head-type", type=click.Choice(["coral", "listmle"]),
    default="coral", help="Which head type to use.",
)
@click.option(
    "--backbone", type=click.Choice(["mpnet", "mental-roberta", "clinical-bert", "all"]),
    default="all", help="Backbone model(s) for inference.",
)
@click.option("--folds", type=int, default=5, help="Number of folds to average.")
@click.option("--top-n", type=int, default=1000, help="Top-N per symptom.")
@click.option("--batch-size", type=int, default=128, help="Inference batch size.")
@click.pass_context
def infer_v2(
    ctx: click.Context,
    head_type: str,
    backbone: str,
    folds: int,
    top_n: int,
    batch_size: int,
) -> None:
    """Run cross-encoder v2 inference to produce scored rankings."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    checkpoint_dir = config.output_dir / "training_v2"
    candidates_dir = config.output_dir / "candidates"
    output_dir = config.output_dir / "encoder_scores_v2"

    if backbone == "all":
        from hipert.training.inference_v2 import run_v2_ensemble_inference

        run_v2_ensemble_inference(
            checkpoint_dir=checkpoint_dir,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            head_type=head_type,
            num_folds=folds,
            batch_size=batch_size,
            top_n=top_n,
        )
    else:
        from hipert.training.inference_v2 import run_v2_inference

        run_v2_inference(
            checkpoint_dir=checkpoint_dir,
            candidates_dir=candidates_dir,
            output_dir=output_dir,
            head_type=head_type,
            backbone_name=backbone,
            num_folds=folds,
            batch_size=batch_size,
            top_n=top_n,
        )

    click.echo(f"v2 encoder scores written to {output_dir}")


@cli.command("diagnose-v2")
@click.option(
    "--head-type", type=click.Choice(["coral", "listmle"]),
    default="coral", help="Which head type to diagnose.",
)
@click.option(
    "--backbone", type=click.Choice(["mpnet", "mental-roberta", "clinical-bert"]),
    default="mpnet", help="Backbone model to diagnose.",
)
@click.option("--fold", type=int, default=1, help="Which fold checkpoint to use.")
@click.pass_context
def diagnose_v2(
    ctx: click.Context,
    head_type: str,
    backbone: str,
    fold: int,
) -> None:
    """Run score spread diagnostic on a trained v2 model."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(ctx.obj["config_path"], ctx.obj["symptoms_path"])

    from hipert.training.cross_encoder import CrossEncoderReranker
    from hipert.training.extract_training_data import load_training_data
    from hipert.training.trainer_v2 import diagnose_score_spread

    # Load model
    ckpt_path = (
        config.output_dir / "training_v2" / head_type / backbone
        / f"fold_{fold}" / "best.pt"
    )
    if not ckpt_path.exists():
        click.echo(f"ERROR: Checkpoint not found: {ckpt_path}", err=True)
        sys.exit(1)

    model = CrossEncoderReranker.load_checkpoint(ckpt_path)

    # Load data
    data_path = config.output_dir / "training_v2" / "training_data.jsonl"
    data = load_training_data(data_path)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    report = diagnose_score_spread(model, data, model.tokenizer, device=device)

    if report["healthy"]:
        click.echo(f"\nDiagnosis: HEALTHY (mean CV = {report['mean_cv']:.4f})")
    else:
        click.echo(
            f"\nDiagnosis: COLLAPSED (mean CV = {report['mean_cv']:.4f}) "
            f"— {report['n_collapsed']}/18 symptoms collapsed"
        )


if __name__ == "__main__":
    cli()
