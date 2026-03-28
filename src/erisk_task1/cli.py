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
@click.option("--config", "config_path", default="config/task1.yaml", help="Config YAML path")
@click.option("--talkdep", default="data/TalkDep", help="Path to TalkDep repo")
@click.option("--configs", default=None, help="Comma-separated ablation configs (default: all A0-A7)")
@click.option("--personas", default=None, help="Comma-separated persona names (e.g., Maria,Noah)")
@click.option("--output", default="runs/ablation", help="Output directory")
@click.option("--save-conversations", is_flag=True, default=False, help="Save transcripts, raw LLM responses, and evidence in result JSONs")
@click.option("--provider", default=None, type=click.Choice(["ollama", "together", "huggingface"]), help="Override LLM provider for assessor/orchestrator/justificator (default: from config)")
@click.option("--log-level", default="INFO", help="Log level")
def ablation(config_path: str, talkdep: str, configs: str | None, personas: str | None, output: str, save_conversations: bool, provider: str | None, log_level: str):
    """Run the ablation study against TalkDep conversations.

    Tests pipeline components incrementally (A0-A7) against 12 personas
    with golden BDI-II scores.

    Example:
      python -m erisk_task1.cli ablation --configs A0,A1,A4
      python -m erisk_task1.cli ablation --personas Maria,Noah,Ethan
      python -m erisk_task1.cli ablation --provider together  # use Together AI (no VPN needed)
      python -m erisk_task1.cli ablation  # runs all A0-A7 on all 12 personas
    """
    log_file = str(Path(output) / "ablation.log")
    setup_logging(log_level, log_file)

    config = load_config(config_path)

    # Override provider for assessor/orchestrator/justificator
    if provider in ("together", "huggingface"):
        model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo" if provider == "together" else "meta-llama/Llama-3.3-70B-Instruct"
        for mc in (config.assessor, config.orchestrator_llm, config.justificator):
            mc.provider = provider
            mc.model = model_name

    config_list = None
    if configs:
        config_list = [c.strip() for c in configs.split(",")]

    persona_list = None
    if personas:
        persona_list = [p.strip() for p in personas.split(",")]

    from .ablation import ABLATION_CONFIGS, run_full_ablation_study
    from .evaluation import format_comparison_table, format_error_analysis

    valid_configs = config_list or list(ABLATION_CONFIGS.keys())
    click.echo(f"Ablation study")
    click.echo(f"  Configs: {valid_configs}")
    click.echo(f"  TalkDep: {talkdep}")
    click.echo(f"  Personas: {persona_list or 'all 12'}")
    click.echo(f"  Assessor: {config.assessor.model} via {config.assessor.provider}")
    click.echo(f"  Output: {output}")
    click.echo()

    results = run_full_ablation_study(
        pipeline_cfg=config,
        talkdep_dir=talkdep,
        configs=config_list,
        personas=persona_list,
        output_dir=output,
        save_conversations=save_conversations,
    )

    # Print comparison table
    click.echo()
    click.echo(format_comparison_table(results))

    # Print error analysis for full pipeline (A4)
    for r in results:
        if r.config_name == "A4_justificator":
            click.echo()
            click.echo(format_error_analysis(r))

    click.echo(f"\nDetailed results saved to: {output}/")


@cli.command("save-talkdep")
@click.option("--talkdep", default="data/TalkDep", help="Path to TalkDep repo root")
@click.option(
    "--output",
    default="data/talkdep_conversations",
    help="Output directory for structured JSONs",
)
@click.option(
    "--no-combined",
    "no_combined",
    is_flag=True,
    default=False,
    help="Skip saving all_sessions.json per persona",
)
@click.option("--log-level", default="INFO", help="Log level")
def save_talkdep(talkdep: str, output: str, no_combined: bool, log_level: str):
    """Export TalkDep conversations as structured JSONs for ToM analysis.

    Saves per-session and combined JSON files for each of the 12 TalkDep personas,
    plus a ground_truth.json with approximate 21-dim BDI-II vectors extracted from
    patient profiles, and a golden_scores.json with total BDI-II scores.

    Output structure:
      {output}/{Name}/session_1.json  ... session_5.json
      {output}/{Name}/all_sessions.json
      {output}/ground_truth.json
      {output}/golden_scores.json

    Example:
      uv run python -m erisk_task1.cli save-talkdep
      uv run python -m erisk_task1.cli save-talkdep --output runs/tom/talkdep_data
    """
    setup_logging(log_level)

    from .evaluation import save_talkdep_conversations

    talkdep_path = Path(talkdep)
    if not talkdep_path.exists():
        click.echo(f"TalkDep directory not found: {talkdep_path}", err=True)
        raise SystemExit(1)

    click.echo(f"TalkDep source: {talkdep_path}")
    click.echo(f"Output dir:     {output}")
    click.echo()

    save_talkdep_conversations(
        talkdep_dir=talkdep,
        output_dir=output,
        combined=not no_combined,
    )

    output_path = Path(output)
    click.echo(f"Done. Files written to {output_path.resolve()}/")
    click.echo(f"  Per persona: session_1..N.json + all_sessions.json")
    click.echo(f"  Shared:      ground_truth.json  (partial BDI-II vectors from profiles)")
    click.echo(f"               golden_scores.json (total BDI-II + band)")
    click.echo()
    click.echo("Next steps:")
    click.echo(
        "  Run assessors per turn on any session JSON with the ToM analysis pipeline."
    )


@cli.command("tom-ablation")
@click.option("--config", "config_path", default="config/task1.yaml", help="Config YAML path")
@click.option("--data", "data_dir", default="data/talkdep_conversations",
              help="Path to saved TalkDep conversations (from save-talkdep)")
@click.option("--personas", default=None,
              help="Comma-separated persona names (e.g., Maria,Noah)")
@click.option("--output", default="runs/tom_ablation", help="Output directory")
@click.option("--assess-every", default=0, type=int,
              help="Run assessors every N persona turns (0 = config default)")
@click.option("--conditions", default=None,
              help="Comma-separated conditions to run (default: tom_off,tom_on). "
                   "Available: tom_off, tom_on, tom_c1, tom_c1c2, "
                   "tom_c1c2_conservative, tom_c1c2_walign")
@click.option("--sessions", default=None,
              help="Comma-separated session numbers to use (e.g., 1,2,3; default: all combined)")
@click.option("--provider", default=None,
              type=click.Choice(["ollama", "together", "huggingface"]),
              help="Override LLM provider for assessor/orchestrator")
@click.option("--log-level", default="INFO", help="Log level")
def tom_ablation(config_path: str, data_dir: str, personas: str | None,
                 output: str, assess_every: int, conditions: str | None,
                 sessions: str | None, provider: str | None, log_level: str):
    """Run ToM ablation study on TalkDep conversations.

    Replays pre-recorded TalkDep conversations through the Orchestrator
    with incremental assessment, comparing different ToM conditions.

    Requires saved TalkDep data (run 'save-talkdep' first).

    Example:
      uv run python -m erisk_task1.cli tom-ablation
      uv run python -m erisk_task1.cli tom-ablation --personas Maria,Noah,Ethan
      uv run python -m erisk_task1.cli tom-ablation --conditions tom_off,tom_c1,tom_c1c2
      uv run python -m erisk_task1.cli tom-ablation --conditions tom_c1c2 --personas Maria,Elena
      uv run python -m erisk_task1.cli tom-ablation --provider together
    """
    log_file = str(Path(output) / "tom_ablation.log")
    setup_logging(log_level, log_file)

    config = load_config(config_path)

    # Override provider
    if provider in ("together", "huggingface"):
        model_name = (
            "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            if provider == "together"
            else "meta-llama/Llama-3.3-70B-Instruct"
        )
        for mc in (config.assessor, config.orchestrator_llm):
            mc.provider = provider
            mc.model = model_name

    persona_list = None
    if personas:
        persona_list = [p.strip() for p in personas.split(",")]

    session_list = None
    if sessions:
        session_list = [int(s.strip()) for s in sessions.split(",")]

    from .evaluation import format_comparison_table
    from .tom_ablation import (
        ABLATION_CONDITIONS,
        format_tom_analysis_table,
        format_tom_comparison,
        run_tom_ablation as _run_tom_ablation,
    )

    condition_list = None
    if conditions:
        condition_list = [c.strip() for c in conditions.split(",")]

    click.echo("ToM Ablation Study")
    click.echo(f"  Data: {data_dir}")
    click.echo(f"  Personas: {persona_list or 'all'}")
    click.echo(f"  Conditions: {condition_list or ['tom_off', 'tom_on']}")
    click.echo(f"  Sessions: {session_list or 'all combined'}")
    click.echo(f"  Assessor: {config.assessor.model} via {config.assessor.provider}")
    click.echo(f"  Assess every: {assess_every or 'config default'} turns")
    click.echo(f"  Output: {output}")
    click.echo(f"  Available conditions: {list(ABLATION_CONDITIONS.keys())}")
    click.echo()

    results = _run_tom_ablation(
        pipeline_cfg=config,
        data_dir=data_dir,
        personas=persona_list,
        output_dir=output,
        assess_every_n=assess_every,
        sessions=session_list,
        conditions=condition_list,
    )

    # Print comparison table if we have 2+ conditions
    if len(results) >= 2:
        result_list = list(results.values())
        click.echo()
        click.echo(format_comparison_table(result_list))
    elif len(results) == 1:
        r = next(iter(results.values()))
        click.echo(f"\n{r.config_name}: DCHR={r.dchr*100:.1f}%, "
                   f"MAD={r.mad:.1f}, ADODL={r.adodl:.3f}")

    # Print ToM analysis
    analysis_path = Path(output) / "tom_analysis.json"
    if analysis_path.exists():
        click.echo()
        click.echo(format_tom_analysis_table(analysis_path))

    click.echo(f"\nDetailed results saved to: {output}/")


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
