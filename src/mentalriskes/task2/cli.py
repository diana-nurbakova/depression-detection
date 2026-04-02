"""CLI for MentalRiskES Task 2: Therapist Response Selection."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mentalriskes.task2")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        click.echo(f"Config file not found: {path}", err=True)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_llm_config(cfg: dict):
    from ..config import LLMConfig
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")

    # Resolve API key based on provider
    if provider == "huggingface":
        api_key = llm_cfg.get("api_key", os.getenv("HF_TOKEN", ""))
        base_url = ""  # not used by HF client
    elif provider == "together":
        api_key = llm_cfg.get("api_key", os.getenv("TOGETHER_API_KEY", ""))
        base_url = llm_cfg.get("base_url", os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"))
    else:
        api_key = llm_cfg.get("api_key", os.getenv("OLLAMA_API_KEY", os.getenv("OPENAI_API_KEY", "")))
        base_url = llm_cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=llm_cfg.get("model", "llama3.3:70b"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=llm_cfg.get("timeout", 300),
    )


def _make_fallback_config(cfg: dict):
    """Build a TogetherAI fallback LLM config from env vars."""
    from ..config import LLMConfig
    together_key = os.getenv("TOGETHER_API_KEY", "")
    together_url = os.getenv("TOGETHER_BASE_URL", "")
    if not together_key or not together_url:
        return None
    llm_cfg = cfg.get("llm", {})
    return LLMConfig(
        provider="openai",
        base_url=together_url,
        api_key=together_key,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=300,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--config", "-c", default="config/mentalriskes_task2.yaml", help="Config file path.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: str) -> None:
    """MentalRiskES 2026 Task 2: Therapist Response Selection."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--run", "-r", default=None, help="Run config name (from YAML). If not set, runs default.")
@click.option("--framing", type=click.Choice(["FUNC", "HYB", "TOM-B", "TOM-C"]), default=None)
@click.option("--pipeline", type=click.Choice(["A", "B", "B+"]), default=None)
@click.option("--lang", type=click.Choice(["es", "en"]), default=None)
@click.option("--window", "-w", type=int, default=None, help="Lookback window size.")
@click.option("--perm", is_flag=True, help="Enable permutation voting.")
@click.pass_context
def trial(ctx: click.Context, run: str | None, framing: str | None, pipeline: str | None,
          lang: str | None, window: int | None, perm: bool) -> None:
    """Run pipeline on local trial data."""
    from ..llm_client import create_llm_client
    from .data import TRIAL_INFERRED_LABELS
    from .evaluation import evaluate_result, format_evaluation_report
    from .pipeline import PipelineConfig, Task2Pipeline

    cfg = _load_config(ctx.obj["config_path"])
    llm_config = _make_llm_config(cfg)

    # Build pipeline config from CLI overrides or YAML run config
    data_cfg = cfg.get("data", {})
    trial_dir = Path(data_cfg.get("trial_dir", "data/MentalRiskES-2026/task2_trial/data"))
    output_dir = Path(data_cfg.get("output_dir", "output/mentalriskes_task2"))

    if run and "runs" in cfg:
        run_cfg = cfg["runs"].get(run, {})
    else:
        run_cfg = cfg.get("default_run", {})

    pcfg = PipelineConfig(
        name=run or "trial",
        model=run_cfg.get("model", llm_config.model),
        framing=framing or run_cfg.get("framing", "FUNC"),
        pipeline=pipeline or run_cfg.get("pipeline", "B"),
        lang=lang or run_cfg.get("lang", "es"),
        lookback_window=window if window is not None else run_cfg.get("lookback_window", 3),
        permutation_voting=perm or run_cfg.get("permutation_voting", False),
    )

    click.echo(f"Running trial: {pcfg.config_id}")
    llm = create_llm_client(llm_config, model_override=pcfg.model)
    pipe = Task2Pipeline(llm=llm, config=pcfg)
    result = pipe.run_trial(trial_dir)
    result_path = pipe.save_result(result, output_dir)

    # Evaluate
    eval_result = evaluate_result(result_path, TRIAL_INFERRED_LABELS)
    report = format_evaluation_report(eval_result)
    click.echo(report)


@cli.command()
@click.option("--configs", "-n", default=None, help="Comma-separated config indices (1-indexed) or 'all'.")
@click.option("--provider", type=click.Choice(["ollama", "together", "huggingface"]), default=None,
              help="Override LLM provider.")
@click.pass_context
def ablation(ctx: click.Context, configs: str | None, provider: str | None) -> None:
    """Run ablation study across multiple configurations."""
    from .ablation import format_ablation_summary, get_ablation_configs, run_ablation

    cfg = _load_config(ctx.obj["config_path"])

    if provider == "together":
        llm_config = _make_fallback_config(cfg)
        if not llm_config:
            click.echo("TOGETHER_API_KEY / TOGETHER_BASE_URL not set in .env", err=True)
            sys.exit(1)
    elif provider:
        cfg.setdefault("llm", {})["provider"] = provider
        llm_config = _make_llm_config(cfg)
    else:
        llm_config = _make_llm_config(cfg)

    click.echo(f"Primary LLM: {llm_config.provider} / {llm_config.model}")

    data_cfg = cfg.get("data", {})
    trial_dir = Path(data_cfg.get("trial_dir", "data/MentalRiskES-2026/task2_trial/data"))
    output_dir = Path(data_cfg.get("output_dir", "output/mentalriskes_task2")) / "ablation"

    all_configs = get_ablation_configs(
        local_model=cfg.get("ablation", {}).get("local_model", "llama3.3:70b"),
        api_model=cfg.get("ablation", {}).get("api_model", "claude-sonnet-4-20250514"),
    )

    if configs and configs != "all":
        indices = [int(i) - 1 for i in configs.split(",")]
        selected = [all_configs[i] for i in indices if 0 <= i < len(all_configs)]
    else:
        selected = all_configs

    # Build fallback config if not already using together as primary
    fallback_config = None
    if provider != "together":
        fallback_config = _make_fallback_config(cfg)
        if fallback_config:
            click.echo(f"Fallback LLM: {fallback_config.provider} / {fallback_config.model}")

    click.echo(f"Running ablation: {len(selected)} configurations")
    results = run_ablation(selected, trial_dir, output_dir, llm_config, fallback_config=fallback_config)
    summary = format_ablation_summary(results)
    click.echo(summary)


@cli.command()
@click.option("--run-configs", "-r", required=True, help="Comma-separated run config names from YAML.")
@click.pass_context
def server(ctx: click.Context, run_configs: str) -> None:
    """Run pipeline against the competition server (GET/POST loop)."""
    from ..llm_client import create_llm_client
    from .data import parse_server_round
    from .pipeline import PipelineConfig, Task2Pipeline
    from .server import Task2Client

    cfg = _load_config(ctx.obj["config_path"])
    llm_config = _make_llm_config(cfg)

    server_cfg = cfg.get("server", {})
    client = Task2Client(
        base_url=server_cfg.get("base_url", os.getenv("MENTALRISKES_SERVER_URL", "")),
        token=server_cfg.get("token", os.getenv("MENTALRISKES_TOKEN", "")),
        use_trial=server_cfg.get("use_trial", False),
        retries=server_cfg.get("retries", 5),
        backoff=server_cfg.get("backoff", 0.1),
    )

    run_names = [r.strip() for r in run_configs.split(",")]
    runs_cfg = cfg.get("runs", {})
    data_cfg = cfg.get("data", {})
    output_dir = Path(data_cfg.get("output_dir", "output/mentalriskes_task2"))

    # Build pipelines for each run
    pipelines: list[Task2Pipeline] = []
    for name in run_names:
        run_cfg = runs_cfg.get(name, {})
        pcfg = PipelineConfig(
            name=name,
            model=run_cfg.get("model", llm_config.model),
            framing=run_cfg.get("framing", "FUNC"),
            pipeline=run_cfg.get("pipeline", "B"),
            lang=run_cfg.get("lang", "es"),
            lookback_window=run_cfg.get("lookback_window", 3),
            permutation_voting=run_cfg.get("permutation_voting", False),
            calibration=run_cfg.get("calibration", False),
        )
        llm = create_llm_client(llm_config, model_override=pcfg.model)
        pipelines.append(Task2Pipeline(llm=llm, config=pcfg))

    click.echo(f"Server mode: {len(pipelines)} runs, trial={client.use_trial}")

    # GET/POST loop
    max_rounds = 30
    for round_num in range(1, max_rounds + 1):
        data = client.get_round()
        if not data:
            click.echo("No more data from server. Done.")
            break

        # Process each session
        predictions_per_run: list[list[dict]] = [[] for _ in pipelines]

        for session_id, session_data in data.items():
            record = parse_server_round(session_data)

            for run_idx, pipe in enumerate(pipelines):
                result = pipe.selector.process_round(
                    record.round_id, record.patient_message, record.options
                )
                predictions_per_run[run_idx].append({
                    "id": session_id,
                    "round": record.round_id,
                    "prediction": result.chosen_option,
                })

        # Submit all runs
        emissions = {}  # TODO: CodeCarbon integration
        client.submit_all_runs(
            predictions_per_run, emissions,
            save_dir=output_dir / "server_submissions",
            round_number=round_num,
        )
        click.echo(f"Round {round_num}: submitted {len(pipelines)} runs")


@cli.command("simulated-ablation")
@click.option("--configs", "-n", default=None, help="Comma-separated config indices (1-indexed) or 'all'.")
@click.option("--provider", type=click.Choice(["ollama", "together", "huggingface"]), default=None,
              help="Override LLM provider.")
@click.option("--data-dir", "-d", default=None,
              help="Path to simulated data directory. Overrides config value.")
@click.option("--force", "-f", is_flag=True, help="Force re-run all sessions, ignoring cached results.")
@click.pass_context
def simulated_ablation(ctx: click.Context, configs: str | None, provider: str | None,
                       data_dir: str | None, force: bool) -> None:
    """Run ablation study on simulated persona sessions."""
    from .ablation import format_multi_session_summary, get_ablation_configs, run_multi_session_ablation

    cfg = _load_config(ctx.obj["config_path"])

    if provider == "together":
        llm_config = _make_fallback_config(cfg)
        if not llm_config:
            click.echo("TOGETHER_API_KEY / TOGETHER_BASE_URL not set in .env", err=True)
            sys.exit(1)
    elif provider:
        # Override provider in config before building LLM config
        cfg.setdefault("llm", {})["provider"] = provider
        llm_config = _make_llm_config(cfg)
    else:
        llm_config = _make_llm_config(cfg)

    click.echo(f"Primary LLM: {llm_config.provider} / {llm_config.model}")

    data_cfg = cfg.get("data", {})
    simulated_dir = Path(data_dir) if data_dir else Path(
        data_cfg.get("simulated_dir", "output/mentalriskes/data_prep/simulated/task2")
    )
    output_dir = Path(data_cfg.get("output_dir", "output/mentalriskes_task2")) / "simulated_ablation"

    if not simulated_dir.exists():
        click.echo(f"Simulated data directory not found: {simulated_dir}", err=True)
        sys.exit(1)

    all_configs = get_ablation_configs(
        local_model=cfg.get("ablation", {}).get("local_model", "llama3.3:70b"),
        api_model=cfg.get("ablation", {}).get("api_model", "claude-sonnet-4-20250514"),
    )

    if configs and configs != "all":
        indices = [int(i) - 1 for i in configs.split(",")]
        selected = [all_configs[i] for i in indices if 0 <= i < len(all_configs)]
    else:
        selected = all_configs

    # Build fallback config if not already using together as primary
    fallback_config = None
    if provider != "together":
        fallback_config = _make_fallback_config(cfg)
        if fallback_config:
            click.echo(f"Fallback LLM: {fallback_config.provider} / {fallback_config.model}")

    click.echo(f"Running simulated ablation: {len(selected)} configs on {simulated_dir}")
    if force:
        click.echo("Force mode: all sessions will be re-run")
    results = run_multi_session_ablation(selected, simulated_dir, output_dir, llm_config, fallback_config, force=force)
    summary = format_multi_session_summary(results)
    click.echo(summary)


@cli.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.pass_context
def evaluate(ctx: click.Context, result_path: str) -> None:
    """Evaluate a pipeline result JSONL against inferred labels."""
    from .data import TRIAL_INFERRED_LABELS
    from .evaluation import evaluate_result, format_evaluation_report

    eval_result = evaluate_result(Path(result_path), TRIAL_INFERRED_LABELS)
    report = format_evaluation_report(eval_result)
    click.echo(report)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show current configuration and status."""
    cfg = _load_config(ctx.obj["config_path"])

    click.echo("=== MentalRiskES Task 2 Configuration ===")
    click.echo(f"Config: {ctx.obj['config_path']}")

    llm = cfg.get("llm", {})
    click.echo(f"LLM: {llm.get('provider', 'ollama')} / {llm.get('model', 'llama3.3:70b')}")

    data = cfg.get("data", {})
    trial_dir = Path(data.get("trial_dir", "data/MentalRiskES-2026/task2_trial/data"))
    n_rounds = len(list(trial_dir.glob("round_*.json"))) if trial_dir.exists() else 0
    click.echo(f"Trial data: {trial_dir} ({n_rounds} rounds)")

    if "runs" in cfg:
        click.echo(f"Configured runs: {', '.join(cfg['runs'].keys())}")

    from .data import TRIAL_INFERRED_LABELS
    click.echo(f"Inferred labels: {len(TRIAL_INFERRED_LABELS)} rounds (2-18)")
