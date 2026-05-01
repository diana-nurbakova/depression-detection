"""Combined server runner for MentalRiskES Tasks 1 and 2.

Orchestrates the required round protocol:
  Task 1 GET → Task 1 POSTs (3 runs) → Task 2 GET → Task 2 POSTs (3 runs)
before the server advances to the next round.

Entry point: ``mentalriskes-server``
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _setup_logging(level: str, log_dir: Path) -> None:
    """Configure logging to console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "combined_server.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _save_server_response(data: dict, task: str, round_number: int, log_dir: Path) -> None:
    """Save raw server GET response for future analysis."""
    path = log_dir / f"{task}_round_{round_number}_server_response.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "round": round_number,
                "n_sessions": len(data) if isinstance(data, dict) else 0,
                "data": data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved %s round %d server response to %s", task, round_number, path)


def _save_round_log(
    task: str,
    round_number: int,
    predictions_per_run: list,
    elapsed_ms: float,
    log_dir: Path,
) -> None:
    """Save per-round decision log (predictions submitted per run)."""
    path = log_dir / f"{task}_round_{round_number}_decisions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "round": round_number,
                "elapsed_ms": elapsed_ms,
                "runs": [
                    {"run_index": i, "predictions": preds}
                    for i, preds in enumerate(predictions_per_run)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _create_together_fallback():
    """Create a Together AI LLM client for fallback, or None if not configured."""
    from .llm_client import LLMClient

    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        logger.warning("TOGETHER_API_KEY not set — fallback disabled")
        return None

    base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    return LLMClient(
        provider="together",
        base_url=base_url,
        api_key=api_key,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.1,
        max_tokens=4096,
        timeout=180,
        rate_limit_delay=0.5,
    )


def _log_task2_decision(
    log_dir: Path,
    run_name: str,
    session_id: str,
    round_id: int,
    options: dict,
    result: object,
) -> None:
    """Append a Task 2 per-round decision to a JSONL log (one file per run)."""
    path = log_dir / f"task2_decisions_{run_name}.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "round": round_id,
        "chosen_option": result.chosen_option,
        "primary_tag": getattr(result, "primary_tag", ""),
        "reasoning": getattr(result, "reasoning", ""),
        "options": options,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@click.command()
@click.option("--task1-config", default="config/mentalriskes.yaml",
              help="Path to Task 1 config YAML.")
@click.option("--task2-config", default="config/mentalriskes_task2.yaml",
              help="Path to Task 2 config YAML.")
@click.option("--task2-runs", default="run0,run1,run2",
              help="Comma-separated Task 2 run config names from YAML.")
@click.option("--max-rounds", default=200,
              help="Safety cap on rounds. Loop normally exits when the server returns "
                   "no more messages; this cap only fires if the server keeps serving "
                   "data far beyond expectations. NOT a target round count.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--log-dir", default="output/mentalriskes/server_logs",
              help="Directory for combined server logs.")
@click.option("--fallback/--no-fallback", default=True,
              help="Enable Together AI fallback if HuggingFace fails.")
def main(
    task1_config: str,
    task2_config: str,
    task2_runs: str,
    max_rounds: int,
    verbose: bool,
    log_dir: str,
    fallback: bool,
) -> None:
    """Run Task 1 and Task 2 against MentalRiskES server in lockstep.

    Each round: Task1 GET → Task1 POST×3 → Task2 GET → Task2 POST×3.
    """
    log_path = Path(log_dir)
    _setup_logging("DEBUG" if verbose else "INFO", log_path)

    # ── Task 1 setup ──────────────────────────────────────────────
    from .config import load_config as load_config_t1
    from .task1.pipeline import Pipeline as Task1Pipeline

    t1_cfg = load_config_t1(task1_config)
    if not t1_cfg.server.base_url or not t1_cfg.server.token:
        click.echo("Error: MENTALRISKES_BASE_URL and MENTALRISKES_TOKEN env vars required.", err=True)
        sys.exit(1)

    t1_pipeline = Task1Pipeline(t1_cfg)
    t1_server = t1_pipeline.create_server_client()
    t1_state = t1_pipeline.init_server_state()

    # Create fallback client (shared by both tasks)
    _fallback_client = _create_together_fallback() if fallback else None

    # Attach fallback to Task 1 LLM clients
    if _fallback_client:
        for client in t1_state["run_clients"].values():
            client.with_fallback(_fallback_client)
        click.echo("Fallback: Together AI (meta-llama/Llama-3.3-70B-Instruct-Turbo)")

    click.echo(f"Task 1: {len(t1_cfg.runs)} runs configured")

    # ── Task 2 setup ──────────────────────────────────────────────
    import yaml as _yaml

    from .llm_client import create_llm_client
    from .task2.data import parse_server_round
    from .task2.pipeline import PipelineConfig, Task2Pipeline
    from .task2.server import Task2Client

    with open(task2_config, encoding="utf-8") as f:
        t2_raw = _yaml.safe_load(f)

    # Build LLM config for Task 2
    from .config import LLMConfig

    t2_llm_section = t2_raw.get("llm", {})
    t2_llm_config = LLMConfig(
        provider=t2_llm_section.get("provider", "huggingface"),
        base_url=t2_llm_section.get("base_url", os.getenv("HF_BASE_URL", "")),
        api_key=t2_llm_section.get("api_key", os.getenv("HF_TOKEN", "")),
        model=t2_llm_section.get("model", "meta-llama/Llama-3.3-70B-Instruct"),
        temperature=t2_llm_section.get("temperature", 0.1),
        max_tokens=t2_llm_section.get("max_tokens", 4096),
        timeout=t2_llm_section.get("timeout", 300),
    )

    t2_server_cfg = t2_raw.get("server", {})
    t2_client = Task2Client(
        base_url=t2_server_cfg.get("base_url", os.getenv("MENTALRISKES_SERVER_URL", "")),
        token=t2_server_cfg.get("token", os.getenv("MENTALRISKES_TOKEN", "")),
        use_trial=t2_server_cfg.get("use_trial", False),
        retries=t2_server_cfg.get("retries", 5),
        backoff=t2_server_cfg.get("backoff", 0.1),
    )

    t2_data_cfg = t2_raw.get("data", {})
    t2_output_dir = Path(t2_data_cfg.get("output_dir", "output/mentalriskes_task2"))

    run_names = [r.strip() for r in task2_runs.split(",")]
    t2_runs_cfg = t2_raw.get("runs", {})

    t2_pipelines: list[Task2Pipeline] = []
    for name in run_names:
        run_cfg = t2_runs_cfg.get(name, {})
        pcfg = PipelineConfig(
            name=name,
            model=run_cfg.get("model", t2_llm_config.model),
            framing=run_cfg.get("framing", "FUNC"),
            pipeline=run_cfg.get("pipeline", "B"),
            lang=run_cfg.get("lang", "es"),
            lookback_window=run_cfg.get("lookback_window", 3),
            permutation_voting=run_cfg.get("permutation_voting", False),
            calibration=run_cfg.get("calibration", False),
        )
        llm = create_llm_client(t2_llm_config, model_override=pcfg.model)
        # Attach fallback to Task 2 LLM clients
        if fallback and _fallback_client:
            llm.with_fallback(_fallback_client)
        t2_pipelines.append(Task2Pipeline(llm=llm, config=pcfg))

    click.echo(f"Task 2: {len(t2_pipelines)} runs configured ({', '.join(run_names)})")
    click.echo(f"Server:        {t1_cfg.server.base_url}")
    click.echo(f"use_trial:     T1={t1_cfg.server.use_trial}  T2={t2_client.use_trial}")
    click.echo(f"max_rounds:    {max_rounds}  (safety cap; loop exits on empty server response)")
    click.echo(f"Logs:          {log_path}")
    click.echo("")

    # ── Main round loop ──────────────────────────────────────────
    # Termination is data-driven: stop when both Task 1 and Task 2 GETs
    # return empty. `max_rounds` is a safety cap to avoid infinite loops if
    # the server misbehaves; hitting it is a hard error, not a normal exit.
    last_t1_round_seen = 0
    last_t2_round_seen = 0
    hit_safety_cap = False

    for round_num in range(1, max_rounds + 1):
        round_t0 = time.monotonic()
        click.echo(f"{'='*60}")
        click.echo(f"ROUND {round_num}")
        click.echo(f"{'='*60}")

        # ── Step 1: Task 1 GET ────────────────────────────────────
        t1_messages = t1_server.get_messages()
        if not t1_messages:
            click.echo("Task 1: server returned no messages -> end of test.")
            break

        t1_round = next(iter(t1_messages.values()))["round"]
        last_t1_round_seen = t1_round
        click.echo(f"Task 1: received round {t1_round} ({len(t1_messages)} sessions)")
        _save_server_response(t1_messages, "task1", t1_round, log_path)

        # ── Step 2: Task 1 process + POST ─────────────────────────
        t1_t0 = time.monotonic()
        t1_pipeline.process_server_round(t1_messages, t1_server, t1_state)
        t1_elapsed = (time.monotonic() - t1_t0) * 1000
        click.echo(f"Task 1: round {t1_round} submitted ({t1_elapsed:.0f}ms)")

        # ── Step 3: Task 2 GET ────────────────────────────────────
        t2_data = t2_client.get_round()
        if not t2_data:
            click.echo("Task 2: server returned no messages -> end of test.")
            break

        # Detect round number from first session
        first_session = next(iter(t2_data.values()))
        t2_round = first_session.get("round", round_num)
        last_t2_round_seen = t2_round
        click.echo(f"Task 2: received round {t2_round} ({len(t2_data)} sessions)")
        _save_server_response(t2_data, "task2", t2_round, log_path)

        # ── Step 4: Task 2 process + POST ─────────────────────────
        t2_t0 = time.monotonic()
        t2_predictions_per_run: list[list[dict]] = [[] for _ in t2_pipelines]

        for session_id, session_data in t2_data.items():
            record = parse_server_round(session_data)

            for run_idx, pipe in enumerate(t2_pipelines):
                result = pipe.process_single_round(
                    record.round_id, record.patient_message, record.options,
                )
                t2_predictions_per_run[run_idx].append({
                    "id": session_id,
                    "round": record.round_id,
                    "prediction": result.chosen_option,
                })

                # Log detailed decision for this run/session
                _log_task2_decision(
                    log_dir=log_path,
                    run_name=run_names[run_idx],
                    session_id=session_id,
                    round_id=record.round_id,
                    options=record.options,
                    result=result,
                )

        emissions = {}
        t2_client.submit_all_runs(
            t2_predictions_per_run, emissions,
            save_dir=t2_output_dir / "server_submissions",
            round_number=t2_round,
        )
        t2_elapsed = (time.monotonic() - t2_t0) * 1000

        # Save Task 2 decision log
        _save_round_log("task2", t2_round, t2_predictions_per_run, t2_elapsed, log_path)

        round_elapsed = (time.monotonic() - round_t0) * 1000
        click.echo(f"Task 2: round {t2_round} submitted ({t2_elapsed:.0f}ms)")
        click.echo(f"Round {round_num} complete ({round_elapsed:.0f}ms total)\n")
    else:
        # `for ... else` runs when the loop exhausts the iterator without `break`,
        # i.e. we hit max_rounds without the server signalling end-of-test.
        hit_safety_cap = True

    click.echo("=" * 60)
    click.echo(f"Last Task 1 round seen: {last_t1_round_seen}")
    click.echo(f"Last Task 2 round seen: {last_t2_round_seen}")
    if hit_safety_cap:
        click.echo(
            f"WARNING: hit --max-rounds safety cap of {max_rounds} but the server "
            f"was still serving messages. The submission is INCOMPLETE. Re-launch "
            f"with --max-rounds {max_rounds * 2} (or more) to continue.",
            err=True,
        )
        sys.exit(2)
    click.echo("All rounds processed (server signalled end-of-test).")


if __name__ == "__main__":
    main()
