"""Multi-session replay for Task 2 on the released test data.

The built-in `mentalriskes-task2 trial` command expects single-session legacy
trial format (one session per round wrapped in a "trial" key). The released
test set is multi-session (round_N.json keyed by session_id with up to 10
sessions per round).

This script:
1. Loads the multi-session test rounds.
2. For each (run, session) pair, instantiates a fresh Task2Pipeline so its
   state tracker resets between sessions, and processes the session's rounds
   in order.
3. Saves predictions per-round per-run in `server_submissions` format
   (`round{N}_run{R}.json`) — the same format the analysis scripts read.

Usage:
    uv run python analysis/MentalRiskES_test/replay_task2_test.py \
        --config config/mentalriskes_task2_test_replay.yaml \
        --runs run0,run1,run2

Each run is processed sequentially in this script; launch parallel processes
(one per --runs value) to fan out across DeepInfra.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

load_dotenv(REPO_ROOT / ".env")

from mentalriskes.config import LLMConfig  # noqa: E402
from mentalriskes.llm_client import create_llm_client  # noqa: E402
from mentalriskes.task2.models import RoundRecord  # noqa: E402
from mentalriskes.task2.pipeline import PipelineConfig, Task2Pipeline  # noqa: E402

logger = logging.getLogger("replay_task2")


def _make_llm_config(cfg: dict) -> LLMConfig:
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "deepinfra")
    if provider == "deepinfra":
        api_key = llm_cfg.get("api_key", os.getenv("DEEPINFRA_API_KEY", ""))
        base_url = llm_cfg.get("base_url", os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"))
    elif provider == "huggingface":
        api_key = llm_cfg.get("api_key", os.getenv("HF_TOKEN", ""))
        base_url = ""
    elif provider == "together":
        api_key = llm_cfg.get("api_key", os.getenv("TOGETHER_API_KEY", ""))
        base_url = llm_cfg.get("base_url", os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"))
    else:
        api_key = llm_cfg.get("api_key", os.getenv("OLLAMA_API_KEY", ""))
        base_url = llm_cfg.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=llm_cfg.get("model", "meta-llama/Llama-3.3-70B-Instruct"),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=llm_cfg.get("timeout", 300),
    )


def load_multi_session_rounds(trial_dir: Path) -> dict[str, list[RoundRecord]]:
    """Walk round_*.json and group rounds by session_id."""
    sessions: dict[str, list[RoundRecord]] = {}
    files = sorted(trial_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            data = json.load(fh)
        for sid, payload in data.items():
            sessions.setdefault(sid, []).append(RoundRecord(
                round_id=payload["round"],
                patient_message=payload["patient_input"],
                options={
                    "option_1": payload.get("option_1", ""),
                    "option_2": payload.get("option_2", ""),
                    "option_3": payload.get("option_3", ""),
                },
            ))
    for sid in sessions:
        sessions[sid].sort(key=lambda r: r.round_id)
    logger.info("Loaded %d sessions from %s", len(sessions), trial_dir)
    return sessions


def replay_run(cfg: dict, run_name: str, output_dir: Path) -> None:
    runs_cfg = cfg.get("runs", {})
    if run_name not in runs_cfg:
        raise KeyError(f"Run config '{run_name}' not in YAML; available: {list(runs_cfg)}")
    rc = runs_cfg[run_name]

    llm_config = _make_llm_config(cfg)
    data_cfg = cfg.get("data", {})
    trial_dir = Path(data_cfg.get("trial_dir"))
    sessions = load_multi_session_rounds(trial_dir)

    # Fresh LLM client per run (re-uses TCP pool internally)
    llm = create_llm_client(llm_config, model_override=rc.get("model", llm_config.model))
    pcfg = PipelineConfig(
        name=run_name,
        model=rc.get("model", llm_config.model),
        framing=rc.get("framing", "FUNC"),
        pipeline=rc.get("pipeline", "B"),
        lang=rc.get("lang", "es"),
        lookback_window=rc.get("lookback_window", 3),
        permutation_voting=rc.get("permutation_voting", False),
        calibration=rc.get("calibration", False),
    )

    # Map run_name -> integer index expected by the analysis (run0/run1/run2 -> 0/1/2)
    run_idx = int(run_name.replace("run", ""))
    submissions_dir = output_dir / "server_submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Predictions accumulate per round across sessions
    per_round_preds: dict[int, list[dict]] = {}

    t0 = time.monotonic()
    for sid, rounds in sessions.items():
        # Fresh pipeline state per session — mirrors how the live server runs
        # treated each session as an independent state machine.
        pipe = Task2Pipeline(llm=llm, config=pcfg)
        logger.info("Replaying session %s (%d rounds) under %s", sid, len(rounds), run_name)
        result = pipe.run_rounds(rounds)
        for rnd in result.rounds:
            per_round_preds.setdefault(rnd.round_id, []).append({
                "id": sid,
                "round": rnd.round_id,
                "prediction": rnd.selection.chosen_option,
            })

    # Save per-round in the same JSON shape as the original server submissions:
    # [{"predictions": [...], "emissions": {}}]
    for rnd_id, preds in sorted(per_round_preds.items()):
        path = submissions_dir / f"round{rnd_id}_run{run_idx}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([{"predictions": preds, "emissions": {}}], fh, ensure_ascii=False, indent=2)
    elapsed = time.monotonic() - t0
    logger.info(
        "Run %s done: %d sessions, %d rounds saved, %.1fs, LLM stats=%s",
        run_name, len(sessions), len(per_round_preds), elapsed, llm.stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to Task 2 replay config YAML")
    parser.add_argument("--runs", required=True, help="Comma-separated run names (e.g. run0,run1,run2)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    output_dir = Path(cfg.get("data", {}).get("output_dir", "output/mentalriskes_task2_test_replay"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_name in (r.strip() for r in args.runs.split(",")):
        replay_run(cfg, run_name, output_dir)


if __name__ == "__main__":
    main()
