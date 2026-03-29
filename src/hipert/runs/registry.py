"""Run registry and dispatch for eRisk 2026 Task 3 submissions.

Each run generator takes a config and output directory, and returns
rankings: dict[int, list[tuple[str, float]]] mapping symptom_id
to a list of (sentence_id, score) tuples sorted by score descending.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from hipert.config import PipelineConfig

logger = logging.getLogger(__name__)

# Type alias for run generators
Rankings = dict[int, list[tuple[str, float]]]
RunGenerator = Callable[[PipelineConfig], Rankings]

# System names per v2 spec Section 8.1
# v2 change: LLM cascade is now primary (Run 1), HiPerT is secondary (Run 2)
SYSTEM_NAMES = {
    1: "INSALyon_LLM_cascade",
    2: "INSALyon_HiPerT_full",
    3: "INSALyon_Ensemble",
    4: "INSALyon_DepTransfer",
    5: "INSALyon_BiEnc_baseline",
}

# Registry populated by run modules
RUN_REGISTRY: dict[int, RunGenerator] = {}


def register_run(run_id: int) -> Callable[[RunGenerator], RunGenerator]:
    """Decorator to register a run generator."""
    def wrapper(fn: RunGenerator) -> RunGenerator:
        RUN_REGISTRY[run_id] = fn
        return fn
    return wrapper


def generate_run(
    run_id: int,
    config: PipelineConfig,
) -> Rankings:
    """Generate rankings for a specific run."""
    if run_id not in RUN_REGISTRY:
        raise ValueError(
            f"Run {run_id} not registered. Available: {sorted(RUN_REGISTRY.keys())}"
        )

    generator = RUN_REGISTRY[run_id]
    logger.info("Generating Run %d (%s)...", run_id, SYSTEM_NAMES.get(run_id, "?"))
    rankings = generator(config)

    total = sum(len(v) for v in rankings.values())
    logger.info(
        "Run %d: %d symptoms, %d total ranked sentences",
        run_id, len(rankings), total,
    )
    return rankings


def list_runs() -> list[dict]:
    """List available runs with their status."""
    from hipert.runs import run1_full, run2_llm, run3_ensemble, run4_transfer, run5_bienc  # noqa: F401

    descriptions = {
        1: "LLM cascade scoring (PRIMARY)",
        2: "HiPerT v2 cross-encoder reranker",
        3: "Ensemble of Run 1 + Run 2 (RRF)",
        4: "Depression-only cross-condition transfer",
        5: "Bi-encoder cosine similarity baseline",
    }

    runs = []
    for run_id in sorted(SYSTEM_NAMES.keys()):
        runs.append({
            "id": run_id,
            "system_name": SYSTEM_NAMES[run_id],
            "description": descriptions.get(run_id, ""),
            "available": run_id in RUN_REGISTRY,
        })
    return runs
