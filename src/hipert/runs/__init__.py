"""Run generators for eRisk 2026 Task 3 submission.

5 runs, each producing TREC-format rankings with different strategies:
- Run 1: HiPerT_full (full pipeline with trained encoder)
- Run 2: LLM_cascade (LLM scoring only)
- Run 3: Ensemble_1+2 (RRF combination of Run 1 + Run 2)
- Run 4: DepTransfer (depression-only cross-condition transfer)
- Run 5: BiEnc_baseline (cosine similarity only)
"""

from hipert.runs.registry import RUN_REGISTRY, generate_run, list_runs

# Import run modules to trigger @register_run decorators
from hipert.runs import run1_full, run2_llm, run3_ensemble, run4_transfer, run5_bienc  # noqa: F401

__all__ = ["RUN_REGISTRY", "generate_run", "list_runs"]
