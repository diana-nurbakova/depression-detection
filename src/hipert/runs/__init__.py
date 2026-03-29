"""Run generators for eRisk 2026 Task 3 submission (v2 ordering).

5 runs, each producing TREC-format rankings with different strategies:
- Run 1: LLM_cascade (LLM scoring — PRIMARY)
- Run 2: HiPerT_full (cross-encoder reranker v2)
- Run 3: Ensemble (RRF fusion of Run 1 + Run 2)
- Run 4: DepTransfer (depression-only cross-condition transfer)
- Run 5: BiEnc_baseline (cosine similarity only)
"""

from hipert.runs.registry import RUN_REGISTRY, generate_run, list_runs

# Import run modules to trigger @register_run decorators
from hipert.runs import run1_full, run2_llm, run3_ensemble, run4_transfer, run5_bienc  # noqa: F401

__all__ = ["RUN_REGISTRY", "generate_run", "list_runs"]
