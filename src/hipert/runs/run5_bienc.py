"""Run 5: BiEnc_baseline — Bi-encoder cosine similarity only.

The simplest system: rank by cosine similarity between mpnet embeddings
and expanded symptom query embeddings. No LLM, no trained model.

Scoring function:
    φ(s, q) = max_k cosine(emb(s), emb(q_k))
              + 0.05 * 1[first_person(s)]
              + 0.03 * 1[keyword_match(s, q)]

Reads pre-computed candidate files (already contain retrieval_score + keyword_boost).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)


@register_run(5)
def generate_bienc_baseline(config: PipelineConfig) -> Rankings:
    """Generate Run 5 rankings from pre-computed retrieval candidates."""
    candidates_dir = config.output_dir / "candidates"
    rankings: Rankings = {}

    for symptom_id in range(1, 19):
        candidates_path = candidates_dir / f"symptom_{symptom_id}.json"
        if not candidates_path.exists():
            logger.warning("No candidates for symptom %d", symptom_id)
            continue

        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Rank by combined_score (retrieval_score + keyword_boost), already sorted
        scored = [
            (c["docno"], c.get("combined_score", c.get("retrieval_score", 0.0)))
            for c in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        rankings[symptom_id] = scored[:1000]

        logger.debug(
            "Symptom %d: %d candidates ranked (top score=%.4f)",
            symptom_id, len(rankings[symptom_id]),
            scored[0][1] if scored else 0.0,
        )

    return rankings
