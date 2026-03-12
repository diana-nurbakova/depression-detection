"""Run 2: LLM_cascade — LLM scoring only (no trained encoder).

Pure LLM scoring with the full prompt engineering stack. Ranks by
LLM-assigned score with tie-breaking by confidence and cosine similarity.

Scoring function:
    φ(s, q) = final_label + confidence_weight
              (ties broken by confidence value, then retrieval cosine)

Reads silver label JSONL files and candidate JSON files (for cosine scores).
"""

from __future__ import annotations

import json
import logging

from hipert.config import PipelineConfig
from hipert.runs.registry import Rankings, register_run

logger = logging.getLogger(__name__)


@register_run(2)
def generate_llm_cascade(config: PipelineConfig) -> Rankings:
    """Generate Run 2 rankings from LLM scoring results."""
    silver_labels_dir = config.output_dir / "silver_labels"
    candidates_dir = config.output_dir / "candidates"
    rankings: Rankings = {}

    for symptom_id in range(1, 19):
        jsonl_path = silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not jsonl_path.exists():
            logger.warning("No silver labels for symptom %d", symptom_id)
            continue

        # Load LLM scoring results
        results = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not results:
            continue

        # Load retrieval cosine scores for tie-breaking
        cosine_scores: dict[str, float] = {}
        candidates_path = candidates_dir / f"symptom_{symptom_id}.json"
        if candidates_path.exists():
            with open(candidates_path, "r", encoding="utf-8") as f:
                for c in json.load(f):
                    cosine_scores[c["docno"]] = c.get("combined_score", 0.0)

        # Build ranking score per spec Section 5.3:
        # Primary: final_label (0-3)
        # Secondary: confidence from LLM output (1-5)
        # Tertiary: cosine similarity
        scored = []
        for r in results:
            docno = r["sentence_id"]
            label = r.get("final_label", 0)
            conf_weight = r.get("confidence_weight", 0.0)

            # Extract LLM confidence from the output that was used for the final label
            if r.get("escalated") and r.get("gpt_output"):
                llm_confidence = r["gpt_output"].get("confidence", 3)
            else:
                llm_confidence = r.get("llama_output", {}).get("confidence", 3)

            cosine = cosine_scores.get(docno, 0.0)

            # Composite ranking score:
            # label (0-3) dominates, then confidence_weight (0-1), then
            # LLM confidence (0.01-0.05), then cosine (0.001-0.001)
            score = (
                label * 10.0
                + conf_weight * 5.0
                + llm_confidence * 0.01
                + cosine * 0.001
            )
            scored.append((docno, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        rankings[symptom_id] = scored[:1000]

        logger.debug(
            "Symptom %d: %d sentences ranked by LLM score",
            symptom_id, len(rankings[symptom_id]),
        )

    return rankings
