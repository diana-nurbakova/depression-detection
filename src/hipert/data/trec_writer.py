"""Write TREC-format ranking output for eRisk submission.

Output format (one line per ranked sentence):
    symptom_id Q0 sentence_docno rank score run_name

Top-1000 sentences per symptom, 18 symptoms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RUN_NAME = "HiPerT_ADHD"


def write_trec_rankings(
    silver_labels_dir: Path,
    output_dir: Path,
    top_n: int = 1000,
    run_name: str = RUN_NAME,
) -> None:
    """Generate TREC-format rankings from silver label JSONL files.

    Args:
        silver_labels_dir: Directory containing symptom_N.jsonl files.
        output_dir: Directory to write ranking files.
        top_n: Number of top sentences per symptom.
        run_name: Run identifier for the TREC output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.trec"

    all_lines: list[str] = []

    for symptom_id in range(1, 19):
        jsonl_path = silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not jsonl_path.exists():
            logger.warning(
                "No results for symptom %d at %s", symptom_id, jsonl_path,
            )
            continue

        # Load and rank by final_label (desc), then confidence_weight (desc)
        results = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Sort: primary by final_label descending, secondary by confidence_weight descending
        results.sort(
            key=lambda r: (r.get("final_label", 0), r.get("confidence_weight", 0)),
            reverse=True,
        )

        # Take top-N
        top_results = results[:top_n]

        for rank, result in enumerate(top_results, 1):
            docno = result["sentence_id"]
            # Score: combine label and confidence for a continuous ranking score
            label = result.get("final_label", 0)
            weight = result.get("confidence_weight", 0)
            score = label + weight  # Simple combination for ranking

            line = f"{symptom_id} Q0 {docno} {rank} {score:.4f} {run_name}"
            all_lines.append(line)

        logger.info(
            "Symptom %d: %d results ranked (top %d from %d total)",
            symptom_id, len(top_results), top_n, len(results),
        )

    # Write all rankings to a single file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")

    logger.info(
        "TREC rankings written: %d lines to %s",
        len(all_lines), output_path,
    )
