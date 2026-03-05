"""eRisk submission file formatting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import BDI_ITEMS, PersonaResult

logger = logging.getLogger(__name__)


def format_interactions(results: list[PersonaResult]) -> list[dict]:
    """Format all persona conversations for eRisk interactions_runN.json.

    Format:
    [
      {
        "LLM": "1",  # String number (1-based)
        "conversation": [
          {"role": "user", "message": "..."},
          {"role": "assistant", "message": "..."}
        ]
      }
    ]
    """
    interactions = []
    for result in results:
        conv = []
        for turn in result.conversation:
            conv.append({
                "role": turn.role,
                "message": turn.message,
            })

        interactions.append({
            "LLM": str(result.persona_number + 1),  # 1-based string
            "conversation": conv,
        })

    return interactions


def format_results(results: list[PersonaResult]) -> list[dict]:
    """Format all persona results for eRisk results_runN.json.

    Format:
    [
      {
        "LLM": "1",           # String number (1-based)
        "bdi-score": 37,       # Integer
        "key-symptoms": [...]  # Up to 4 canonical BDI-II names
      }
    ]
    """
    erisk_results = []
    for result in results:
        # Only include symptoms if depression is present
        symptoms = result.top_4_symptoms if result.final_total >= 14 else []

        erisk_results.append({
            "LLM": str(result.persona_number + 1),
            "bdi-score": result.final_total,
            "key-symptoms": symptoms[:4],
        })

    return erisk_results


def save_internal_results(result: PersonaResult, output_dir: Path, suffix: str = ""):
    """Save detailed internal results for a single persona.

    Args:
        result: PersonaResult to save.
        output_dir: Directory to save into (e.g., runs/task1/persona04/).
        suffix: Optional suffix for filenames (e.g., "_1" for run 1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Item scores
    item_scores_dict = {}
    for item_id, item in result.item_scores.items():
        key = f"{item_id}_{BDI_ITEMS.get(item_id, 'unknown').lower().replace(' ', '_')}"
        item_scores_dict[key] = {
            "score": item.score,
            "confidence": item.confidence,
            "state": item.state.value,
            "source": item.source,
            "evidence": item.evidence,
        }

    internal = {
        "persona_id": result.persona_id,
        "bdi-score": result.final_total,
        "severity_band": result.final_band.value,
        "key-symptoms": result.top_4_symptoms,
        "item_scores": item_scores_dict,
        "scoring_metadata": {
            "pass1_total": result.pass1_total,
            "pass2_total": result.pass2_total,
            "final_total": result.final_total,
            "conversation_turns": len(result.conversation),
        },
    }

    # Add justificator output if available
    if result.justificator_output:
        internal["justificator"] = {
            "patterns_detected": result.justificator_output.patterns_detected,
            "adjustments_made": result.justificator_output.adjustments_made,
            "clinical_narrative": result.justificator_output.clinical_narrative,
        }

    path = output_dir / f"internal{suffix}.json"
    with open(path, "w") as f:
        json.dump(internal, f, indent=2)
    logger.info("Saved internal results: %s", path)

    # Save full conversation
    conv_path = output_dir / f"conversation{suffix}.json"
    conv_data = [
        {"role": t.role, "message": t.message, "turn": t.turn_number}
        for t in result.conversation
    ]
    with open(conv_path, "w") as f:
        json.dump(conv_data, f, indent=2)
