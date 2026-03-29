"""Extract training data for cross-encoder v2 from LLM cascade outputs.

Reads silver label JSONL files and candidate JSON files to produce
(symptom_text, sentence_text, score, confidence) training triples.

Source: LLM cascade has already scored all candidates (1,000 per symptom).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ASRS-v1.1 item text (Layer 1 definitions)
ASRS_ITEMS = {
    1: "How often do you have trouble wrapping up the final details of a project, once the challenging parts have been done?",
    2: "How often do you have difficulty getting things in order when you have to do a task that requires organization?",
    3: "How often do you have problems remembering appointments or obligations?",
    4: "When you have a task that requires a lot of thought, how often do you avoid or delay getting started?",
    5: "How often do you fidget or squirm with your hands or feet when you have to sit down for a long time?",
    6: "How often do you leave your seat in meetings or other situations in which you are expected to remain seated?",
    7: "How often do you feel overly active and compelled to do things, like you were driven by a motor?",
    8: "How often do you make careless mistakes when you have to work on a boring or difficult project?",
    9: "How often do you have difficulty keeping your attention when you are doing boring or repetitive work?",
    10: "How often are you distracted by activity or noise around you?",
    11: "How often do you have difficulty concentrating on what people say to you, even when they are speaking to you directly?",
    12: "How often do you leave your seat in meetings or other situations in which you are expected to remain seated?",
    13: "How often do you feel restless or fidgety?",
    14: "How often do you have difficulty unwinding and relaxing when you have time to yourself?",
    15: "How often do you find yourself talking too much when you are in social situations?",
    16: "When you're in a conversation, how often do you find yourself finishing the sentences of the people you are talking to, before they can finish them themselves?",
    17: "How often do you have difficulty waiting your turn in situations when turn-taking is required?",
    18: "How often do you interrupt others when they are busy?",
}


def extract_training_data(
    silver_labels_dir: Path,
    candidates_dir: Path,
    symptom_ids: list[int] | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Extract training data from LLM cascade outputs.

    Joins silver label scores with candidate text/context to produce
    training triples for the cross-encoder.

    Args:
        silver_labels_dir: Directory with symptom_{id}.jsonl files.
        candidates_dir: Directory with symptom_{id}.json files.
        symptom_ids: Which symptoms to load (default: all 1-18).
        min_confidence: Minimum confidence_weight to include.

    Returns:
        List of dicts with keys:
            symptom_id, symptom_text, sentence_id, sentence_text,
            score (0-3), confidence (0-1), pre, post
    """
    if symptom_ids is None:
        symptom_ids = list(range(1, 19))

    training_data = []

    for sid in symptom_ids:
        # Load candidate texts
        candidates_path = candidates_dir / f"symptom_{sid}.json"
        if not candidates_path.exists():
            logger.warning("No candidates file for symptom %d", sid)
            continue

        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Build lookup: docno -> candidate data
        cand_lookup: dict[str, dict] = {}
        for c in candidates:
            cand_lookup[c["docno"]] = c

        # Load silver labels
        labels_path = silver_labels_dir / f"symptom_{sid}.jsonl"
        if not labels_path.exists():
            logger.warning("No silver labels for symptom %d", sid)
            continue

        count = 0
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    label_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sentence_id = label_data["sentence_id"]
                confidence = label_data.get("confidence_weight", 0.5)

                if confidence < min_confidence:
                    continue

                # Look up candidate text
                cand = cand_lookup.get(sentence_id)
                if cand is None:
                    continue

                training_data.append({
                    "symptom_id": sid,
                    "symptom_text": ASRS_ITEMS[sid],
                    "sentence_id": sentence_id,
                    "sentence_text": cand["text"],
                    "pre": cand.get("pre", ""),
                    "post": cand.get("post", ""),
                    "score": label_data.get("final_label", 0),
                    "confidence": confidence,
                })
                count += 1

        logger.info("Symptom %d: extracted %d training examples", sid, count)

    # Log distribution
    if training_data:
        from collections import Counter
        dist = Counter(d["score"] for d in training_data)
        total = len(training_data)
        logger.info(
            "Training data: %d examples. Distribution: %s",
            total,
            {k: f"{v} ({100*v/total:.1f}%)" for k, v in sorted(dist.items())},
        )

    return training_data


def save_training_data(data: list[dict], output_path: Path) -> None:
    """Save extracted training data to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Saved %d training examples to %s", len(data), output_path)


def load_training_data(input_path: Path) -> list[dict]:
    """Load training data from JSONL."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d training examples from %s", len(data), input_path)
    return data
