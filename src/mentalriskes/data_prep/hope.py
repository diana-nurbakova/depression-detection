"""HOPE (READER) data loader for counseling dialogue acts.

HOPE contains counseling dialogues with response-act annotations.
The actual dataset CSVs are not included in the repo; if available,
this module loads them. Otherwise it provides a stub.

Used for: therapist behavior distribution analysis, dialogue act labels.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Response-act labels from the HOPE paper
RESPONSE_ACTS = [
    "Question",
    "Restatement",
    "Reflection",
    "Self-disclosure",
    "Affirmation",
    "Information",
    "Suggestion",
    "Others",
]


@dataclass
class HOPETurn:
    speaker: str  # "therapist" or "client"
    text: str
    response_act: str = ""


@dataclass
class HOPEDialogue:
    dialogue_id: int
    turns: list[HOPETurn] = field(default_factory=list)


def load_hope_csv(csv_path: str | Path) -> list[HOPEDialogue]:
    """Load HOPE dataset from CSV file (if available).

    Expected CSV columns: dialogue_id, turn_id, speaker, text, response_act

    Args:
        csv_path: Path to train_new_final.csv or test.csv

    Returns:
        List of HOPEDialogue objects.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("HOPE CSV not found: %s (dataset not included in repo)", csv_path)
        return []

    dialogues_map: dict[int, HOPEDialogue] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_id = int(row.get("dialogue_id", row.get("dialog_id", 0)))
            if d_id not in dialogues_map:
                dialogues_map[d_id] = HOPEDialogue(dialogue_id=d_id)

            dialogues_map[d_id].turns.append(HOPETurn(
                speaker=row.get("speaker", ""),
                text=row.get("text", row.get("utterance", "")),
                response_act=row.get("response_act", row.get("label", "")),
            ))

    dialogues = sorted(dialogues_map.values(), key=lambda d: d.dialogue_id)
    logger.info("Loaded %d HOPE dialogues from %s", len(dialogues), csv_path)
    return dialogues


def check_hope_data(repo_dir: str | Path) -> dict[str, bool]:
    """Check which HOPE data files are available.

    Returns dict mapping expected filename -> exists.
    """
    repo_dir = Path(repo_dir)
    expected = ["train_new_final.csv", "test.csv", "s_train.csv"]
    return {f: (repo_dir / f).exists() for f in expected}


def extract_response_act_distribution(dialogues: list[HOPEDialogue]) -> dict[str, int]:
    """Compute response-act frequency distribution across therapist turns."""
    counts: dict[str, int] = {}
    for d in dialogues:
        for t in d.turns:
            if t.speaker in ("therapist", "counselor") and t.response_act:
                counts[t.response_act] = counts.get(t.response_act, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
