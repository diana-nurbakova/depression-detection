"""Data loading for Task 2 trial and server data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import RoundRecord

logger = logging.getLogger(__name__)


def load_trial_rounds(trial_dir: Path) -> list[RoundRecord]:
    """Load all trial round JSON files, sorted by round number."""
    rounds = []
    for f in sorted(trial_dir.glob("round_*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)["trial"]
        rounds.append(RoundRecord(
            round_id=data["round"],
            patient_message=data["patient_input"],
            options={
                "option_1": data["option_1"],
                "option_2": data["option_2"],
                "option_3": data["option_3"],
            },
        ))
    rounds.sort(key=lambda r: r.round_id)
    logger.info("Loaded %d trial rounds from %s", len(rounds), trial_dir)
    return rounds


def parse_server_round(data: dict) -> RoundRecord:
    """Parse a single round from server GET response."""
    return RoundRecord(
        round_id=data["round"],
        patient_message=data["patient_input"],
        options={
            "option_1": data["option_1"],
            "option_2": data["option_2"],
            "option_3": data["option_3"],
        },
    )


# Ground-truth labels derived from therapist_response field in Task 1 trial data.
# Task 1 round_{n+1}.therapist_response == Task 2 round_n.option_X.
# All 18 rounds (1-18) have verified labels; round 19 has no next round.
TRIAL_GROUND_TRUTH: dict[int, int] = {
    1: 2,   # crisis/engagement: empathic validation + values-oriented question
    2: 3,   # committed action: remove pressure, just "do something"
    3: 3,   # acceptance: "¿qué se siente? ¿Es más liviano?"
    4: 3,   # acceptance: anxiety as uninvited guest metaphor
    5: 2,   # defusion: car passing metaphor ("dejar pasar")
    6: 1,   # defusion deepening: voice personification ("su propia vida")
    7: 3,   # present moment: tension + curiosity as spark
    8: 3,   # permanence with difficulty: courage, cloud metaphor
    9: 3,   # behavioral activation: library metaphor, read one phrase
    10: 2,  # graded exposure: observe the phrase as object
    11: 1,  # activation: continue, observe the experience
    12: 3,  # integration: explore new territory, object passing
    13: 1,  # pause: acknowledge pattern, give space
    14: 3,  # self-as-context: pause to process, strength recognition
    15: 1,  # integration: close eyes, observe what emerges
    16: 1,  # present moment: close eyes, observe
    17: 3,  # identity insight: math as anchor, competence
    18: 2,  # closing: hold onto that strength, schedule next
}

# Legacy alias (deprecated — use TRIAL_GROUND_TRUTH instead)
TRIAL_INFERRED_LABELS = TRIAL_GROUND_TRUTH


def load_session_labels(session_dir: Path) -> dict[int, int]:
    """Load gold labels from a session's labels.json file.

    Args:
        session_dir: path to a session directory containing labels.json.

    Returns:
        {round_id (int): correct_option (int)}
    """
    labels_path = session_dir / "labels.json"
    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def discover_sessions(simulated_dir: Path) -> list[Path]:
    """Discover all simulated session directories (those containing labels.json).

    Args:
        simulated_dir: root directory containing session subdirectories.

    Returns:
        Sorted list of session directory paths.
    """
    sessions = sorted(
        d for d in simulated_dir.iterdir()
        if d.is_dir() and (d / "labels.json").exists()
    )
    logger.info("Discovered %d simulated sessions in %s", len(sessions), simulated_dir)
    return sessions
