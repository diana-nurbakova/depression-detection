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


# Inferred labels from trial data analysis (patient-echo method).
# Rounds 1 and 19 are uncertain (no next-turn echo available).
TRIAL_INFERRED_LABELS: dict[int, int] = {
    2: 3,   # acceptance: remove pressure, just "do something"
    3: 1,   # committed action: 15 min concrete step
    4: 2,   # acceptance: let anxiety be present
    5: 2,   # defusion: car passing metaphor
    6: 1,   # defusion deepening: cloud metaphor
    7: 2,   # present moment: physical awareness
    8: 1,   # permanence with difficulty: coexist with tension
    9: 3,   # behavioral activation: read one phrase
    10: 2,  # graded exposure: step by step
    11: 1,  # activation: continue with the exercise
    12: 3,  # integration: notice the pattern
    13: 2,  # pause: give space
    14: 1,  # self-as-context: observer perspective
    15: 3,  # integration: strength recognition
    16: 1,  # present moment: close eyes, observe
    17: 3,  # identity insight: "I am good at math"
    18: 3,  # closing: schedule next session
}
