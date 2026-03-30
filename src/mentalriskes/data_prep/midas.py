"""MIDAS data loader for Spanish MI (Motivational Interviewing) counseling sessions.

MIDAS contains 74 Spanish MI counseling sessions with dialogue act annotations.
Already in Spanish — no translation needed. Useful for:
1. Spanish therapeutic language examples for few-shot prompting
2. Counselor response patterns and dialogue acts
3. Vocabulary calibration for Spanish mental health register
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MIDASTurn:
    speaker: str  # "counselor", "patient", or "unclear"
    text_es: str  # Original Spanish
    text_en: str  # English translation (from dataset)
    turn_number: int
    tags: list[str] = field(default_factory=list)  # "question", "reflection", "empty"


@dataclass
class MIDASSession:
    session_id: str
    turns: list[MIDASTurn] = field(default_factory=list)

    @property
    def counselor_turns(self) -> list[MIDASTurn]:
        return [t for t in self.turns if t.speaker == "counselor"]

    @property
    def patient_turns(self) -> list[MIDASTurn]:
        return [t for t in self.turns if t.speaker == "patient"]


def load_midas(data_path: str | Path) -> list[MIDASSession]:
    """Load MIDAS Spanish MI counseling sessions.

    Args:
        data_path: Path to Spanish_MI.json

    Returns:
        List of MIDASSession objects.
    """
    data_path = Path(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sessions = []
    for session_id, turns_raw in raw.items():
        turns = []
        for t in turns_raw:
            tags = [tp["tag"] for tp in t.get("tag_positions", []) if tp.get("tag")]
            turns.append(MIDASTurn(
                speaker=t["speaker"],
                text_es=t.get("turn", ""),
                text_en=t.get("translated_turn", ""),
                turn_number=t.get("turn_number", 0),
                tags=tags,
            ))
        sessions.append(MIDASSession(session_id=session_id, turns=turns))

    logger.info("Loaded %d MIDAS sessions (%d total turns)", len(sessions), sum(len(s.turns) for s in sessions))
    return sessions


def extract_counselor_responses(
    sessions: list[MIDASSession],
    min_length: int = 30,
    with_context: bool = True,
) -> list[dict]:
    """Extract counselor responses with optional preceding patient context.

    Returns list of dicts with:
    - session_id, turn_number
    - counselor_text_es, counselor_text_en
    - patient_text_es, patient_text_en (preceding patient turn, if with_context)
    - tags (dialogue act labels)
    """
    examples = []
    for session in sessions:
        for i, turn in enumerate(session.turns):
            if turn.speaker != "counselor":
                continue
            if len(turn.text_es) < min_length:
                continue

            example = {
                "session_id": session.session_id,
                "turn_number": turn.turn_number,
                "counselor_text_es": turn.text_es,
                "counselor_text_en": turn.text_en,
                "tags": turn.tags,
            }

            if with_context and i > 0:
                prev = session.turns[i - 1]
                if prev.speaker == "patient":
                    example["patient_text_es"] = prev.text_es
                    example["patient_text_en"] = prev.text_en

            examples.append(example)

    logger.info("Extracted %d counselor responses from %d sessions", len(examples), len(sessions))
    return examples


def extract_dialogue_segments(
    sessions: list[MIDASSession],
    segment_length: int = 6,
    overlap: int = 2,
) -> list[dict]:
    """Extract overlapping dialogue segments for few-shot examples.

    Each segment is a window of consecutive turns, formatted for prompt injection.

    Args:
        sessions: MIDAS sessions.
        segment_length: Number of turns per segment.
        overlap: Overlap between consecutive segments.

    Returns:
        List of segment dicts with turns and metadata.
    """
    segments = []
    for session in sessions:
        turns = session.turns
        step = segment_length - overlap
        for start in range(0, len(turns) - segment_length + 1, step):
            segment_turns = turns[start : start + segment_length]
            segments.append({
                "session_id": session.session_id,
                "start_turn": segment_turns[0].turn_number,
                "end_turn": segment_turns[-1].turn_number,
                "turns": [
                    {
                        "speaker": t.speaker,
                        "text_es": t.text_es,
                        "text_en": t.text_en,
                        "tags": t.tags,
                    }
                    for t in segment_turns
                ],
            })

    logger.info("Extracted %d dialogue segments from %d sessions", len(segments), len(sessions))
    return segments


def save_counselor_responses(examples: list[dict], output_path: str | Path) -> None:
    """Save extracted counselor responses to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d counselor responses to %s", len(examples), output_path)
