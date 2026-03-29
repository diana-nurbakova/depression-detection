"""Data loading and conversation history management for MentalRiskES Task 1."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single turn in a therapeutic conversation."""
    round: int
    role: str  # "patient" or "therapist"
    text: str


@dataclass
class SessionHistory:
    """Accumulated conversation history for a single session."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, round_n: int, role: str, text: str) -> None:
        self.turns.append(Turn(round=round_n, role=role, text=text))

    def format_for_prompt(self) -> str:
        """Format conversation history as a string for LLM prompt injection."""
        lines = []
        for turn in self.turns:
            role = turn.role.upper()
            lines.append(f"[Round {turn.round} — {role}]: {turn.text}")
        return "\n\n".join(lines)

    @property
    def latest_round(self) -> int:
        return self.turns[-1].round if self.turns else 0


class ConversationStore:
    """Manages conversation histories for all sessions."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionHistory] = {}

    def update_from_server_response(self, messages: dict) -> list[str]:
        """
        Ingest a round of server messages, updating session histories.

        Args:
            messages: dict from server GET, keyed by session_id.
                Each value has 'round', 'patient_input', and optionally 'therapist_response'.

        Returns:
            List of session_ids that were updated.
        """
        updated = []
        for session_id, data in messages.items():
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionHistory(session_id=session_id)

            session = self.sessions[session_id]
            round_n = data["round"]

            # Add therapist response first (from previous round)
            if "therapist_response" in data and data["therapist_response"]:
                session.add_turn(round_n, "therapist", data["therapist_response"])

            # Add patient input
            session.add_turn(round_n, "patient", data["patient_input"])
            updated.append(session_id)

        return updated

    def get_history(self, session_id: str) -> SessionHistory:
        return self.sessions.get(session_id, SessionHistory(session_id=session_id))

    def get_context(self, session_id: str, max_turns: int | None = None) -> str:
        """
        Get formatted conversation context for a session.

        Applies context window management for long conversations:
        - Always includes the first patient turn
        - Always includes the last 3 complete exchanges
        - Middle turns included if within budget
        """
        session = self.get_history(session_id)
        turns = session.turns

        if max_turns is None or len(turns) <= max_turns:
            return session.format_for_prompt()

        # Keep first turn + last N turns
        keep_last = min(6, max_turns - 1)  # ~3 exchanges
        selected = [turns[0]] + turns[-keep_last:]

        lines = []
        for turn in selected:
            role = turn.role.upper()
            lines.append(f"[Round {turn.round} — {role}]: {turn.text}")
        return "\n\n".join(lines)


def load_trial_data(trial_dir: str | Path) -> dict[int, dict]:
    """
    Load all trial round files from disk.

    Returns:
        dict mapping round_number -> raw server-format data
    """
    trial_dir = Path(trial_dir)
    rounds = {}

    for path in sorted(trial_dir.glob("round_*.json")):
        round_n = int(path.stem.split("_")[1])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rounds[round_n] = data
        logger.debug("Loaded trial round %d from %s", round_n, path)

    logger.info("Loaded %d trial rounds from %s", len(rounds), trial_dir)
    return rounds


def load_primate_dataset(path: str | Path) -> list[dict]:
    """Load the PRIMATE dataset (list of posts with PHQ-9 annotations)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        posts = data
    elif isinstance(data, dict):
        posts = data.get("data", data.get("posts", list(data.values())))
    else:
        raise ValueError(f"Unexpected PRIMATE data format: {type(data)}")

    logger.info("Loaded %d posts from PRIMATE dataset", len(posts))
    return posts
