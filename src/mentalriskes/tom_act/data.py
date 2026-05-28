"""Multi-session test-data loader for the ToM × ACT analysis.

The released MentalRiskES test files are **multi-session** JSON objects keyed by
session id (``S01``, ``S03`` …), one file per round number — *not* the
per-session files described in spec §2.2. This loader merges the three sources
(task1 alternating dialogue, task2 candidates, gold selection) into per-session
round sequences and provides the context builders the prompts need.

Cross-task alignment (verified, spec §2.3):
  - task1 ``round_t.patient_input`` == task2 ``round_t.patient_input``.
  - task1 ``round_t.therapist_response`` (t≥2) is the therapist turn delivered
    *before* patient turn t — i.e. the gold-selected candidate from round t-1.
  - task2 ``round_t`` gold-selected option == task1 ``round_{t+1}.therapist_response``.

For ``gold = actual = therapist_response`` we drive state regeneration from the
actual delivered therapist turn (task1 ``therapist_response``); the gold
candidate *identity* for RQ3/RQ4 comes from the task2 gold file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ROUND_RE = re.compile(r"round_(\d+)")


@dataclass
class RoundData:
    """One patient-round within a session."""

    session_id: str
    round: int
    patient_input: str
    # Therapist turn delivered immediately BEFORE this patient turn (None at round 1).
    therapist_response: str | None
    # Three candidate therapist responses to THIS patient turn (task2).
    options: dict[str, str] = field(default_factory=dict)
    # Gold-selected candidate index 1|2|3 (parsed from "option_X"), None if absent.
    gold_option: int | None = None

    @property
    def gold_option_text(self) -> str | None:
        if self.gold_option is None:
            return None
        return self.options.get(f"option_{self.gold_option}")

    def candidate_text(self, option: int) -> str:
        return self.options[f"option_{option}"]


@dataclass
class Session:
    """A full session: ordered rounds plus session-level gold instrument arrays."""

    session_id: str
    rounds: list[RoundData]
    # Session-level gold item arrays from gold_label.json (raw, not reverse-scored).
    gold_phq9: list[int] = field(default_factory=list)
    gold_gad7: list[int] = field(default_factory=list)
    gold_compact10: list[int] = field(default_factory=list)

    def round(self, t: int) -> RoundData:
        for r in self.rounds:
            if r.round == t:
                return r
        raise KeyError(f"{self.session_id} has no round {t}")

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)


# ---------------------------------------------------------------------------
# Gold parsing
# ---------------------------------------------------------------------------

def parse_gold_option(value) -> int | None:
    """Parse a gold ``correct_option`` value into an int 1|2|3.

    Handles the released string form (``"option_2"``) and the spec's documented
    integer form (``2``).
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Multi-session round-file reading
# ---------------------------------------------------------------------------

def _read_round_files(directory: Path) -> dict[str, dict[int, dict]]:
    """Read all ``round_*.json`` in a dir → {session_id: {round_n: payload}}."""
    out: dict[str, dict[int, dict]] = {}
    for f in sorted(directory.glob("round_*.json")):
        m = _ROUND_RE.search(f.name)
        if not m:
            continue
        n = int(m.group(1))
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for session_id, payload in data.items():
            out.setdefault(session_id, {})[n] = payload
    return out


def load_sessions(
    task1_dir: str | Path,
    task2_dir: str | Path,
    gold_dir: str | Path,
    session_gold_path: str | Path,
    sessions: list[str] | None = None,
) -> dict[str, Session]:
    """Load all sessions, merging task1, task2, and gold sources.

    Args:
        task1_dir: dir of task1 ``round_N.json`` (alternating dialogue).
        task2_dir: dir of task2 ``round_N.json`` (candidates).
        gold_dir: dir of task2 ``round_N_gold.json`` (selected option).
        session_gold_path: ``gold_label.json`` with session-level instrument arrays.
        sessions: optional whitelist of session ids to keep.

    Returns:
        {session_id: Session}, sessions sorted, rounds sorted ascending.
    """
    t1 = _read_round_files(Path(task1_dir))
    t2 = _read_round_files(Path(task2_dir))
    gold = _read_round_files(Path(gold_dir))

    with open(session_gold_path, encoding="utf-8") as fh:
        session_gold = json.load(fh)

    keep = set(sessions) if sessions else set(t2.keys())
    result: dict[str, Session] = {}

    for session_id in sorted(keep, key=lambda s: int(re.sub(r"\D", "", s) or 0)):
        if session_id not in t2:
            logger.warning("Session %s requested but absent from task2 data; skipping", session_id)
            continue

        rounds: list[RoundData] = []
        for n in sorted(t2[session_id].keys()):
            t2_payload = t2[session_id][n]
            t1_payload = t1.get(session_id, {}).get(n, {})
            gold_payload = gold.get(session_id, {}).get(n, {})

            options = {
                k: t2_payload[k]
                for k in ("option_1", "option_2", "option_3")
                if k in t2_payload
            }
            rounds.append(RoundData(
                session_id=session_id,
                round=n,
                patient_input=t2_payload.get("patient_input")
                or t1_payload.get("patient_input", ""),
                therapist_response=t1_payload.get("therapist_response"),
                options=options,
                gold_option=parse_gold_option(gold_payload.get("correct_option")),
            ))

        sg = session_gold.get(session_id, {})
        result[session_id] = Session(
            session_id=session_id,
            rounds=rounds,
            gold_phq9=sg.get("PHQ-9", []),
            gold_gad7=sg.get("GAD-7", []),
            gold_compact10=sg.get("CompACT-10", []),
        )
        logger.info("Loaded session %s: %d rounds", session_id, len(rounds))

    return result


# ---------------------------------------------------------------------------
# Context builders (for prompts, spec Appendix A notation)
# ---------------------------------------------------------------------------

def patient_turn(session: Session, t: int) -> str:
    """Single patient turn at round t — ``{PATIENT_TURN}``."""
    return session.round(t).patient_input


def cumulative_patient(session: Session, t: int) -> str:
    """Concatenated patient turns rounds 1..t with round numbers — ``{CUMULATIVE_PATIENT}``."""
    lines = []
    for r in session.rounds:
        if r.round > t:
            break
        lines.append(f"[Ronda {r.round} — PACIENTE]: {r.patient_input}")
    return "\n\n".join(lines)


def cumulative_dialogue(session: Session, t: int) -> str:
    """Full alternating patient+therapist dialogue rounds 1..t — ``{CUMULATIVE_DIALOGUE}``.

    Each round's therapist turn (the response delivered before that round's
    patient turn) precedes the patient turn, reconstructing the real exchange.
    """
    lines = []
    for r in session.rounds:
        if r.round > t:
            break
        if r.therapist_response:
            lines.append(f"[Ronda {r.round} — TERAPEUTA]: {r.therapist_response}")
        lines.append(f"[Ronda {r.round} — PACIENTE]: {r.patient_input}")
    return "\n\n".join(lines)
