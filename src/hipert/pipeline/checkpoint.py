"""Resumable checkpoint management.

Saves pipeline state and scored results incrementally.
Results are stored as JSONL (one line per scored pair) so partial
progress is preserved across crashes.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from hipert.models import ScoringResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringProgress:
    """Track scoring progress for one symptom."""

    symptom_id: int
    total_candidates: int = 0
    scored_count: int = 0
    completed: bool = False


@dataclass
class PipelineState:
    """Track overall pipeline progress."""

    corpus_loaded: bool = False
    embeddings_computed: dict[str, bool] = field(default_factory=dict)
    candidates_selected: dict[int, bool] = field(default_factory=dict)
    scoring_progress: dict[int, ScoringProgress] = field(default_factory=dict)


class CheckpointManager:
    """Manages pipeline state persistence and result storage."""

    def __init__(self, checkpoint_dir: Path, silver_labels_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.silver_labels_dir = silver_labels_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        silver_labels_dir.mkdir(parents=True, exist_ok=True)

        self._state_path = checkpoint_dir / "state.json"
        self._write_lock = threading.Lock()

    def load_state(self) -> PipelineState:
        """Load pipeline state from disk, or return fresh state."""
        if self._state_path.exists():
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = PipelineState()
            state.corpus_loaded = data.get("corpus_loaded", False)
            state.embeddings_computed = data.get("embeddings_computed", {})
            state.candidates_selected = {
                int(k): v
                for k, v in data.get("candidates_selected", {}).items()
            }
            for k, v in data.get("scoring_progress", {}).items():
                state.scoring_progress[int(k)] = ScoringProgress(
                    symptom_id=int(k),
                    total_candidates=v.get("total_candidates", 0),
                    scored_count=v.get("scored_count", 0),
                    completed=v.get("completed", False),
                )

            logger.info("Loaded pipeline state from %s", self._state_path)
            return state

        return PipelineState()

    def save_state(self, state: PipelineState) -> None:
        """Save pipeline state to disk."""
        data = {
            "corpus_loaded": state.corpus_loaded,
            "embeddings_computed": state.embeddings_computed,
            "candidates_selected": {
                str(k): v for k, v in state.candidates_selected.items()
            },
            "scoring_progress": {
                str(k): {
                    "symptom_id": v.symptom_id,
                    "total_candidates": v.total_candidates,
                    "scored_count": v.scored_count,
                    "completed": v.completed,
                }
                for k, v in state.scoring_progress.items()
            },
        }

        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_scored_count(self, symptom_id: int) -> int:
        """Count already-scored results for a symptom from JSONL file."""
        filepath = self.silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not filepath.exists():
            return 0

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_scored_ids(self, symptom_id: int) -> set[str]:
        """Get set of already-scored sentence IDs for a symptom."""
        filepath = self.silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not filepath.exists():
            return set()

        scored: set[str] = set()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        scored.add(data["sentence_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return scored

    def append_result(
        self,
        symptom_id: int,
        result: ScoringResult,
    ) -> None:
        """Append a single scoring result to the JSONL file (thread-safe)."""
        filepath = self.silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        line = json.dumps(result.to_dict(), ensure_ascii=False, default=str) + "\n"

        with self._write_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)

    def load_results(self, symptom_id: int) -> list[dict]:
        """Load all scored results for a symptom."""
        filepath = self.silver_labels_dir / f"symptom_{symptom_id}.jsonl"
        if not filepath.exists():
            return []

        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return results
