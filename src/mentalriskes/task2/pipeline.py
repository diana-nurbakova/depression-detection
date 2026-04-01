"""Pipeline orchestration for Task 2: runs selector over rounds."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..llm_client import LLMClient
from .data import load_trial_rounds
from .models import RoundRecord, SelectionResult, SharedState
from .selector import Task2Selector, run_with_permutation_voting

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""

    name: str = "default"
    model: str = "llama3.3:70b"
    framing: str = "FUNC"  # FUNC | HYB | TOM-B | TOM-C
    pipeline: str = "B"  # A | B | B+
    lang: str = "es"  # es | en
    lookback_window: int = 3  # W1, W3, W5
    permutation_voting: bool = False
    calibration: bool = False  # Experiential tiebreaker calibration

    @property
    def config_id(self) -> str:
        perm = "PERM" if self.permutation_voting else "FIX"
        cal = "_CAL" if self.calibration else ""
        # Sanitize model name for filenames (remove / and :)
        model_short = self.model.split("/")[-1].split(":")[0]
        return f"{self.pipeline}_{model_short}_{self.lang}_{self.framing}_{perm}_W{self.lookback_window}{cal}"


@dataclass
class RoundOutput:
    """Output from processing a single round."""

    round_id: int
    selection: SelectionResult
    state_snapshot: dict
    elapsed_ms: float = 0.0


@dataclass
class PipelineResult:
    """Output from a full pipeline run."""

    config: PipelineConfig
    rounds: list[RoundOutput] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    llm_stats: dict = field(default_factory=dict)

    def predictions(self) -> list[int]:
        """Return list of chosen options in round order."""
        return [r.selection.chosen_option for r in sorted(self.rounds, key=lambda r: r.round_id)]


class Task2Pipeline:
    """Main pipeline for Task 2 response selection."""

    def __init__(self, llm: LLMClient, config: PipelineConfig) -> None:
        self.llm = llm
        self.config = config
        self.selector = Task2Selector(
            llm=llm,
            framing=config.framing,
            pipeline=config.pipeline,
            lang=config.lang,
            lookback_window=config.lookback_window,
            calibration=config.calibration,
        )

    def run_trial(self, trial_dir: Path) -> PipelineResult:
        """Run the pipeline on trial data."""
        rounds = load_trial_rounds(trial_dir)
        return self._run_rounds(rounds)

    def run_rounds(self, rounds: list[RoundRecord]) -> PipelineResult:
        """Run the pipeline on a list of rounds."""
        return self._run_rounds(rounds)

    def _run_rounds(self, rounds: list[RoundRecord]) -> PipelineResult:
        """Process all rounds sequentially."""
        result = PipelineResult(config=self.config)
        t0 = time.monotonic()

        for rnd in rounds:
            t_round = time.monotonic()

            if self.config.permutation_voting:
                # First do state update, then permutation vote
                if self.config.pipeline in ("B", "B+"):
                    self.selector._step1_state_update(
                        rnd.round_id, rnd.patient_message
                    )
                    # Need to add to transcript manually since we bypass process_round
                    self.selector.state.transcript.append(rnd)

                selection = run_with_permutation_voting(
                    llm=self.llm,
                    state=self.selector.state,
                    round_id=rnd.round_id,
                    patient_message=rnd.patient_message,
                    options=rnd.options,
                    framing=self.config.framing,
                    pipeline=self.config.pipeline,
                    lang=self.config.lang,
                    lookback_window=self.config.lookback_window,
                )

                # Record in state
                rnd.selected_option = selection.chosen_option
                rnd.selected_response_text = rnd.options[f"option_{selection.chosen_option}"]
                self.selector.state.selection_log.append({
                    "round": rnd.round_id,
                    "chosen": selection.chosen_option,
                    "tag": selection.primary_tag,
                })
            else:
                selection = self.selector.process_round(
                    rnd.round_id, rnd.patient_message, rnd.options,
                )

            elapsed = (time.monotonic() - t_round) * 1000

            output = RoundOutput(
                round_id=rnd.round_id,
                selection=selection,
                state_snapshot=self.selector.state.to_state_json(),
                elapsed_ms=elapsed,
            )
            result.rounds.append(output)

            logger.info(
                "Round %d: selected option %d (%s) in %.0fms",
                rnd.round_id, selection.chosen_option, selection.primary_tag, elapsed,
            )

        result.total_elapsed_ms = (time.monotonic() - t0) * 1000
        result.llm_stats = self.llm.stats
        return result

    def save_result(self, result: PipelineResult, output_dir: Path) -> Path:
        """Save pipeline result to JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{result.config.config_id}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            # Header line
            header = {
                "type": "config",
                "config_id": result.config.config_id,
                "config": {
                    "name": result.config.name,
                    "model": result.config.model,
                    "framing": result.config.framing,
                    "pipeline": result.config.pipeline,
                    "lang": result.config.lang,
                    "lookback_window": result.config.lookback_window,
                    "permutation_voting": result.config.permutation_voting,
                },
                "total_elapsed_ms": result.total_elapsed_ms,
                "llm_stats": result.llm_stats,
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            # Round outputs
            for rnd in result.rounds:
                entry = {
                    "type": "round",
                    "round_id": rnd.round_id,
                    "chosen_option": rnd.selection.chosen_option,
                    "primary_tag": rnd.selection.primary_tag,
                    "reasoning": rnd.selection.reasoning,
                    "state_snapshot": rnd.state_snapshot,
                    "elapsed_ms": rnd.elapsed_ms,
                    "raw_evaluation": rnd.selection.raw_evaluation,
                }
                if rnd.selection.permutation_votes:
                    entry["permutation_votes"] = rnd.selection.permutation_votes
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Saved result to %s", path)
        return path
