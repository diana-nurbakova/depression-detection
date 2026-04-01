"""Main pipeline orchestration for MentalRiskES Task 1.

Modes:
  - trial: Run pipeline on local trial data for calibration/testing.
  - server: Run pipeline against competition server (GET/POST loop).
  - evaluate: Compare predictions to manual annotations.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assessors import AssessmentResult, assess_all_instruments
from .calibration import (
    apply_level_b_constraints,
    calibrate_scores,
    run_level_c_agent,
    _should_invoke_level_c,
    ConstraintViolation,
)
from .temporal import (
    SessionPredictionHistory,
    apply_temporal_aggregation,
)
from ..config import MentalRiskESConfig, RunConfig
from .data import ConversationStore, load_trial_data
from ..llm_client import LLMClient, HFInferenceClient, create_llm_client
from ..server import MentalRiskESClient

logger = logging.getLogger(__name__)


@dataclass
class RoundPrediction:
    """Prediction for a single session in a single round."""
    session_id: str
    round_number: int
    phq9: list[int]
    gad7: list[int]
    compact10: list[int]
    # Reasoning/metadata
    raw_assessments: dict[str, AssessmentResult] = field(default_factory=dict)
    consistency_warnings: list[dict] = field(default_factory=list)
    level_b_violations: list = field(default_factory=list)    # ConstraintViolation list
    level_c_corrections: list[dict] = field(default_factory=list)
    # Temporal aggregation metadata
    temporal_method: dict[str, str] = field(default_factory=dict)  # per-instrument method used
    temporal_confidence: dict = field(default_factory=dict)  # per-instrument confidence labels
    temporal_anomalous_rounds: dict = field(default_factory=dict)  # flagged rounds

    def to_submission_dict(self) -> dict:
        """Format for server POST."""
        return {
            "id": self.session_id,
            "round": self.round_number,
            "prediction": {
                "GAD-7": self.gad7,
                "PHQ-9": self.phq9,
                "CompACT-10": self.compact10,
            },
        }


class Pipeline:
    """MentalRiskES Task 1 pipeline orchestrator."""

    def __init__(self, config: MentalRiskESConfig) -> None:
        self.config = config
        self.store = ConversationStore()
        self.predictions: list[dict] = []  # all predictions for logging

        # Per-run, per-session prediction histories for temporal aggregation.
        # Keyed by (run_name, session_id).
        self._histories: dict[tuple[str, str], SessionPredictionHistory] = {}

        # Create output dirs
        config.data.output_dir.mkdir(parents=True, exist_ok=True)
        config.data.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.data.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_history(self, run_name: str, session_id: str) -> SessionPredictionHistory:
        """Get or create a SessionPredictionHistory for a (run, session) pair."""
        key = (run_name, session_id)
        if key not in self._histories:
            self._histories[key] = SessionPredictionHistory(session_id=session_id)
        return self._histories[key]

    def _uses_temporal(self, run_config: RunConfig) -> bool:
        """Check if any instrument uses temporal aggregation (not T0)."""
        return any(m != "T0" for m in [
            run_config.temporal_phq9,
            run_config.temporal_gad7,
            run_config.temporal_compact10,
        ])

    def _create_client(self, run_config: RunConfig | None = None) -> LLMClient | HFInferenceClient:
        """Create an LLM client, optionally overriding model from run config."""
        model = run_config.model if run_config else None
        return create_llm_client(self.config.llm, model_override=model)

    # Max conversation turns to include in prompt (first turn + last N).
    # Keeps prompt size manageable for long sessions.
    # 20 turns ≈ 10 exchanges ≈ ~4K chars of context.
    MAX_CONTEXT_TURNS = 20

    def _assess_session(
        self,
        session_id: str,
        run_config: RunConfig,
        client: LLMClient,
    ) -> RoundPrediction:
        """Run the full 3-instrument assessment for a single session.

        Calibration tier flow:
          Level A: prompt anchors injected into assessor prompts (if prompt_anchors=True)
          simple : per-item flat/band_aware correction (run_config.calibration)
          Level B: 7-rule psychometric constraint system (if level_b=True)
          Level C: LLM calibration agent, conditional on violations (if level_c=True)
        """
        context = self.store.get_context(session_id, max_turns=self.MAX_CONTEXT_TURNS)
        session = self.store.get_history(session_id)

        # Level A: prompt anchors injected at assessment time
        assessments = assess_all_instruments(
            client,
            context,
            use_few_shot=run_config.few_shot,
            use_prompt_anchors=run_config.prompt_anchors,
        )

        # Extract raw scores
        phq9_raw = assessments["PHQ-9"].scores
        gad7_raw = assessments["GAD-7"].scores
        compact10_raw = assessments["CompACT-10"].scores

        # Simple per-item calibration (flat / band_aware / none)
        phq9_cal = calibrate_scores(
            phq9_raw, "PHQ-9", run_config.calibration, run_config.calibration_params,
        )
        gad7_cal = calibrate_scores(
            gad7_raw, "GAD-7", run_config.calibration, run_config.calibration_params,
        )
        compact10_cal = calibrate_scores(
            compact10_raw, "CompACT-10", run_config.calibration, run_config.calibration_params,
        )

        # Level B: rule-based psychometric constraints
        level_b_violations: list[ConstraintViolation] = []
        if run_config.level_b:
            phq9_cal, gad7_cal, compact10_cal, level_b_violations = apply_level_b_constraints(
                phq9_cal, gad7_cal, compact10_cal,
            )
            for v in level_b_violations:
                logger.info(
                    "Session %s [Level B %s %s]: %s",
                    session_id, v.rule, v.severity, v.message[:120],
                )

        # Level C: LLM calibration agent (conditional on violations / total ceiling)
        level_c_corrections: list[dict] = []
        if run_config.level_c and _should_invoke_level_c(
            level_b_violations, compact10_cal, sum(phq9_cal), sum(gad7_cal),
        ):
            logger.info(
                "Session %s: invoking Level C calibration agent "
                "(%d Level B violations)",
                session_id, len(level_b_violations),
            )
            phq9_cal, gad7_cal, compact10_cal, level_c_corrections = run_level_c_agent(
                client, phq9_cal, gad7_cal, compact10_cal,
                level_b_violations, assessments,
            )

        # Build consistency warnings for logging (from Level B violations)
        warnings = [{"rule": v.rule, "message": v.message} for v in level_b_violations]

        # --- Temporal aggregation ---
        # Store this round's calibrated scores in the prediction matrix,
        # then aggregate across all rounds seen so far.
        temporal_meta: dict[str, str] = {}
        temporal_conf: dict = {}
        temporal_anom: dict = {}

        if self._uses_temporal(run_config):
            history = self._get_history(run_config.name, session_id)
            history.add_round(session.latest_round, phq9_cal, gad7_cal, compact10_cal)

            if history.n_rounds() >= 2:
                agg = apply_temporal_aggregation(
                    history,
                    phq9_method=run_config.temporal_phq9,
                    gad7_method=run_config.temporal_gad7,
                    compact10_method=run_config.temporal_compact10,
                    decay=run_config.temporal_decay,
                    stability_threshold=run_config.temporal_stability_threshold,
                    w1_threshold_factor=run_config.temporal_w1_threshold,
                    discard_anomalous=run_config.temporal_discard_anomalous,
                )
                phq9_cal = agg["phq9"]
                gad7_cal = agg["gad7"]
                compact10_cal = agg["compact10"]
                temporal_conf = agg.get("confidence_labels", {})
                temporal_anom = agg.get("anomalous_rounds", {})

                logger.info(
                    "Session %s temporal aggregation (%d rounds): "
                    "PHQ-9=%s(%d) GAD-7=%s(%d) CompACT-10=%s(%d)",
                    session_id, history.n_rounds(),
                    phq9_cal, sum(phq9_cal),
                    gad7_cal, sum(gad7_cal),
                    compact10_cal, sum(compact10_cal),
                )

            temporal_meta = {
                "PHQ-9": run_config.temporal_phq9,
                "GAD-7": run_config.temporal_gad7,
                "CompACT-10": run_config.temporal_compact10,
            }

        prediction = RoundPrediction(
            session_id=session_id,
            round_number=session.latest_round,
            phq9=phq9_cal,
            gad7=gad7_cal,
            compact10=compact10_cal,
            raw_assessments=assessments,
            consistency_warnings=warnings,
            level_b_violations=level_b_violations,
            level_c_corrections=level_c_corrections,
            temporal_method=temporal_meta,
            temporal_confidence=temporal_conf,
            temporal_anomalous_rounds=temporal_anom,
        )

        return prediction

    def run_trial(self) -> dict[str, list[RoundPrediction]]:
        """
        Run pipeline on local trial data.

        Returns:
            dict mapping run_name -> list of RoundPredictions across all rounds.
        """
        trial_data = load_trial_data(self.config.data.trial_dir)
        all_results: dict[str, list[RoundPrediction]] = {}

        for run_config in self.config.runs:
            logger.info("=== Run: %s ===", run_config.name)
            client = self._create_client(run_config)
            run_predictions: list[RoundPrediction] = []

            # Reset store and prediction histories for each run
            self.store = ConversationStore()
            self._histories = {
                k: v for k, v in self._histories.items()
                if k[0] != run_config.name
            }

            for round_n in sorted(trial_data.keys()):
                logger.info("--- Round %d ---", round_n)
                messages = trial_data[round_n]

                # Update conversation store
                self.store.update_from_server_response(messages)

                # Assess each session
                for session_id in messages:
                    pred = self._assess_session(session_id, run_config, client)
                    run_predictions.append(pred)

                    logger.info(
                        "Session %s R%d: PHQ-9=%s(%d) GAD-7=%s(%d) CompACT-10=%s(%d)",
                        session_id, round_n,
                        pred.phq9, sum(pred.phq9),
                        pred.gad7, sum(pred.gad7),
                        pred.compact10, sum(pred.compact10),
                    )

                    # Log detailed results
                    self._log_prediction(pred, run_config.name)

            all_results[run_config.name] = run_predictions
            logger.info("Run %s complete: %d predictions, LLM stats: %s",
                        run_config.name, len(run_predictions), client.stats)

        return all_results

    def run_server(self) -> None:
        """
        Run pipeline against competition server in GET/POST loop.

        Round advancement requires ALL 3 POST submissions (runs 0, 1, 2).
        If fewer runs are configured, we still submit all configured runs.

        NOTE: If participating in both tasks, the caller must orchestrate
        task1 GET → task1 POSTs → task2 GET → task2 POSTs before the
        server advances to the next round.
        """
        server = MentalRiskESClient.from_config(self.config.server, task="task1")
        emissions = self._get_emissions_dict()

        # Pre-create one LLM client per run (reused across rounds)
        run_clients = {
            rc.name: self._create_client(rc)
            for rc in self.config.runs
        }

        messages = server.get_messages()
        if not messages:
            logger.warning("No messages received from server")
            return

        while messages:
            round_number = next(iter(messages.values()))["round"]
            n_sessions = len(messages)
            logger.info("=== Server Round %d (%d sessions) ===", round_number, n_sessions)

            # Update conversation store
            self.store.update_from_server_response(messages)

            # Save raw messages
            round_path = self.config.data.output_dir / f"round_{round_number}_messages.json"
            with open(round_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)

            # Build predictions for each run
            predictions_per_run: list[list[dict]] = []

            for run_config in self.config.runs:
                client = run_clients[run_config.name]
                run_preds = []

                for session_id in messages:
                    pred = self._assess_session(session_id, run_config, client)
                    run_preds.append(pred.to_submission_dict())
                    self._log_prediction(pred, run_config.name)

                predictions_per_run.append(run_preds)

            # Submit all runs (runs 0..N-1)
            server.submit_all_runs(
                predictions_per_run, emissions,
                save_dir=self.config.data.output_dir / "predictions",
                round_number=round_number,
            )

            logger.info("Round %d complete: submitted %d runs x %d sessions",
                        round_number, len(predictions_per_run), n_sessions)

            # Next round
            messages = server.get_messages()

        logger.info("All rounds processed.")

    def _log_prediction(self, pred: RoundPrediction, run_name: str) -> None:
        """Log a prediction to the predictions log file."""
        if not self.config.pipeline.log_llm_outputs:
            return

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run": run_name,
            "session_id": pred.session_id,
            "round": pred.round_number,
            "phq9": pred.phq9,
            "gad7": pred.gad7,
            "compact10": pred.compact10,
            "phq9_total": sum(pred.phq9),
            "gad7_total": sum(pred.gad7),
            "compact10_total": sum(pred.compact10),
            "consistency_warnings": pred.consistency_warnings,
            "level_b_violations": [
                {
                    "rule": v.rule,
                    "severity": v.severity,
                    "message": v.message,
                    "correction_applied": v.correction_applied,
                    "correction_detail": v.correction_detail,
                }
                for v in pred.level_b_violations
            ],
            "level_c_corrections": pred.level_c_corrections,
            "temporal_method": pred.temporal_method,
            "temporal_confidence": pred.temporal_confidence,
            "temporal_anomalous_rounds": pred.temporal_anomalous_rounds,
        }

        # Add raw LLM outputs if enabled
        if self.config.pipeline.log_llm_outputs:
            for instrument, result in pred.raw_assessments.items():
                log_entry[f"{instrument}_steps"] = result.steps
                if result.error:
                    log_entry[f"{instrument}_error"] = result.error
                if result.labels:
                    log_entry[f"{instrument}_labels"] = result.labels
                if result.label_mismatches:
                    log_entry[f"{instrument}_label_mismatches"] = result.label_mismatches

        log_path = self.config.data.log_dir / f"predictions_{run_name}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _get_emissions_dict(self) -> dict:
        """Return a CodeCarbon-compatible emissions dict."""
        try:
            import pandas as pd
            emissions_df = pd.read_csv("emissions.csv")
            cols = [
                "duration", "emissions", "cpu_energy", "gpu_energy",
                "ram_energy", "energy_consumed", "cpu_count", "gpu_count",
                "cpu_model", "gpu_model", "ram_total_size", "country_iso_code",
            ]
            raw = emissions_df.iloc[-1][cols].to_dict()
            string_cols = {"cpu_model", "gpu_model", "country_iso_code"}
            result = {}
            for k, v in raw.items():
                if k in string_cols:
                    result[k] = v if isinstance(v, str) else ""
                elif isinstance(v, float) and (v != v or abs(v) == float("inf")):
                    result[k] = 0.0
                else:
                    result[k] = v
            return result
        except Exception:
            return {
                "duration": 0.0, "emissions": 0.0, "cpu_energy": 0.0,
                "gpu_energy": 0.0, "ram_energy": 0.0, "energy_consumed": 0.0,
                "cpu_count": 0, "gpu_count": 0, "cpu_model": "",
                "gpu_model": "", "ram_total_size": 0.0,
                "country_iso_code": self.config.pipeline.codecarbon_country,
            }


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the pipeline."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
