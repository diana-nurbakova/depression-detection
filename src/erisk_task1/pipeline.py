"""Main pipeline runner — orchestrates the full conversation + assessment flow."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .assessors import run_all_assessors
from .config import PipelineConfig
from .justificator import run_justificator
from .linguistic import compute_cumulative_features, detect_persona_profile, extract_features
from .llm_client import LLMClient, make_clients
from .models import (
    BDI_ITEMS,
    ConversationTurn,
    ItemState,
    OrchestratorGuidance,
    PersonaResult,
    SeverityBand,
    TopicArea,
    score_to_band,
)
from .orchestrator import Orchestrator
from .persona import PersonaModel
from .scoring import run_scoring_pipeline
from .submission import format_interactions, format_results, save_internal_results

logger = logging.getLogger(__name__)


def run_persona_conversation(
    persona_id: int,
    persona_model: PersonaModel,
    clients: dict[str, LLMClient],
    config: PipelineConfig,
) -> PersonaResult:
    """Run the full pipeline for a single persona.

    1. Conversation loop (interviewer ↔ persona)
    2. Assessment after each N turns
    3. Termination decision
    4. Post-conversation: 2-pass scoring + justificator
    """
    persona_str = f"{persona_id:02d}"
    logger.info("=" * 60)
    logger.info("Starting persona %s", persona_str)
    logger.info("=" * 60)

    t_start = time.monotonic()

    # Load persona adapter
    persona_model.load_adapter(persona_id)

    # Initialize orchestrator
    orch = Orchestrator(
        interviewer_client=clients["interviewer"],
        assessor_client=clients["assessor"],
        orchestrator_client=clients["orchestrator"],
        max_turns=config.execution.max_turns,
        min_turns=config.execution.min_turns,
        assess_every_n=config.execution.assess_every_n_turns,
        parallel_assessors=config.execution.parallel_assessors,
        termination_confidence=config.execution.termination_confidence,
    )

    # Initial guidance
    guidance = OrchestratorGuidance(
        decision="CONTINUE",
        next_topic=TopicArea.EMOTIONAL_STATE,
        suggested_angle="Start with a warm greeting and ask how they've been doing.",
    )

    turn_number = 0
    termination_reason = ""

    while True:
        turn_number += 1
        logger.info("--- Turn %d/%d ---", turn_number, config.execution.max_turns)

        # 1. Generate interviewer message
        interviewer_msg = orch.generate_interviewer_message(guidance, turn_number)
        logger.info("Interviewer: %s", interviewer_msg[:100])

        orch.conversation.append(ConversationTurn(
            role="user",
            message=interviewer_msg,
            turn_number=turn_number,
        ))

        # 2. Get persona response
        conv_for_persona = [
            {"role": t.role, "content": t.message} for t in orch.conversation
        ]
        persona_response = persona_model.generate(conv_for_persona)
        logger.info("Persona: %s", persona_response[:100])

        # 3. Process persona response (linguistic features)
        orch.process_persona_response(persona_response, turn_number)

        # 4. Run assessors (every N turns)
        if turn_number % config.execution.assess_every_n_turns == 0:
            orch.process_turn_assessment(turn_number)

        # 5. Check termination
        should_stop, reason = orch.should_terminate(turn_number)
        if should_stop:
            termination_reason = reason
            logger.info("Terminating: %s", reason)
            break

        # 6. Get orchestrator guidance for next turn
        guidance = orch.run_orchestrator_reasoning(turn_number)

        if guidance.decision == "TERMINATE":
            termination_reason = "Orchestrator decision"
            logger.info("Orchestrator decided to terminate")
            break

        # Update topic tracking
        if guidance.next_topic and guidance.next_topic not in orch.topics_covered:
            orch.topics_covered.append(guidance.next_topic)
            if guidance.next_topic in orch.topics_remaining:
                orch.topics_remaining.remove(guidance.next_topic)

    # --- Post-conversation ---
    logger.info("Post-conversation processing for persona %s", persona_str)

    # Run final assessment if not done on last turn
    if turn_number % config.execution.assess_every_n_turns != 0:
        orch.process_turn_assessment(turn_number)

    # 2-pass scoring
    scoring_result = run_scoring_pipeline(
        orch.assessor_outputs, orch.features_history
    )

    # Justificator
    justificator_output = run_justificator(
        client=clients["justificator"],
        persona_id=persona_str,
        transcript=orch.get_transcript(),
        assessor_outputs=orch.assessor_outputs,
        item_scores=scoring_result["item_scores"],
        pass2_total=scoring_result["pass2_total"],
        pass2_band=scoring_result["pass2_band"],
        features_history=orch.features_history,
    )

    # Final scores
    final_total = justificator_output.final_total
    final_band = justificator_output.final_band

    # Top-4 symptoms (canonical names)
    top4_names = []
    for sym in justificator_output.top_4_symptoms:
        name = sym.get("item_name", "")
        item_id = sym.get("item_id")
        # Use canonical name
        if item_id and item_id in BDI_ITEMS:
            name = BDI_ITEMS[item_id]
        if name:
            top4_names.append(name)

    elapsed = time.monotonic() - t_start
    logger.info(
        "Persona %s complete: BDI=%d (%s), turns=%d, %.1fs",
        persona_str, final_total, final_band.value, turn_number, elapsed,
    )

    return PersonaResult(
        persona_id=persona_str,
        persona_number=persona_id,
        conversation=orch.conversation,
        assessor_outputs=orch.assessor_outputs,
        linguistic_features_history=orch.features_history,
        pass1_total=scoring_result["pass1_total"],
        pass2_total=scoring_result["pass2_total"],
        final_total=final_total,
        final_band=final_band,
        top_4_symptoms=top4_names,
        justificator_output=justificator_output,
        item_scores=scoring_result["item_scores"],
    )


def run_pipeline(config: PipelineConfig) -> list[PersonaResult]:
    """Run the full Task 1 pipeline for all configured personas."""
    logger.info("Starting Task 1 pipeline (run %d, type=%s)", config.run_id, config.run_type)

    # Create LLM clients
    clients = make_clients(config)

    # Create persona model
    persona_model = PersonaModel(config.persona)
    persona_model.load_base()

    base_dir = Path(config.logging.output_dir)
    results = []
    for persona_id in config.persona_ids:
        try:
            result = run_persona_conversation(
                persona_id, persona_model, clients, config
            )
            results.append(result)

            # Save per-persona outputs immediately (crash-safe)
            persona_dir = base_dir / f"persona{persona_id:02d}"
            save_persona_run(result, config.run_id, persona_dir)

        except Exception as e:
            logger.error("Failed processing persona %02d: %s", persona_id, e, exc_info=True)

    # Unload persona model
    persona_model.unload()

    # Log client stats
    for name, client in clients.items():
        logger.info("Client stats [%s]: %s", name, client.stats)

    return results


def save_persona_run(
    result: PersonaResult,
    run_id: int,
    persona_dir: Path,
):
    """Save all outputs for a single persona run.

    Output structure:
      runs/task1/persona{ID}/
        interactions_{run_id}.json   -- official submission format
        results_{run_id}.json        -- official submission format
        internal_{run_id}.json       -- detailed logs
        conversation_{run_id}.json   -- full conversation log
    """
    persona_dir.mkdir(parents=True, exist_ok=True)

    # Official submission: interactions
    interactions = format_interactions([result])
    interactions_path = persona_dir / f"interactions_{run_id}.json"
    with open(interactions_path, "w") as f:
        json.dump(interactions, f, indent=2)

    # Official submission: results
    erisk_results = format_results([result])
    results_path = persona_dir / f"results_{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(erisk_results, f, indent=2)

    # Internal detailed results
    save_internal_results(result, persona_dir, suffix=f"_{run_id}")

    logger.info(
        "Saved persona %s run %d: BDI=%d (%s) -> %s",
        result.persona_id, run_id, result.final_total,
        result.final_band.value, persona_dir,
    )


def merge_submission_files(
    base_dir: Path,
    persona_ids: list[int],
    run_id: int,
):
    """Merge per-persona files into single submission files.

    Produces:
      runs/task1/interactions_{run_id}.json
      runs/task1/results_{run_id}.json
    """
    all_interactions = []
    all_results = []

    for pid in sorted(persona_ids):
        persona_dir = base_dir / f"persona{pid:02d}"

        interactions_path = persona_dir / f"interactions_{run_id}.json"
        if interactions_path.exists():
            with open(interactions_path) as f:
                all_interactions.extend(json.load(f))

        results_path = persona_dir / f"results_{run_id}.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results.extend(json.load(f))

    # Save merged files
    base_dir.mkdir(parents=True, exist_ok=True)

    merged_interactions = base_dir / f"interactions_{run_id}.json"
    with open(merged_interactions, "w") as f:
        json.dump(all_interactions, f, indent=2)
    logger.info("Merged interactions (%d personas): %s", len(all_interactions), merged_interactions)

    merged_results = base_dir / f"results_{run_id}.json"
    with open(merged_results, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Merged results (%d personas): %s", len(all_results), merged_results)
