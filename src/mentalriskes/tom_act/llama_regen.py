"""Llama regeneration pass (spec §2.4, §5.8).

Recovers the round-level state snapshots the live submission never persisted.
For each session we evolve the Task-2 ``SharedState`` round by round, driven by
the **actual/gold** therapist turn the patient really reacted to, and emit per
round:

  - state-update output: ``procesos_act`` (6 hexaflex incl. ``yo_como_contexto``),
    ``fase_terapeutica``, ``estado_emocional`` — via the reused Task-2 state
    tracker prompt (``task2/prompts.py``).
  - assessor item-vectors: PHQ-9 / GAD-7 / CompACT-10, scored on the **full
    cumulative dialogue through round t** — via the reused Task-1 assessor
    prompts (``task1/assessors.py``).

Four Llama calls per round, each persisted to its own JSONL for sub-call-level
resume. The assembled ``llama_state_snapshot`` is a derived aggregation
(``aggregator.py``), consistent with spec §6.6.

The selection step is intentionally NOT run — no RQ needs it.
"""

from __future__ import annotations

import json
import logging

from ..task1 import assessors
from ..task2.models import SharedState
from ..task2.prompts import STATE_UPDATE_SYSTEM, build_state_update_user
from . import prompts as tom_prompts
from .data import Session, cumulative_dialogue
from .dispatcher import Dispatcher
from .recovery import assessor_scores

logger = logging.getLogger(__name__)

# Sub-call signal types (separate JSONL each; assembled later).
SIG_STATE_UPDATE = "llama_state_update"
SIG_ASSESS = {
    "PHQ-9": "llama_assess_phq9",
    "GAD-7": "llama_assess_gad7",
    "CompACT-10": "llama_assess_compact10",
}
# Combined-mode signal: all three instruments scored in one call (cost lever).
SIG_ASSESS_COMBINED = "llama_assess_combined"


def _extract_scores(parsed: dict | None, instrument: str) -> list[int] | None:
    """Pull a clipped item-score array from a parsed assessor response.

    Delegates to the shared ``recovery.assessor_scores`` so generation,
    validation, and aggregation agree on what counts as a complete vector.
    """
    if not parsed:
        return None
    return assessor_scores(parsed, instrument)


def regenerate_session(
    dispatcher: Dispatcher,
    client,
    session: Session,
    model_id: str,
    provider: str,
    lang: str = "es",
    use_prompt_anchors: bool = False,
    limit_rounds: int | None = None,
    assessor_mode: str = "combined",
) -> None:
    """Run the Llama regeneration pass for one session (resumable).

    ``assessor_mode``:
      - ``combined`` (default): score PHQ-9+GAD-7+CompACT-10 in ONE call/round
        (2 Llama calls/round total → ~1,136 over the corpus).
      - ``per_instrument``: three task1 CoT calls/round (4 total → ~2,272),
        higher per-item calibration fidelity at higher cost.
    """
    state = SharedState()
    system = STATE_UPDATE_SYSTEM[lang]

    for r in session.rounds:
        if limit_rounds is not None and r.round > limit_rounds:
            break

        # --- state update (procesos_act / fase / emotion) ---------------
        prev_state_json = json.dumps(state.to_state_json(), ensure_ascii=False, indent=2)
        user = build_state_update_user(
            previous_state_json=prev_state_json,
            selected_response_text=r.therapist_response,  # actual delivered turn
            patient_input=r.patient_input,
            round_number=r.round,
            lang=lang,
        )
        parsed = dispatcher.process(
            signal_type=SIG_STATE_UPDATE,
            session_id=session.session_id,
            round_n=r.round,
            system_prompt=system,
            user_prompt=user,
            client=client,
            model_id=model_id,
            provider=provider,
            schema=None,
        )
        if parsed:
            state.update_from_llm(parsed)
        else:
            logger.warning("%s r%d: state-update parse failed; state carried forward",
                           session.session_id, r.round)

        # --- assessor views on full cumulative dialogue -----------------
        dialogue = cumulative_dialogue(session, r.round)
        if assessor_mode == "combined":
            # One call scores all three instruments (cost lever); "view" schema.
            dispatcher.process(
                signal_type=SIG_ASSESS_COMBINED,
                session_id=session.session_id,
                round_n=r.round,
                system_prompt=tom_prompts.LLAMA_ASSESSOR_SYSTEM,
                user_prompt=tom_prompts.build_llama_assessor_user(r.round, dialogue),
                client=client,
                model_id=model_id,
                provider=provider,
                schema="view",
                temperature=0.0,
            )
        else:
            for instrument, sig in SIG_ASSESS.items():
                prompt = assessors.build_prompt(
                    instrument, dialogue, use_few_shot=True,
                    use_prompt_anchors=use_prompt_anchors,
                )
                dispatcher.process(
                    signal_type=sig,
                    session_id=session.session_id,
                    round_n=r.round,
                    system_prompt="",          # assessor prompt is self-contained
                    user_prompt=prompt,
                    client=client,
                    model_id=model_id,
                    provider=provider,
                    schema=f"assessor:{instrument}",   # require full score vector
                )
        logger.info("%s r%d: snapshot regenerated (phase=%s)",
                    session.session_id, r.round, state.fase_terapeutica)
