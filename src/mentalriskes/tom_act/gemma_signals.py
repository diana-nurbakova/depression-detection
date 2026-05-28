"""Gemma signal generation pass (spec §5.2–5.6).

Eleven calls per round, all routed through the dispatcher (resumable, fully
logged):

  - 4 perspective views: self_a, self_b, observer_p, observer_pt  (temp 0.0)
  - 1 ToM-tier classification of the patient turn                  (temp 0.2)
  - 3 ToM-stance codings, one per therapist candidate              (temp 0.2)
  - 3 presencia codings, one per therapist candidate               (temp 0.0)

Stance and presencia are separate calls so neither anchors the other (§5.2).
"""

from __future__ import annotations

import logging

from . import prompts
from .constants import CANDIDATE_OPTIONS
from .data import Session, cumulative_dialogue, cumulative_patient, patient_turn
from .dispatcher import Dispatcher

logger = logging.getLogger(__name__)

# Per-signal temperatures (spec §5.3/5.4/5.5/5.6).
TEMPS = {
    "self_a": 0.0, "self_b": 0.0, "observer_p": 0.0, "observer_pt": 0.0,
    "tom_tier_patient": 0.2, "tom_stance": 0.2, "presencia": 0.0,
}


def _candidates_for(r, candidate_filter: str) -> list[int]:
    """Resolve which candidate options to code given the tier filter."""
    available = [o for o in CANDIDATE_OPTIONS if f"option_{o}" in r.options]
    if candidate_filter == "gold":
        return [r.gold_option] if r.gold_option in available else []
    if candidate_filter == "rejected":
        return [o for o in available if o != r.gold_option]
    return available


def generate_session(
    dispatcher: Dispatcher,
    client,
    session: Session,
    model_id: str,
    provider: str,
    limit_rounds: int | None = None,
    signals: set[str] | None = None,
    candidate_filter: str = "all",
) -> None:
    """Run the Gemma signal-generation pass for one session (resumable).

    Args:
        signals: which signal types to generate (default: all). Tier-controlled.
        candidate_filter: ``gold`` | ``rejected`` | ``all`` for stance/presencia.
    """
    signals = signals if signals is not None else set(TEMPS)

    for r in session.rounds:
        if limit_rounds is not None and r.round > limit_rounds:
            break
        t = r.round
        pt = patient_turn(session, t)

        # --- perspective views ------------------------------------------
        view_content = {
            "self_a": pt,
            "self_b": cumulative_patient(session, t),
            "observer_p": cumulative_patient(session, t),
            "observer_pt": cumulative_dialogue(session, t),
        }
        for sig, content in view_content.items():
            if sig not in signals:
                continue
            dispatcher.process(
                signal_type=sig,
                session_id=session.session_id,
                round_n=t,
                system_prompt=prompts.VIEW_SYSTEM[sig],
                user_prompt=prompts.build_view_user(sig, t, content),
                client=client,
                model_id=model_id,
                provider=provider,
                schema="view",
                temperature=TEMPS[sig],
            )

        # --- ToM-tier of the patient turn -------------------------------
        if "tom_tier_patient" in signals:
            dispatcher.process(
                signal_type="tom_tier_patient",
                session_id=session.session_id,
                round_n=t,
                system_prompt=prompts.TOM_TIER_SYSTEM,
                user_prompt=prompts.build_tom_tier_user(t, pt),
                client=client,
                model_id=model_id,
                provider=provider,
                schema="tom_tier_patient",
                temperature=TEMPS["tom_tier_patient"],
            )

        # --- stance + presencia per candidate ---------------------------
        need_cand = {"tom_stance", "presencia"} & signals
        for opt in (_candidates_for(r, candidate_filter) if need_cand else []):
            cand = r.candidate_text(opt)
            if "tom_stance" in signals:
                dispatcher.process(
                    signal_type="tom_stance",
                    session_id=session.session_id,
                    round_n=t,
                    system_prompt=prompts.TOM_STANCE_SYSTEM,
                    user_prompt=prompts.build_tom_stance_user(t, opt, cand, pt),
                    client=client,
                    model_id=model_id,
                    provider=provider,
                    schema="tom_stance",
                    candidate=opt,
                    temperature=TEMPS["tom_stance"],
                )
            if "presencia" in signals:
                dispatcher.process(
                    signal_type="presencia",
                    session_id=session.session_id,
                    round_n=t,
                    system_prompt=prompts.PRESENCIA_SYSTEM,
                    user_prompt=prompts.build_presencia_user(t, opt, cand, pt),
                    client=client,
                    model_id=model_id,
                    provider=provider,
                    schema="presencia",
                    candidate=opt,
                    temperature=TEMPS["presencia"],
                )

        logger.info("%s r%d: Gemma signals generated", session.session_id, t)
