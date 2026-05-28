"""Priority tiers and run sequencing (spec v0.6 §15).

Five-tier structure (T0 pilot → T1 → T2 → T3 → T4 conditional). Each tier is
self-sufficient for a publishable subset of the analysis; downstream tiers can be
cut for budget/time. The §6.4 resume protocol makes pilot (T0) records count
toward T1/T2 without re-running (same prompt hashes).

Cost/time control works by (a) restricting which Gemma signal types run, and
(b) for stance/presencia, restricting which candidates are coded (gold only vs
the two rejected). The Llama regeneration is bundled into T1 (it is upstream of
RQ2/RQ3/RQ4 and the case studies).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import GEMMA_VIEW_SIGNALS, GEMMA_TIER_SIGNAL


@dataclass(frozen=True)
class Tier:
    name: str
    run_llama: bool
    gemma_signals: tuple[str, ...]         # which Gemma signal types to generate
    candidate_filter: str = "all"          # gold | rejected | all (stance/presencia)
    sessions: tuple[str, ...] | None = None  # None = all configured sessions
    max_rounds: int | None = None          # None = all rounds
    description: str = ""


# Signal-type groupings.
_VIEWS = tuple(GEMMA_VIEW_SIGNALS)                       # self_a, self_b, observer_p, observer_pt
_ALL_GEMMA = _VIEWS + (GEMMA_TIER_SIGNAL, "tom_stance", "presencia")

TIERS: dict[str, Tier] = {
    # Pilot: S07, 5 rounds, all 7 signal types (stance/presencia gold-only) = 35 calls.
    "T0": Tier(
        name="T0", run_llama=False,
        gemma_signals=_ALL_GEMMA, candidate_filter="gold",
        sessions=("S07",), max_rounds=5,
        description="Pilot on S07 (5 rounds x 7 Gemma signal types); ~35 calls.",
    ),
    # Minimum viable RQ1: Llama regen + Self-A + Observer-P + ToM-tier (all sessions).
    "T1": Tier(
        name="T1", run_llama=True,
        gemma_signals=("self_a", "observer_p", GEMMA_TIER_SIGNAL),
        candidate_filter="all",
        description="Llama regeneration + Self-A + Observer-P + ToM-tier; ~2,272 calls.",
    ),
    # RQ2 + RQ3 extension: remaining views + stance/presencia on the gold candidate.
    "T2": Tier(
        name="T2", run_llama=False,
        gemma_signals=("self_b", "observer_pt", "tom_stance", "presencia"),
        candidate_filter="gold",
        description="Self-B + Observer-PT + ToM-stance/presencia x gold candidate; ~2,272 calls.",
    ),
    # RQ4 completion: stance/presencia on the two rejected candidates.
    "T3": Tier(
        name="T3", run_llama=False,
        gemma_signals=("tom_stance", "presencia"),
        candidate_filter="rejected",
        description="ToM-stance/presencia x 2 rejected candidates; ~2,272 calls.",
    ),
}

TIER_ORDER = ["T0", "T1", "T2", "T3"]
