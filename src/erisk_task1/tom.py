"""Theory of Mind (ToM) perception tracker for eRisk Task 1.

Tracks two evolving perception profiles per conversation turn:
- E(t): expressed symptom profile — what the assessor reads from the transcript
- I(t): interviewer attention profile — which BDI-II domains the interviewer probed

Computes Wasserstein distances (requires the POT library; degrades gracefully if absent):
- W_self(t, k): profile shift over the last k turns (self-disclosure trajectory)
- W_align(t):   interviewer–persona alignment gap
- W_accuracy(t): assessment accuracy vs. ground truth (when available)

Also provides ToM-stratified coverage guidance for the orchestrator:
which of the three ToM categories (Somatic/Low-ToM, Cognitive-ToM, Affective-ToM)
are underexplored so the interviewer can be steered accordingly.

References:
- Spec: specs/task-1/tom/tom_wasserstein_task1_spec.md
- Analysis prototype: specs/task-1/tom/tom_wasserstein_analysis.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import BDI_ITEMS, ItemScore, ItemState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BDI-II ToM categorization (inferential demand axis)
# ─────────────────────────────────────────────────────────────────────────────

# Categorises BDI-II items by the inferential demand they place on an external
# assessor — NOT by patient response co-variance (the standard factor analysis).
# This is the novel ToM reframing introduced in the spec.
TOM_CATEGORIES: dict[str, list[int]] = {
    "Somatic_LowToM": [11, 15, 16, 18, 20, 21],   # bodily / vegetative; low inference
    "Cognitive_ToM":  [2, 3, 5, 6, 8, 13, 19],     # beliefs, judgements, evaluation
    "Affective_ToM":  [1, 4, 7, 9, 10, 12, 14, 17], # emotional intensity & quality
}

# Reverse: item_id → ToM category name
ITEM_TO_TOM: dict[int, str] = {
    item: cat
    for cat, items in TOM_CATEGORIES.items()
    for item in items
}

# Human-readable ToM category labels for orchestrator guidance
TOM_CATEGORY_LABELS: dict[str, str] = {
    "Somatic_LowToM": "Somatic / low-inference items (sleep, appetite, energy, fatigue)",
    "Cognitive_ToM":  "Cognitive items (pessimism, guilt, self-criticism, indecision)",
    "Affective_ToM":  "Affective items (sadness, loss of pleasure, self-dislike, worthlessness)",
}

# BDI-II item descriptions (abbreviated) — used in coverage gap explanations
BDI_SHORT: dict[int, str] = {
    1: "Sadness", 2: "Pessimism", 3: "Past failure", 4: "Loss of pleasure",
    5: "Guilty feelings", 6: "Punishment feelings", 7: "Self-dislike",
    8: "Self-criticalness", 9: "Suicidal thoughts", 10: "Crying",
    11: "Agitation", 12: "Loss of interest", 13: "Indecisiveness",
    14: "Worthlessness", 15: "Loss of energy", 16: "Sleep changes",
    17: "Irritability", 18: "Appetite changes", 19: "Concentration difficulty",
    20: "Tiredness/fatigue", 21: "Loss of interest in sex",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. BDI-II factor structure (Dozois et al. 1998; Whisman et al. 2000)
#    Used for the clinical ground metric in Wasserstein computation.
# ─────────────────────────────────────────────────────────────────────────────

BDI_FACTORS: dict[str, list[int]] = {
    "Cognitive":            [2, 3, 5, 6, 7, 8, 13, 14],
    "Affective":            [1, 4, 9, 10, 12, 17],
    "Somatic_Performance":  [15, 19, 20],
    "Somatic_Vegetative":   [11, 16, 18, 21],
}

ITEM_TO_FACTOR: dict[int, str] = {
    item: fac
    for fac, items in BDI_FACTORS.items()
    for item in items
}

# Factor pairs with clinical distance = 1 (adjacent).
# All other cross-factor pairs have distance 2.
_ADJACENT_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"Cognitive", "Affective"}),
    frozenset({"Somatic_Performance", "Somatic_Vegetative"}),
})


# ─────────────────────────────────────────────────────────────────────────────
# 3. Clinical cost matrix
# ─────────────────────────────────────────────────────────────────────────────

_COST_MATRIX: Optional[np.ndarray] = None


def get_cost_matrix() -> np.ndarray:
    """Return (and cache) the 21×21 clinical factor distance matrix.

    Cost(i, j):
      0   — same factor
      1   — adjacent factors (Cognitive↔Affective, Somatic-Perf↔Somatic-Veg)
      2   — all other cross-factor pairs
    """
    global _COST_MATRIX
    if _COST_MATRIX is not None:
        return _COST_MATRIX

    n = 21
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fi = ITEM_TO_FACTOR[i + 1]
            fj = ITEM_TO_FACTOR[j + 1]
            if fi == fj:
                C[i, j] = 0.0
            elif frozenset({fi, fj}) in _ADJACENT_PAIRS:
                C[i, j] = 1.0
            else:
                C[i, j] = 2.0
    _COST_MATRIX = C
    return C


# ─────────────────────────────────────────────────────────────────────────────
# 4. Wasserstein distance functions (require POT; degrade gracefully)
# ─────────────────────────────────────────────────────────────────────────────


def _l1_normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    """L1-normalise vec to a probability distribution. Returns None if all-zero."""
    total = float(vec.sum())
    if total < 1e-10:
        return None
    return (vec / total).astype(np.float64)


def wasserstein_balanced(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> Optional[float]:
    """Balanced W₁ between two BDI-II profiles (both L1-normalised).

    Captures profile *shape* accuracy — relative symptom distribution,
    independent of total severity mass.

    Returns None if POT is unavailable, either profile is all-zero,
    or computation fails.
    """
    try:
        import ot
    except ImportError:
        return None

    a = _l1_normalize(np.maximum(source, 0.0))
    b = _l1_normalize(np.maximum(target, 0.0))
    if a is None or b is None:
        return None

    try:
        dist = ot.emd2(a, b, cost_matrix.astype(np.float64))
        return float(dist)
    except Exception as e:
        logger.debug("wasserstein_balanced failed: %s", e)
        return None


def wasserstein_unbalanced(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.5,
    reg_m: float = 0.5,
) -> Optional[float]:
    """Unbalanced Sinkhorn divergence between two BDI-II profiles.

    Preserves mass difference — captures both profile shape and absolute
    severity errors simultaneously.

    Returns None if POT is unavailable or computation fails.
    """
    try:
        import ot
    except ImportError:
        return None

    p = np.maximum(source, 0.0).astype(np.float64) + 1e-10
    q = np.maximum(target, 0.0).astype(np.float64) + 1e-10

    try:
        dist = ot.unbalanced.sinkhorn_unbalanced2(
            p, q, cost_matrix.astype(np.float64), reg=reg, reg_m=reg_m
        )
        return float(dist)
    except Exception as e:
        logger.debug("wasserstein_unbalanced failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interviewer attention heuristic (keyword-based; working-notes version)
# ─────────────────────────────────────────────────────────────────────────────

INTERVIEWER_KEYWORDS: dict[int, list[str]] = {
    1:  ["sad", "down", "unhappy", "depressed", "miserable", "blue", "feeling low"],
    2:  ["future", "hope", "hopeless", "pessimis", "discourag", "worried about"],
    3:  ["failure", "failed", "loser", "disappointment", "mistake", "accomplish"],
    4:  ["enjoy", "pleasure", "interest", "fun", "satisfied", "look forward"],
    5:  ["guilt", "guilty", "fault", "blame", "deserve", "regret", "responsible"],
    6:  ["punish", "punishment", "punished", "pay for"],
    7:  ["dislike myself", "hate myself", "disappointed in myself", "self-image"],
    8:  ["critic", "criticis", "hard on yourself", "judge yourself", "self-blame"],
    9:  ["suicide", "suicidal", "kill myself", "end my life", "not worth living",
         "die", "harm yourself"],
    10: ["cry", "crying", "tears", "weep"],
    11: ["restless", "agitat", "tense", "on edge", "wound up", "fidget", "pace"],
    12: ["interest", "care", "bother", "engaged", "motivat"],
    13: ["decid", "decision", "choice", "indecis", "make up your mind"],
    14: ["worthless", "worth nothing", "useless", "no good", "valueless"],
    15: ["energy", "tired", "fatigue", "exhausted", "run down"],
    16: ["sleep", "sleeping", "insomnia", "wake up", "oversleep", "bed", "rest"],
    17: ["irritab", "annoyed", "angry", "frustrated", "temper", "mood"],
    18: ["appetite", "eating", "food", "weight", "hunger", "meal"],
    19: ["concentrat", "focus", "attention", "think clearly", "mind wander"],
    20: ["fatigue", "fatigued", "worn out", "drained", "exhaustion"],
    21: ["sex", "sexual", "libido", "intimacy", "physical relationship"],
}


def classify_question_keywords(question: str) -> np.ndarray:
    """Map an interviewer question to a 21-dim BDI-II attention vector.

    Returns a probability distribution (sum=1) over items.
    Uses keyword matching heuristic (working-notes version; spec §4.2 alt.).
    General/rapport questions → uniform prior.
    """
    q_lower = question.lower()
    weights = np.zeros(21, dtype=np.float32)

    for item_id, keywords in INTERVIEWER_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                weights[item_id - 1] += 1.0
                break  # count each item at most once per question

    total = weights.sum()
    if total < 1e-10:
        return np.ones(21, dtype=np.float32) / 21.0  # uniform: general question
    return weights / total


# ─────────────────────────────────────────────────────────────────────────────
# 6. TomPerceptionTracker
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TomPerceptionTracker:
    """Per-persona tracker for dual perception profiles and Wasserstein metrics.

    Usage in the pipeline:
      tracker = TomPerceptionTracker(guide_interviewer=True)
      # After each assessor run:
      tracker.update_expressed(turn_number, item_scores)
      tracker.compute_wasserstein_metrics(turn_number)
      # After each interviewer message:
      tracker.update_interviewer(turn_number, interviewer_message)
      # For orchestrator reasoning:
      context = tracker.get_orchestrator_context()
      # For saving:
      summary = tracker.to_summary_dict()
    """

    guide_interviewer: bool = True
    """If True, get_orchestrator_context() returns gap data for the orchestrator."""

    ground_truth: Optional[np.ndarray] = None
    """Optional 21-dim ground-truth BDI-II profile. Enables W_accuracy computation."""

    cost_metric: str = "clinical"
    """Ground metric: 'clinical' (default). 'embedding' and 'all' reserved for Phase 2."""

    # Turn → 21-dim float32 arrays
    E_profiles: dict[int, np.ndarray] = field(default_factory=dict)
    """Expressed symptom profiles: E(t) from assessor item scores."""

    I_profiles: dict[int, np.ndarray] = field(default_factory=dict)
    """Interviewer attention profiles: I(t) from keyword classification."""

    # Wasserstein metrics
    W_self: dict[int, dict[int, float]] = field(default_factory=dict)
    """W_self[t][k] = W₁(E(t), E(t-k)) for k in {1, 2, 5}."""

    W_align: dict[int, float] = field(default_factory=dict)
    """W_align[t] = W₁(E(t), I(t)) — interviewer–persona alignment gap."""

    W_accuracy: dict[int, float] = field(default_factory=dict)
    """W_accuracy[t] = W₁(E(t), G) — accuracy vs. ground truth (if available)."""

    _pot_available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import ot  # noqa: F401
            self._pot_available = True
            logger.debug("TomPerceptionTracker: POT available — Wasserstein enabled")
        except ImportError:
            self._pot_available = False
            logger.warning(
                "TomPerceptionTracker: POT library not installed. "
                "Install with: pip install POT. "
                "Wasserstein distances disabled; ToM coverage tracking still active."
            )

    # ── Profile builders ──────────────────────────────────────────────────────

    def update_expressed(
        self, turn_number: int, item_scores: dict[int, ItemScore]
    ) -> None:
        """Build E(t) from the current assessor item scores.

        SCORED items contribute their score (0–3); NO_EVIDENCE and
        EVIDENCE_OF_ABSENCE items contribute 0.
        """
        vec = np.zeros(21, dtype=np.float32)
        for item_id in range(1, 22):
            if item_id not in item_scores:
                continue
            score_obj = item_scores[item_id]
            if score_obj.state == ItemState.SCORED and score_obj.score is not None:
                vec[item_id - 1] = float(score_obj.score)
        self.E_profiles[turn_number] = vec
        logger.debug(
            "ToM E(t=%d): total_mass=%.1f, non_zero=%d",
            turn_number, float(vec.sum()), int((vec > 0).sum()),
        )

    def update_interviewer(
        self, turn_number: int, interviewer_question: str
    ) -> None:
        """Build the cumulative interviewer attention profile I(t).

        I(t) is the running average of per-turn keyword classifications
        up to and including turn t (spec §4.2).
        """
        q_weights = classify_question_keywords(interviewer_question)

        # Cumulative running average across all interviewer turns seen so far
        existing_turns = sorted(t for t in self.I_profiles if t < turn_number)
        if not existing_turns:
            cumulative = q_weights.copy()
        else:
            prev_I = self.I_profiles[existing_turns[-1]]
            n = len(existing_turns)
            cumulative = (prev_I * n + q_weights) / (n + 1)

        self.I_profiles[turn_number] = cumulative

        active_items = [
            BDI_SHORT.get(i + 1, str(i + 1))
            for i in range(21) if q_weights[i] > 0
        ]
        logger.debug(
            "ToM I(t=%d): this-question targeted=%s",
            turn_number, active_items if active_items else ["(none)"],
        )

    # ── Wasserstein computation ───────────────────────────────────────────────

    def compute_wasserstein_metrics(self, turn_number: int) -> bool:
        """Compute W_self, W_align, W_accuracy for turn_number.

        Returns True if at least one metric was computed, False otherwise
        (POT unavailable or profiles missing).
        """
        if not self._pot_available:
            return False
        if turn_number not in self.E_profiles:
            logger.debug(
                "compute_wasserstein_metrics: E(%d) not built yet, skipping",
                turn_number,
            )
            return False

        cost_mat = get_cost_matrix()
        E_t = self.E_profiles[turn_number]
        computed_any = False

        # W_self: profile shift over last k turns
        self.W_self[turn_number] = {}
        for k in (1, 2, 5):
            t_prev = turn_number - k
            if t_prev in self.E_profiles:
                d = wasserstein_balanced(self.E_profiles[t_prev], E_t, cost_mat)
                if d is not None:
                    self.W_self[turn_number][k] = d
                    computed_any = True

        # W_align: interviewer vs. persona
        if turn_number in self.I_profiles:
            d = wasserstein_balanced(self.I_profiles[turn_number], E_t, cost_mat)
            if d is not None:
                self.W_align[turn_number] = d
                computed_any = True

        # W_accuracy: current assessment vs. ground truth
        if self.ground_truth is not None:
            d = wasserstein_balanced(E_t, self.ground_truth, cost_mat)
            if d is not None:
                self.W_accuracy[turn_number] = d
                computed_any = True

        if computed_any:
            logger.debug(
                "ToM Wasserstein t=%d: W_self=%s, W_align=%s, W_accuracy=%s",
                turn_number,
                {k: round(v, 3) for k, v in self.W_self.get(turn_number, {}).items()},
                round(self.W_align[turn_number], 3) if turn_number in self.W_align else None,
                round(self.W_accuracy[turn_number], 3) if turn_number in self.W_accuracy else None,
            )
        return computed_any

    # ── Coverage gap analysis ─────────────────────────────────────────────────

    def tom_coverage_gaps(self) -> dict:
        """Compute ToM-category mass distribution and identify coverage gaps.

        Uses the latest available E(t). Returns a structured dict with:
        - category_mass: per-category absolute and relative mass
        - category_trajectory: rising / stable / falling
        - gaps: list of underrepresented or uncovered categories

        Only flags gaps when total_mass >= 3.0 (meaningful evidence present).
        """
        if not self.E_profiles:
            return {"available": False, "gaps": [], "category_mass": {}}

        latest_turn = max(self.E_profiles.keys())
        E = self.E_profiles[latest_turn]
        total_mass = float(E.sum())

        category_mass: dict[str, dict] = {}
        for cat_name, item_ids in TOM_CATEGORIES.items():
            indices = [iid - 1 for iid in item_ids]
            cat_sum = float(E[indices].sum())
            n_with_evidence = int((E[indices] > 0).sum())
            category_mass[cat_name] = {
                "absolute": round(cat_sum, 2),
                "pct_of_total": round(100.0 * cat_sum / total_mass, 1)
                if total_mass > 0 else 0.0,
                "n_items_with_evidence": n_with_evidence,
                "n_items": len(item_ids),
                "label": TOM_CATEGORY_LABELS[cat_name],
            }

        # Trajectory: compare latest vs. previous E(t) per category
        sorted_turns = sorted(self.E_profiles.keys())
        category_trajectory: dict[str, str] = {}
        for cat_name, item_ids in TOM_CATEGORIES.items():
            indices = [iid - 1 for iid in item_ids]
            masses = [float(self.E_profiles[t][indices].sum()) for t in sorted_turns]
            if len(masses) >= 2:
                delta = masses[-1] - masses[-2]
                if delta > 0.5:
                    traj = "rising"
                elif delta < -0.5:
                    traj = "falling"
                else:
                    traj = "stable"
            else:
                traj = "insufficient_data"
            category_trajectory[cat_name] = traj

        # Gap identification (only meaningful if total evidence >= 3 points)
        gaps: list[dict] = []
        if total_mass >= 3.0:
            for cat_name, stats in category_mass.items():
                if stats["n_items_with_evidence"] == 0:
                    gaps.append({
                        "category": cat_name,
                        "label": TOM_CATEGORY_LABELS[cat_name],
                        "severity": "uncovered",
                        "trajectory": category_trajectory.get(cat_name, "unknown"),
                    })
                elif stats["pct_of_total"] < 10.0:
                    gaps.append({
                        "category": cat_name,
                        "label": TOM_CATEGORY_LABELS[cat_name],
                        "severity": "underrepresented",
                        "trajectory": category_trajectory.get(cat_name, "unknown"),
                    })

        return {
            "available": True,
            "turn": latest_turn,
            "total_mass": round(total_mass, 2),
            "category_mass": category_mass,
            "category_trajectory": category_trajectory,
            "gaps": gaps,
        }

    # ── Orchestrator context ──────────────────────────────────────────────────

    def get_orchestrator_context(self) -> dict:
        """Return a compact dict for injection into the orchestrator reasoning prompt.

        Provides:
        - Textual gap descriptions (which ToM categories are underexplored)
        - Per-category coverage summary
        - Most recent W_align value (if available)
        """
        if not self.guide_interviewer:
            return {"tom_available": False}

        gaps_data = self.tom_coverage_gaps()
        if not gaps_data["available"]:
            return {"tom_available": False}

        gap_descriptions = [
            f"{g['label']} is {g['severity']} "
            f"(symptom mass trajectory: {g['trajectory']})"
            for g in gaps_data["gaps"]
        ]

        w_align_info: Optional[dict] = None
        if self.W_align:
            latest_t = max(self.W_align.keys())
            w_val = self.W_align[latest_t]
            if w_val > 1.0:
                interp = "high misalignment: interviewer is probing different domains than expressed"
            elif w_val > 0.5:
                interp = "moderate misalignment: some domains being missed"
            else:
                interp = "good alignment: interviewer probing matches what persona has expressed"
            w_align_info = {
                "turn": latest_t,
                "distance": round(w_val, 3),
                "interpretation": interp,
            }

        return {
            "tom_available": True,
            "tom_coverage_gaps": gap_descriptions,
            "tom_interviewer_alignment": w_align_info,
            "tom_category_summary": {
                cat: (
                    f"{data['n_items_with_evidence']}/{data['n_items']} items, "
                    f"{data['pct_of_total']}% of total symptom mass"
                )
                for cat, data in gaps_data["category_mass"].items()
            },
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_summary_dict(self) -> dict:
        """Return a JSON-serialisable summary of all tracked metrics."""
        gaps_data = self.tom_coverage_gaps()

        return {
            "pot_available": self._pot_available,
            "turns_tracked": sorted(self.E_profiles.keys()),
            "E_profiles": {
                str(t): v.tolist() for t, v in self.E_profiles.items()
            },
            "I_profiles": {
                str(t): v.tolist() for t, v in self.I_profiles.items()
            },
            "W_self": {
                str(t): {str(k): round(v, 4) for k, v in kv.items()}
                for t, kv in self.W_self.items()
            },
            "W_align": {
                str(t): round(v, 4) for t, v in self.W_align.items()
            },
            "W_accuracy": {
                str(t): round(v, 4) for t, v in self.W_accuracy.items()
            },
            "coverage_gaps": gaps_data,
            "ground_truth_provided": self.ground_truth is not None,
        }
