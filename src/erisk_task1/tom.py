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


def wasserstein_transport_plan(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> Optional[tuple[float, np.ndarray]]:
    """Balanced W₁ with full transport plan between two BDI-II profiles.

    Returns (distance, gamma) where gamma is the 21×21 transport plan matrix.
    gamma[i][j] = mass moved from predicted item i to ground-truth item j.
    Returns None if POT is unavailable, either profile is all-zero, or fails.
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
        M = cost_matrix.astype(np.float64)
        gamma = ot.emd(a, b, M)
        dist = float(np.sum(gamma * M))
        return dist, gamma
    except Exception as e:
        logger.debug("wasserstein_transport_plan failed: %s", e)
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
# 5. Three-layer transport plan analysis
# ─────────────────────────────────────────────────────────────────────────────


def _listed_item_indices(gt: np.ndarray) -> list[int]:
    """Return 0-based indices of items with non-zero ground-truth scores."""
    return [i for i in range(len(gt)) if gt[i] > 1e-10]


def transport_analysis_layer1(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    cost_matrix: np.ndarray,
    reg: float = 0.1,
    reg_m: float = 0.4,
) -> Optional[dict]:
    """Layer 1: Unbalanced OT on full raw (unnormalized) profiles.

    Uses unbalanced Sinkhorn so excess predicted mass is absorbed rather than
    forced into transport.  The plan reveals: correct matches, misattributions,
    and hallucinated mass that has no GT counterpart.

    Returns dict with distance, plan (21×21), and per-item breakdown, or None.
    """
    try:
        import ot
    except ImportError:
        return None

    pred = np.maximum(predicted, 0.0).astype(np.float64) + 1e-10
    gt = np.maximum(ground_truth, 0.0).astype(np.float64) + 1e-10
    M = cost_matrix.astype(np.float64)

    try:
        gamma = ot.unbalanced.sinkhorn_unbalanced(
            pred, gt, M, reg=reg, reg_m=reg_m,
        )
        dist = float(np.sum(gamma * M))
    except Exception as e:
        logger.debug("transport_analysis_layer1 failed: %s", e)
        return None

    # Per-item breakdown
    listed_idx = _listed_item_indices(ground_truth)
    items = []
    for i in range(21):
        row_mass = float(gamma[i].sum())
        diag_mass = float(gamma[i, i])
        # Where does item i's mass go?
        top_targets = sorted(
            [(j, float(gamma[i, j])) for j in range(21) if gamma[i, j] > 1e-8],
            key=lambda x: -x[1],
        )[:5]
        items.append({
            "item_id": i + 1,
            "item_name": BDI_SHORT[i + 1],
            "predicted": float(predicted[i]),
            "gt": float(ground_truth[i]),
            "transported_mass": round(row_mass, 6),
            "self_match": round(diag_mass, 6),
            "in_gt": i in listed_idx,
            "top_targets": [
                {"item_id": j + 1, "name": BDI_SHORT[j + 1], "mass": round(m, 6)}
                for j, m in top_targets
            ],
        })

    # Absorbed mass = predicted mass not transported
    total_pred = float(pred.sum())
    total_transported = float(gamma.sum())

    return {
        "distance": round(dist, 6),
        "reg": reg,
        "reg_m": reg_m,
        "total_predicted_mass": round(float(predicted.sum()), 2),
        "total_gt_mass": round(float(ground_truth.sum()), 2),
        "total_transported": round(total_transported, 6),
        "absorbed_mass": round(total_pred - total_transported, 6),
        "plan": np.round(gamma, 6).tolist(),
        "items": items,
    }


def transport_analysis_layer2(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> Optional[dict]:
    """Layer 2: Per-item MAE restricted to listed GT items.

    The cleanest diagnostic — no OT, just direct comparison on items where
    the clinician explicitly specified severity.

    Returns dict with per-item errors and aggregate MAE, or None if GT is empty.
    """
    listed_idx = _listed_item_indices(ground_truth)
    if not listed_idx:
        return None

    items = []
    total_ae = 0.0
    for i in listed_idx:
        p = float(predicted[i])
        g = float(ground_truth[i])
        ae = abs(p - g)
        total_ae += ae
        items.append({
            "item_id": i + 1,
            "item_name": BDI_SHORT[i + 1],
            "predicted": p,
            "gt": g,
            "error": round(p - g, 2),
            "abs_error": round(ae, 2),
        })

    mae = total_ae / len(listed_idx)

    # Direction summary
    over = sum(1 for it in items if it["error"] > 0.5)
    under = sum(1 for it in items if it["error"] < -0.5)
    correct = len(items) - over - under

    return {
        "n_listed_items": len(listed_idx),
        "mae": round(mae, 4),
        "total_absolute_error": round(total_ae, 2),
        "n_overestimated": over,
        "n_underestimated": under,
        "n_correct": correct,
        "items": items,
    }


def transport_analysis_layer3(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    cost_matrix: np.ndarray,
) -> Optional[dict]:
    """Layer 3: Balanced W₁ restricted to listed GT items only.

    Zeros out everything except listed items, normalizes, and computes W₁
    with transport plan.  Produces a small K×K confusion matrix (K = number
    of listed items) showing whether the assessor confused those specific
    symptoms with each other.

    Returns dict with distance, restricted plan, and item labels, or None.
    """
    try:
        import ot
    except ImportError:
        return None

    listed_idx = _listed_item_indices(ground_truth)
    if len(listed_idx) < 2:
        return None  # Need at least 2 items for meaningful transport

    # Restrict to listed items
    pred_restricted = np.zeros_like(predicted, dtype=np.float64)
    gt_restricted = np.zeros_like(ground_truth, dtype=np.float64)
    for i in listed_idx:
        pred_restricted[i] = max(predicted[i], 0.0)
        gt_restricted[i] = max(ground_truth[i], 0.0)

    # Normalize
    a = _l1_normalize(pred_restricted)
    b = _l1_normalize(gt_restricted)
    if a is None or b is None:
        return None

    M = cost_matrix.astype(np.float64)

    try:
        gamma_full = ot.emd(a, b, M)
        dist = float(np.sum(gamma_full * M))
    except Exception as e:
        logger.debug("transport_analysis_layer3 failed: %s", e)
        return None

    # Extract the K×K submatrix for listed items only
    K = len(listed_idx)
    gamma_sub = np.zeros((K, K), dtype=np.float64)
    for ri, i in enumerate(listed_idx):
        for ci, j in enumerate(listed_idx):
            gamma_sub[ri, ci] = gamma_full[i, j]

    # Build readable confusion entries (off-diagonal)
    confusions = []
    for ri, i in enumerate(listed_idx):
        for ci, j in enumerate(listed_idx):
            if ri != ci and gamma_sub[ri, ci] > 1e-6:
                confusions.append({
                    "from_item": i + 1,
                    "from_name": BDI_SHORT[i + 1],
                    "to_item": j + 1,
                    "to_name": BDI_SHORT[j + 1],
                    "mass": round(float(gamma_sub[ri, ci]), 6),
                })
    confusions.sort(key=lambda x: -x["mass"])

    item_labels = [
        {"index": i, "item_id": i + 1, "name": BDI_SHORT[i + 1]}
        for i in listed_idx
    ]

    return {
        "distance": round(dist, 6),
        "n_items": K,
        "item_labels": item_labels,
        "plan_submatrix": np.round(gamma_sub, 6).tolist(),
        "plan_full": np.round(gamma_full, 6).tolist(),
        "confusions": confusions,
        "diagonal_mass": round(float(np.trace(gamma_sub)), 6),
        "off_diagonal_mass": round(float(gamma_sub.sum() - np.trace(gamma_sub)), 6),
    }


def compute_transport_analysis(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
) -> dict:
    """Run all three layers of transport plan analysis.

    Args:
        predicted: 21-dim predicted BDI-II profile (raw scores).
        ground_truth: 21-dim ground-truth BDI-II profile (raw scores).
        cost_matrix: 21×21 clinical factor distance matrix (default: auto).

    Returns:
        Dict with keys layer1, layer2, layer3 (each may be None if computation fails).
    """
    if cost_matrix is None:
        cost_matrix = get_cost_matrix()

    return {
        "layer1_unbalanced_ot": transport_analysis_layer1(
            predicted, ground_truth, cost_matrix,
        ),
        "layer2_listed_item_mae": transport_analysis_layer2(
            predicted, ground_truth,
        ),
        "layer3_restricted_w1": transport_analysis_layer3(
            predicted, ground_truth, cost_matrix,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Interviewer attention heuristic (keyword-based; working-notes version)
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

    transport_plans: dict[int, np.ndarray] = field(default_factory=dict)
    """transport_plans[t] = 21×21 optimal transport plan γ(E(t), G)."""

    transport_analysis: dict[int, dict] = field(default_factory=dict)
    """transport_analysis[t] = 3-layer analysis dict (layer1/layer2/layer3)."""

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

        # W_accuracy: current assessment vs. ground truth (with transport plan)
        if self.ground_truth is not None:
            result = wasserstein_transport_plan(E_t, self.ground_truth, cost_mat)
            if result is not None:
                d, gamma = result
                self.W_accuracy[turn_number] = d
                self.transport_plans[turn_number] = gamma
                computed_any = True

            # 3-layer transport analysis (on raw unnormalized profiles)
            analysis = compute_transport_analysis(E_t, self.ground_truth, cost_mat)
            if any(v is not None for v in analysis.values()):
                self.transport_analysis[turn_number] = analysis

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
            "transport_plans": {
                str(t): np.round(gamma, 6).tolist()
                for t, gamma in self.transport_plans.items()
            },
            "transport_analysis": {
                str(t): analysis
                for t, analysis in self.transport_analysis.items()
            },
            "coverage_gaps": gaps_data,
            "ground_truth_provided": self.ground_truth is not None,
        }
