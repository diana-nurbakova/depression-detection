"""Wasserstein-anchored temporal aggregation for MentalRiskES Task 1.

Stores per-round, per-item predictions and aggregates across rounds to
combat recency bias (within-session therapeutic improvement distorting
past-two-weeks assessments).

Methods:
  T0: Last-round only (baseline)
  T1: Uniform median
  T2: Early-weighted median (step/inverse/linear decay)
  T3: Stability-adaptive (per-item: stable→latest, unstable→early-weighted)

Also provides:
  - Wasserstein (W1) anomaly detection between rounds
  - Per-item confidence estimation from prediction stability
  - Ground metric matrices for PHQ-9, GAD-7, CompACT-10

Spec: mentalriskes2026_wasserstein_temporal_spec.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground metrics (inter-item clinical distance matrices)
# ---------------------------------------------------------------------------

def build_phq9_metric() -> np.ndarray:
    """PHQ-9 9x9 ground metric based on bifactor structure + ToM tiers."""
    categories = {
        "affective": [0, 1, 8],     # anhedonia, mood, suicidality
        "cognitive": [5, 6],         # self-worth, concentration
        "somatic":   [2, 3, 4, 7],   # sleep, fatigue, appetite, psychomotor
    }
    distances = {
        "within": 0.2,
        "adjacent": 0.5,   # cognitive-affective, cognitive-somatic
        "far": 0.8,        # affective-somatic
    }
    M = np.zeros((9, 9))
    item_cat = {}
    for cat, items in categories.items():
        for i in items:
            item_cat[i] = cat
    for i in range(9):
        for j in range(9):
            if i == j:
                continue
            ci, cj = item_cat[i], item_cat[j]
            if ci == cj:
                M[i, j] = distances["within"]
            elif {ci, cj} in [{"cognitive", "affective"}, {"cognitive", "somatic"}]:
                M[i, j] = distances["adjacent"]
            else:
                M[i, j] = distances["far"]
    return M


def build_gad7_metric() -> np.ndarray:
    """GAD-7 7x7 ground metric based on anxiety factor structure."""
    categories = {
        "somatic_anxiety":      [0, 3, 4],  # nervousness, relaxation, restlessness
        "cognitive_anxiety":    [1, 2],      # worry control, excessive worry
        "emotional_reactivity": [5, 6],      # irritability, fear/dread
    }
    M = np.zeros((7, 7))
    item_cat = {}
    for cat, items in categories.items():
        for i in items:
            item_cat[i] = cat
    for i in range(7):
        for j in range(7):
            if i == j:
                continue
            if item_cat[i] == item_cat[j]:
                M[i, j] = 0.2
            else:
                M[i, j] = 0.6
    return M


def build_compact10_metric() -> np.ndarray:
    """CompACT-10 10x10 ground metric based on hexaflex structure."""
    categories = {
        "openness":      [2, 4, 7],       # items 3,5,8
        "awareness":     [0, 5, 8],       # items 1,6,9
        "valued_action": [1, 3, 6, 9],    # items 2,4,7,10
    }
    pair_distances = {
        ("openness", "awareness"): 0.4,
        ("openness", "valued_action"): 0.8,
        ("awareness", "valued_action"): 0.6,
    }
    M = np.zeros((10, 10))
    item_cat = {}
    for cat, items in categories.items():
        for i in items:
            item_cat[i] = cat
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            ci, cj = item_cat[i], item_cat[j]
            if ci == cj:
                M[i, j] = 0.2
            else:
                key = tuple(sorted([ci, cj]))
                M[i, j] = pair_distances.get(key, 0.6)
    return M


# Pre-built metrics (module-level singletons)
_PHQ9_METRIC: np.ndarray | None = None
_GAD7_METRIC: np.ndarray | None = None
_COMPACT10_METRIC: np.ndarray | None = None


def get_ground_metric(instrument: str) -> np.ndarray:
    """Get or build the ground metric for an instrument."""
    global _PHQ9_METRIC, _GAD7_METRIC, _COMPACT10_METRIC
    if instrument == "PHQ-9":
        if _PHQ9_METRIC is None:
            _PHQ9_METRIC = build_phq9_metric()
        return _PHQ9_METRIC
    elif instrument == "GAD-7":
        if _GAD7_METRIC is None:
            _GAD7_METRIC = build_gad7_metric()
        return _GAD7_METRIC
    elif instrument == "CompACT-10":
        if _COMPACT10_METRIC is None:
            _COMPACT10_METRIC = build_compact10_metric()
        return _COMPACT10_METRIC
    raise ValueError(f"Unknown instrument: {instrument}")


# ---------------------------------------------------------------------------
# Instrument specs (duplicated from assessors for independence)
# ---------------------------------------------------------------------------

INSTRUMENT_SPECS = {
    "PHQ-9":      {"n_items": 9,  "scale_max": 3},
    "GAD-7":      {"n_items": 7,  "scale_max": 3},
    "CompACT-10": {"n_items": 10, "scale_max": 6},
}


# ---------------------------------------------------------------------------
# PredictionMatrix
# ---------------------------------------------------------------------------

@dataclass
class PredictionMatrix:
    """Stores per-round, per-item predictions for a single instrument/session.

    Shape: rounds × items.
    """

    instrument: str
    n_items: int
    scale_max: int
    matrix: list[list[int]] = field(default_factory=list)
    round_numbers: list[int] = field(default_factory=list)

    def add_round(self, round_number: int, scores: list[int]) -> None:
        """Add a new round's predictions."""
        assert len(scores) == self.n_items, (
            f"Expected {self.n_items} scores, got {len(scores)}"
        )
        self.matrix.append(list(scores))
        self.round_numbers.append(round_number)

    def get_item_history(self, item_idx: int) -> list[int]:
        """Get all predictions for a single item across rounds."""
        return [row[item_idx] for row in self.matrix]

    def n_rounds(self) -> int:
        return len(self.matrix)

    def latest(self) -> list[int]:
        """Return the most recent round's scores."""
        return list(self.matrix[-1]) if self.matrix else [0] * self.n_items


# ---------------------------------------------------------------------------
# SessionPredictionHistory — manages matrices for all 3 instruments
# ---------------------------------------------------------------------------

@dataclass
class SessionPredictionHistory:
    """Prediction history for a single session across all instruments."""

    session_id: str
    matrices: dict[str, PredictionMatrix] = field(default_factory=dict)
    # W1 anomaly trajectories per instrument
    w1_trajectories: dict[str, list[float]] = field(default_factory=dict)
    # Rounds flagged as anomalous per instrument
    anomalous_rounds: dict[str, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for inst, spec in INSTRUMENT_SPECS.items():
            if inst not in self.matrices:
                self.matrices[inst] = PredictionMatrix(
                    instrument=inst,
                    n_items=spec["n_items"],
                    scale_max=spec["scale_max"],
                )
            if inst not in self.w1_trajectories:
                self.w1_trajectories[inst] = []
            if inst not in self.anomalous_rounds:
                self.anomalous_rounds[inst] = set()

    def add_round(
        self,
        round_number: int,
        phq9: list[int],
        gad7: list[int],
        compact10: list[int],
    ) -> None:
        """Store one round's calibrated predictions for all instruments."""
        self.matrices["PHQ-9"].add_round(round_number, phq9)
        self.matrices["GAD-7"].add_round(round_number, gad7)
        self.matrices["CompACT-10"].add_round(round_number, compact10)

    def n_rounds(self) -> int:
        return self.matrices["PHQ-9"].n_rounds()


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

def aggregate_last_round(matrix: PredictionMatrix) -> list[int]:
    """T0: Return the latest round's raw prediction."""
    return matrix.latest()


def aggregate_uniform_median(matrix: PredictionMatrix) -> list[int]:
    """T1: Per-item median across all rounds."""
    arr = np.array(matrix.matrix)
    return np.median(arr, axis=0).round().astype(int).tolist()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of values with given weights."""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumweight = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumweight, 0.5)
    return float(sorted_values[min(median_idx, len(sorted_values) - 1)])


def aggregate_early_weighted(
    matrix: PredictionMatrix,
    decay: str = "step",
) -> list[int]:
    """T2: Weighted aggregation favouring early rounds.

    decay options:
      'inverse': weight = 1/round_number (strong early preference)
      'linear':  weight = (K - k + 1) / K (mild early preference)
      'step':    weight = 2.0 if k <= 5 else 1.0 (first 5 rounds doubled)
    """
    K = matrix.n_rounds()
    if K == 0:
        return [0] * matrix.n_items
    if K == 1:
        return matrix.latest()

    arr = np.array(matrix.matrix, dtype=float)

    if decay == "inverse":
        weights = np.array([1.0 / k for k in matrix.round_numbers])
    elif decay == "linear":
        weights = np.array([(K - i) / K for i in range(K)])
    elif decay == "step":
        weights = np.array([2.0 if k <= 5 else 1.0 for k in matrix.round_numbers])
    else:
        raise ValueError(f"Unknown decay: {decay}")

    weights = weights / weights.sum()

    result = []
    for item_idx in range(matrix.n_items):
        values = arr[:, item_idx]
        wm = _weighted_median(values, weights)
        result.append(int(round(wm)))
    return result


def aggregate_stability_adaptive(
    matrix: PredictionMatrix,
    stability_threshold: float = 0.5,
) -> tuple[list[int], list[str]]:
    """T3: Per-item adaptive aggregation based on prediction stability.

    Stable items (std < threshold): use latest round (model is confident).
    Unstable items (std >= threshold): use early-weighted median.

    Returns (scores, confidence_labels).
    """
    K = matrix.n_rounds()
    if K == 0:
        return [0] * matrix.n_items, ["insufficient_data"] * matrix.n_items
    if K == 1:
        return matrix.latest(), ["insufficient_data"] * matrix.n_items

    arr = np.array(matrix.matrix, dtype=float)
    latest = arr[-1]

    # Build inverse-decay weights for the early-weighted fallback
    weights = np.array([1.0 / (i + 1) for i in range(K)])
    weights /= weights.sum()

    result = []
    confidences = []

    for item_idx in range(matrix.n_items):
        values = arr[:, item_idx]
        item_std = float(np.std(values))

        if item_std < stability_threshold:
            result.append(int(latest[item_idx]))
            confidences.append("high")
        else:
            wm = _weighted_median(values, weights)
            result.append(int(round(wm)))
            confidences.append("low")

    return result, confidences


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def aggregate(
    matrix: PredictionMatrix,
    method: str = "T2",
    decay: str = "step",
    stability_threshold: float = 0.5,
) -> tuple[list[int], list[str] | None]:
    """Dispatch to the appropriate aggregation method.

    Returns (aggregated_scores, confidence_labels_or_None).
    """
    if method == "T0":
        return aggregate_last_round(matrix), None
    elif method == "T1":
        return aggregate_uniform_median(matrix), None
    elif method == "T2":
        return aggregate_early_weighted(matrix, decay=decay), None
    elif method == "T3":
        return aggregate_stability_adaptive(matrix, stability_threshold=stability_threshold)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Wasserstein anomaly detection
# ---------------------------------------------------------------------------

def compute_w1_trajectory(matrix: PredictionMatrix) -> list[float]:
    """Compute W1 between each round's prediction and the running mean.

    Uses the instrument's ground metric if POT is available,
    falls back to L1 distance otherwise.
    """
    if matrix.n_rounds() < 2:
        return [0.0] * matrix.n_rounds()

    try:
        import ot
        M = get_ground_metric(matrix.instrument)
        use_ot = True
    except ImportError:
        use_ot = False
        M = None

    trajectory: list[float] = []

    for k in range(1, matrix.n_rounds() + 1):
        sub = np.array(matrix.matrix[:k], dtype=float)
        barycenter = np.mean(sub, axis=0)
        current = np.array(matrix.matrix[k - 1], dtype=float)

        b_sum = barycenter.sum()
        c_sum = current.sum()

        if use_ot and b_sum > 0 and c_sum > 0:
            b_dist = barycenter / b_sum
            c_dist = current / c_sum
            w1 = float(ot.emd2(b_dist, c_dist, M))
        elif b_sum > 0 and c_sum > 0:
            # L1 fallback on normalised vectors
            w1 = float(np.sum(np.abs(barycenter / b_sum - current / c_sum)))
        else:
            w1 = 0.0

        trajectory.append(w1)

    return trajectory


def detect_anomalous_rounds(
    w1_trajectory: list[float],
    threshold_factor: float = 2.0,
) -> list[tuple[int, float, bool]]:
    """Flag rounds where W1 exceeds threshold_factor × running_std.

    Returns list of (round_idx, w1_value, is_anomalous).
    """
    results: list[tuple[int, float, bool]] = []
    for k in range(len(w1_trajectory)):
        if k < 3:
            results.append((k, w1_trajectory[k], False))
            continue
        past = w1_trajectory[:k]
        mean_w1 = float(np.mean(past))
        std_w1 = float(np.std(past)) + 1e-6
        is_anomalous = w1_trajectory[k] > mean_w1 + threshold_factor * std_w1
        results.append((k, w1_trajectory[k], is_anomalous))
    return results


def update_anomaly_tracking(
    history: SessionPredictionHistory,
    threshold_factor: float = 2.0,
) -> dict[str, list[tuple[int, float, bool]]]:
    """Recompute W1 trajectories and anomalous round flags for all instruments.

    Returns per-instrument anomaly results.
    """
    all_results: dict[str, list[tuple[int, float, bool]]] = {}

    for inst, matrix in history.matrices.items():
        if matrix.n_rounds() < 2:
            history.w1_trajectories[inst] = [0.0] * matrix.n_rounds()
            all_results[inst] = []
            continue

        traj = compute_w1_trajectory(matrix)
        history.w1_trajectories[inst] = traj

        anomalies = detect_anomalous_rounds(traj, threshold_factor)
        history.anomalous_rounds[inst] = {
            idx for idx, _, is_anom in anomalies if is_anom
        }
        all_results[inst] = anomalies

        n_anom = len(history.anomalous_rounds[inst])
        if n_anom > 0:
            logger.info(
                "Session %s %s: %d anomalous round(s) detected: %s",
                history.session_id, inst, n_anom,
                sorted(history.anomalous_rounds[inst]),
            )

    return all_results


# ---------------------------------------------------------------------------
# Anomaly-aware aggregation
# ---------------------------------------------------------------------------

def aggregate_with_anomaly_handling(
    history: SessionPredictionHistory,
    instrument: str,
    method: str = "T2",
    decay: str = "step",
    stability_threshold: float = 0.5,
    discard_anomalous: bool = True,
) -> tuple[list[int], list[str] | None]:
    """Aggregate with optional discarding of anomalous rounds.

    For PHQ-9/GAD-7: discard anomalous rounds (symptoms are constant).
    For CompACT-10: flag only, do NOT discard (may genuinely evolve).
    """
    matrix = history.matrices[instrument]

    # CompACT-10 never discards anomalous rounds
    if instrument == "CompACT-10":
        discard_anomalous = False

    anomalous = history.anomalous_rounds.get(instrument, set())

    if discard_anomalous and anomalous and matrix.n_rounds() > 1:
        # Build a filtered matrix excluding anomalous rounds
        filtered = PredictionMatrix(
            instrument=matrix.instrument,
            n_items=matrix.n_items,
            scale_max=matrix.scale_max,
        )
        for idx, (rnd, scores) in enumerate(
            zip(matrix.round_numbers, matrix.matrix)
        ):
            if idx not in anomalous:
                filtered.add_round(rnd, scores)

        # If all rounds are anomalous, fall back to full matrix
        if filtered.n_rounds() == 0:
            logger.warning(
                "Session %s %s: all rounds anomalous, using full matrix",
                history.session_id, instrument,
            )
            filtered = matrix

        return aggregate(filtered, method=method, decay=decay,
                         stability_threshold=stability_threshold)

    return aggregate(matrix, method=method, decay=decay,
                     stability_threshold=stability_threshold)


# ---------------------------------------------------------------------------
# Per-item confidence estimation
# ---------------------------------------------------------------------------

def compute_item_confidence(
    matrix: PredictionMatrix,
) -> list[tuple[float, str]]:
    """Compute per-item confidence based on prediction stability across rounds.

    Returns list of (confidence_score, confidence_label) per item.
    confidence_score: 0.0 (no confidence) to 1.0 (perfect stability).
    """
    if matrix.n_rounds() < 2:
        return [(0.5, "insufficient_data")] * matrix.n_items

    arr = np.array(matrix.matrix, dtype=float)
    max_possible_std = matrix.scale_max / 2

    results: list[tuple[float, str]] = []
    for item_idx in range(matrix.n_items):
        values = arr[:, item_idx]
        item_std = float(np.std(values))
        confidence = 1.0 - min(item_std / max_possible_std, 1.0)

        if confidence >= 0.8:
            label = "high"
        elif confidence >= 0.5:
            label = "moderate"
        else:
            label = "low"

        results.append((round(confidence, 3), label))

    return results


# ---------------------------------------------------------------------------
# Full temporal aggregation entry point
# ---------------------------------------------------------------------------

def apply_temporal_aggregation(
    history: SessionPredictionHistory,
    phq9_method: str = "T2",
    gad7_method: str = "T2",
    compact10_method: str = "T3",
    decay: str = "step",
    stability_threshold: float = 0.5,
    w1_threshold_factor: float = 2.0,
    discard_anomalous: bool = True,
) -> dict:
    """Apply the full temporal aggregation pipeline for a session.

    Steps:
      1. Update W1 anomaly tracking
      2. Per-instrument aggregation with anomaly handling
      3. Compute per-item confidence

    Returns dict with aggregated scores, confidence, and anomaly info.
    """
    methods = {
        "PHQ-9": phq9_method,
        "GAD-7": gad7_method,
        "CompACT-10": compact10_method,
    }

    # Step 1: anomaly detection
    anomaly_results = update_anomaly_tracking(history, w1_threshold_factor)

    # Step 2: aggregate per instrument
    aggregated: dict[str, list[int]] = {}
    confidence_labels: dict[str, list[str] | None] = {}

    for inst, method in methods.items():
        scores, conf = aggregate_with_anomaly_handling(
            history, inst,
            method=method,
            decay=decay,
            stability_threshold=stability_threshold,
            discard_anomalous=discard_anomalous,
        )
        # Clip to valid range
        spec = INSTRUMENT_SPECS[inst]
        scores = [max(0, min(spec["scale_max"], s)) for s in scores]
        aggregated[inst] = scores
        confidence_labels[inst] = conf

    # Step 3: per-item confidence
    item_confidence = {
        inst: compute_item_confidence(history.matrices[inst])
        for inst in methods
    }

    return {
        "phq9": aggregated["PHQ-9"],
        "gad7": aggregated["GAD-7"],
        "compact10": aggregated["CompACT-10"],
        "confidence_labels": confidence_labels,
        "item_confidence": item_confidence,
        "anomaly_results": anomaly_results,
        "w1_trajectories": dict(history.w1_trajectories),
        "anomalous_rounds": {
            k: sorted(v) for k, v in history.anomalous_rounds.items()
        },
    }
