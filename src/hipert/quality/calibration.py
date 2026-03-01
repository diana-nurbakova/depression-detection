"""Bias correction via isotonic regression.

LLMs systematically overestimate mental health relevance.
Target distribution (from eRisk BDI-II patterns):
    Score 0: ~50-60%
    Score 1: ~20-25%
    Score 2: ~10-15%
    Score 3: ~5-10%
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

# Target score distribution from spec Section 6.1
TARGET_DISTRIBUTION = {
    0: 0.55,  # ~50-60%
    1: 0.225, # ~20-25%
    2: 0.125, # ~10-15%
    3: 0.075, # ~5-10%
}


def compute_score_distribution(
    labels: list[int],
) -> dict[int, float]:
    """Compute the empirical distribution of scores."""
    counter = Counter(labels)
    total = sum(counter.values())
    return {
        score: counter.get(score, 0) / total if total > 0 else 0
        for score in range(4)
    }


def fit_isotonic_calibration(
    raw_scores: list[float],
    raw_labels: list[int],
) -> IsotonicRegression:
    """Fit isotonic regression for bias correction.

    Maps continuous raw scores to calibrated scores that better
    match the target distribution.

    Args:
        raw_scores: Continuous confidence-weighted scores.
        raw_labels: Discrete labels (0-3).

    Returns:
        Fitted IsotonicRegression model.
    """
    X = np.array(raw_scores)
    y = np.array(raw_labels)

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)

    logger.info(
        "Isotonic calibration fitted on %d samples", len(raw_scores),
    )
    return ir


def apply_calibration(
    scores: list[float],
    model: IsotonicRegression,
) -> list[float]:
    """Apply isotonic calibration to raw scores."""
    return model.predict(np.array(scores)).tolist()
