"""Wasserstein Distance for within-user temporal shift detection (Spec Section 5.1).

3 window scales × (21 symptom + emotion + embedding + topic) = 72 features.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein_1d(early: np.ndarray, recent: np.ndarray) -> float:
    """1-Wasserstein (EMD) between two 1D distributions."""
    if len(early) == 0 or len(recent) == 0:
        return 0.0
    return float(wasserstein_distance(early, recent))


def compute_sliced_wasserstein(
    early: np.ndarray,
    recent: np.ndarray,
    n_projections: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    """Sliced Wasserstein Distance for high-dimensional distributions.

    Projects onto random 1D directions and averages 1D Wasserstein distances.

    Args:
        early: (n_early, dim) array
        recent: (n_recent, dim) array
        n_projections: number of random projections
    """
    if len(early) == 0 or len(recent) == 0:
        return 0.0

    if rng is None:
        rng = np.random.default_rng(42)

    dim = early.shape[1]
    distances = []

    for _ in range(n_projections):
        # Random unit direction
        direction = rng.standard_normal(dim)
        direction /= np.linalg.norm(direction) + 1e-10

        # Project
        proj_early = early @ direction
        proj_recent = recent @ direction
        distances.append(wasserstein_distance(proj_early, proj_recent))

    return float(np.mean(distances))


def compute_symptom_wasserstein(
    activations: list[np.ndarray],
    round_k: int,
    short_window: int = 5,
    medium_window: int = 25,
) -> np.ndarray:
    """Compute Wasserstein distances for symptom activations at 3 time scales.

    Args:
        activations: list of (21,) arrays, one per round
        round_k: current round number
        short_window: half-window for short-range
        medium_window: half-window for medium-range

    Returns:
        (63,) array: 3 scales × 21 symptoms
    """
    n = len(activations)
    result = np.zeros(63)

    if n < 10:
        return result

    arr = np.stack(activations)  # (n, 21)

    # Short-range: [k-10, k-5] vs [k-5, k]
    if n >= 10:
        half = short_window
        early = arr[max(0, n - 2 * half):n - half]
        recent = arr[n - half:]
        for s in range(21):
            result[s] = compute_wasserstein_1d(early[:, s], recent[:, s])

    # Medium-range: [k-50, k-25] vs [k-25, k]
    if n >= 50:
        half = medium_window
        early = arr[max(0, n - 2 * half):n - half]
        recent = arr[n - half:]
        for s in range(21):
            result[21 + s] = compute_wasserstein_1d(early[:, s], recent[:, s])

    # Long-range: [0, n/2] vs [n/2, n]
    if n >= 20:
        mid = n // 2
        early = arr[:mid]
        recent = arr[mid:]
        for s in range(21):
            result[42 + s] = compute_wasserstein_1d(early[:, s], recent[:, s])

    return result


def compute_scalar_wasserstein(
    values: list[float],
    short_window: int = 5,
    medium_window: int = 25,
) -> np.ndarray:
    """Compute Wasserstein for a scalar time series at 3 scales.

    Returns (3,) array: [short, medium, long]
    """
    n = len(values)
    result = np.zeros(3)
    arr = np.array(values)

    if n >= 10:
        half = short_window
        result[0] = compute_wasserstein_1d(
            arr[max(0, n - 2 * half):n - half],
            arr[n - half:],
        )

    if n >= 50:
        half = medium_window
        result[1] = compute_wasserstein_1d(
            arr[max(0, n - 2 * half):n - half],
            arr[n - half:],
        )

    if n >= 20:
        mid = n // 2
        result[2] = compute_wasserstein_1d(arr[:mid], arr[mid:])

    return result


def compute_all_wasserstein(
    symptom_activations: list[np.ndarray],
    emotion_entropies: list[float],
    embedding_history: list[np.ndarray],
    topic_entropies: list[float],
    n_projections: int = 50,
) -> np.ndarray:
    """Compute full 72d Wasserstein feature vector.

    Components:
        - 63d: symptom Wasserstein (3 scales × 21)
        - 3d: emotion Wasserstein (3 scales)
        - 3d: embedding sliced Wasserstein (3 scales)
        - 3d: topic Wasserstein (3 scales)
    """
    # Symptoms: 63d
    symptom_w = compute_symptom_wasserstein(
        symptom_activations, len(symptom_activations)
    )

    # Emotion: 3d
    emotion_w = compute_scalar_wasserstein(emotion_entropies)

    # Embedding: 3d (sliced Wasserstein at 3 scales)
    embedding_w = np.zeros(3)
    if len(embedding_history) >= 10:
        n = len(embedding_history)
        embs = np.stack(embedding_history)
        half = 5
        embedding_w[0] = compute_sliced_wasserstein(
            embs[max(0, n - 2 * half):n - half],
            embs[n - half:],
            n_projections,
        )
    if len(embedding_history) >= 50:
        n = len(embedding_history)
        embs = np.stack(embedding_history)
        half = 25
        embedding_w[1] = compute_sliced_wasserstein(
            embs[max(0, n - 2 * half):n - half],
            embs[n - half:],
            n_projections,
        )
    if len(embedding_history) >= 20:
        n = len(embedding_history)
        embs = np.stack(embedding_history)
        mid = n // 2
        embedding_w[2] = compute_sliced_wasserstein(
            embs[:mid], embs[mid:], n_projections,
        )

    # Topic: 3d
    topic_w = compute_scalar_wasserstein(topic_entropies)

    return np.concatenate([symptom_w, emotion_w, embedding_w, topic_w])
