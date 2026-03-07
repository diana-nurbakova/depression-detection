"""Feature vector assembly from UserProfile (Spec Section 9.1).

Concatenates all feature components into a single vector for classification.
Total approximate dimension: ~2341.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from erisk_task2.features.layer1 import (
    compute_lexical_features,
    compute_symptom_distribution_stats,
    get_running_mean,
)
from erisk_task2.models import NUM_SYMPTOMS, UserProfile


# Feature component dimensions
DIMS = {
    "embedding": 1920,
    "symptom_max": NUM_SYMPTOMS,       # 21
    "symptom_mean": NUM_SYMPTOMS,      # 21
    "symptom_stats": 147,              # 21 × 7
    "lexical": 4,
    "reply_sentiment": 3,              # mean, std, trend
    "concern": 1,
    "conv_position": 3,               # is_author_ratio, silent_ratio, reply_depth
    "thread_topic": NUM_SYMPTOMS,      # 21
    "emotion": 9,                      # 8 + entropy
    "bertopic": 41,                    # ~40 topics + depression ratio
    "wasserstein": 72,                 # 3 scales × 24
    "mahalanobis": 3,
    "combined_dist": 2,
    "tom": 47,
    "bandit": 25,                      # weighted_score + entropy + count + 21 weights + uncertainty
    "meta": 1,                         # rounds_seen
}


def assemble_feature_vector(
    profile: UserProfile,
    wasserstein_features: Optional[np.ndarray] = None,
    mahalanobis_features: Optional[np.ndarray] = None,
    tom_features: Optional[np.ndarray] = None,
    bandit_features: Optional[np.ndarray] = None,
    emotion_features: Optional[np.ndarray] = None,
    topic_features: Optional[np.ndarray] = None,
    feature_mask: Optional[list[str]] = None,
) -> np.ndarray:
    """Assemble full feature vector from user profile and computed features.

    Args:
        profile: UserProfile with accumulated features
        wasserstein_features: (72,) from distances module
        mahalanobis_features: (3,) from distances module
        tom_features: (47,) from ToM module
        bandit_features: (25,) from Thompson Sampling
        emotion_features: (9,) emotion distribution + entropy
        topic_features: (41,) BERTopic distribution + depression ratio
        feature_mask: list of component names to exclude (e.g. ["no_tom"])

    Returns:
        Feature vector of dimension ~2341
    """
    mask = set(feature_mask or [])
    components = []

    # Layer 1: Embeddings (1920d)
    emb = get_running_mean(profile.embedding_sum, profile.embedding_weight)
    components.append(emb)

    # Layer 1: Symptom max-pool (21d) + mean-pool (21d)
    if profile.symptom_activations:
        arr = np.stack(profile.symptom_activations)
        components.append(arr.max(axis=0))
        components.append(arr.mean(axis=0))
    else:
        components.append(np.zeros(NUM_SYMPTOMS))
        components.append(np.zeros(NUM_SYMPTOMS))

    # Layer 1: Symptom distributional stats (147d)
    components.append(compute_symptom_distribution_stats(profile.symptom_activations))

    # Layer 1: Lexical (4d)
    components.append(compute_lexical_features(profile.all_target_texts))

    # Layer 2: Reply sentiment (3d)
    if profile.reply_sentiments:
        sent = np.array(profile.reply_sentiments)
        trend = _trend(profile.reply_sentiments)
        components.append(np.array([sent.mean(), sent.std(), trend]))
    else:
        components.append(np.zeros(3))

    # Layer 2: Concern (1d)
    if profile.concern_flags:
        components.append(np.array([sum(profile.concern_flags) / len(profile.concern_flags)]))
    else:
        components.append(np.zeros(1))

    # Layer 2: Conversational position (3d)
    n = max(profile.rounds_seen, 1)
    components.append(np.array([
        sum(profile.is_author_flags) / n if profile.is_author_flags else 0.0,
        profile.target_silent_rounds / n,
        np.mean(profile.reply_depths) if profile.reply_depths else 0.0,
    ]))

    # Layer 2: Thread topic similarity (21d)
    if profile.thread_topic_sims:
        components.append(np.stack(profile.thread_topic_sims).mean(axis=0))
    else:
        components.append(np.zeros(NUM_SYMPTOMS))

    # Layer 3: Emotion (9d)
    if emotion_features is not None:
        components.append(emotion_features)
    else:
        components.append(np.zeros(9))

    # Layer 3: BERTopic (41d)
    if topic_features is not None:
        components.append(topic_features)
    else:
        components.append(np.zeros(41))

    # Wasserstein distances (72d)
    components.append(wasserstein_features if wasserstein_features is not None else np.zeros(72))

    # Mahalanobis (3d)
    components.append(mahalanobis_features if mahalanobis_features is not None else np.zeros(3))

    # Combined distributional score (2d)
    if mahalanobis_features is not None and wasserstein_features is not None:
        d_m = mahalanobis_features[0]  # D_M_control
        w_mean = wasserstein_features.mean()
        components.append(np.array([d_m, w_mean]))
    else:
        components.append(np.zeros(2))

    # ToM (47d) — can be masked
    if "no_tom" not in mask and tom_features is not None:
        components.append(tom_features)
    else:
        components.append(np.zeros(47))

    # Bandits (25d)
    components.append(bandit_features if bandit_features is not None else np.zeros(25))

    # Meta (1d)
    components.append(np.array([profile.rounds_seen]))

    return np.concatenate(components)


def get_feature_dim(feature_mask: Optional[list[str]] = None) -> int:
    """Get total feature dimension given a feature mask."""
    total = sum(DIMS.values())
    if feature_mask and "no_tom" in feature_mask:
        total -= DIMS["tom"]
        total += DIMS["tom"]  # still 47d of zeros
    return total


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-10:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)
