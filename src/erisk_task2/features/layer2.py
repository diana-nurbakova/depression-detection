"""Layer 2: Conversational Context Features (Spec Section 4.2).

- Reply sentiment (VADER)
- Community concern detection
- Conversational position features
- Thread topic similarity
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from erisk_task2.models import Thread

logger = logging.getLogger(__name__)

# Concern/support phrases (~40 curated)
CONCERN_PHRASES = [
    "are you okay", "are you alright", "are you ok", "u okay", "u ok",
    "you okay", "you alright", "you ok",
    "please talk to someone", "please reach out", "please get help",
    "seek help", "get help", "talk to someone", "reach out to someone",
    "i'm worried", "i'm concerned", "im worried", "im concerned",
    "are you safe", "are you doing ok", "how are you doing",
    "thinking of you", "praying for you", "sending love", "sending hugs",
    "hang in there", "stay strong", "don't give up", "dont give up",
    "you matter", "you're not alone", "you are not alone",
    "suicide hotline", "crisis line", "crisis text", "988",
    "please be safe", "take care of yourself", "hope you're okay",
    "hope you are okay", "hope you're doing better",
]

# Pre-compile patterns for efficiency
_CONCERN_PATTERNS = [re.compile(re.escape(p), re.IGNORECASE) for p in CONCERN_PHRASES]


def compute_reply_sentiment(thread: Thread) -> tuple[float, list[str]]:
    """Compute VADER sentiment for replies to the target user.

    Falls back to comments in target's conversational branches when
    no direct replies exist.

    Returns (mean_compound, list_of_analyzed_texts).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Try direct replies first
    direct_replies = thread.direct_replies_to_target
    if direct_replies:
        texts = [c.body for c in direct_replies if c.body]
    else:
        # Fallback: comments in branches where target participates
        target_ids = {c.comment_id for c in thread.target_comments}
        if thread.target_is_author:
            target_ids.add(thread.submission_id)

        # Find comments sharing a parent chain with target
        branch_texts = []
        for c in thread.other_comments:
            if c.parent_id in target_ids and c.body:
                branch_texts.append(c.body)
        texts = branch_texts

    if not texts:
        return 0.0, []

    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return float(np.mean(scores)), texts


def detect_concern(thread: Thread) -> tuple[bool, list[str]]:
    """Scan all non-target text for concern/support phrases.

    Returns (concern_detected, matched_phrases).
    """
    matched = []
    for c in thread.other_comments:
        if not c.body:
            continue
        for pattern in _CONCERN_PATTERNS:
            if pattern.search(c.body):
                matched.append(pattern.pattern.replace("\\", ""))
                break  # one match per comment is enough

    # Also check submission body if target is not author
    if not thread.target_is_author and thread.body:
        for pattern in _CONCERN_PATTERNS:
            if pattern.search(thread.body):
                matched.append(pattern.pattern.replace("\\", ""))
                break

    return len(matched) > 0, matched


def compute_conversational_position(
    thread: Thread,
    profile_is_author_count: int,
    profile_total_threads: int,
    profile_silent_count: int,
    profile_word_counts: list[int],
) -> dict[str, float]:
    """Compute conversational position features.

    Returns dict with:
        is_author_ratio, target_silent_ratio, text_volume_mean,
        text_volume_trend, reply_depth_mean
    """
    n = max(profile_total_threads, 1)

    # Reply chain depth for target's comments
    depths = []
    parent_map = {thread.submission_id: 0}
    for c in thread.comments:
        parent_depth = parent_map.get(c.parent_id, 0)
        parent_map[c.comment_id] = parent_depth + 1
        if c.is_target:
            depths.append(parent_depth + 1)

    return {
        "is_author_ratio": profile_is_author_count / n,
        "target_silent_ratio": profile_silent_count / n,
        "text_volume_mean": float(np.mean(profile_word_counts)) if profile_word_counts else 0.0,
        "text_volume_trend": _compute_trend(profile_word_counts) if len(profile_word_counts) > 5 else 0.0,
        "reply_depth_mean": float(np.mean(depths)) if depths else 0.0,
    }


def compute_thread_topic_similarity(
    title_embedding: np.ndarray,
    symptom_references: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between thread title and 21 symptom references.

    Args:
        title_embedding: (dim,) normalized embedding of thread title
        symptom_references: (21, dim) normalized symptom reference embeddings

    Returns:
        (21,) similarity scores
    """
    norm = np.linalg.norm(title_embedding)
    if norm < 1e-8:
        return np.zeros(21)
    normed = title_embedding / norm
    return symptom_references @ normed


def _compute_trend(values: list[int | float]) -> float:
    """Compute linear trend (slope) over a sequence of values."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    # Simple linear regression slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-10:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)
