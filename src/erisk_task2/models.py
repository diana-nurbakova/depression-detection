"""Data models for eRisk 2026 Task 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# BDI-II symptom definitions (21 items)
BDI_SYMPTOMS: dict[int, str] = {
    1: "Sadness",
    2: "Pessimism",
    3: "Past failure",
    4: "Loss of pleasure",
    5: "Guilty feelings",
    6: "Punishment feelings",
    7: "Self-dislike",
    8: "Self-criticalness",
    9: "Suicidal thoughts",
    10: "Crying",
    11: "Agitation",
    12: "Loss of interest",
    13: "Indecisiveness",
    14: "Worthlessness",
    15: "Loss of energy",
    16: "Sleep changes",
    17: "Irritability",
    18: "Appetite changes",
    19: "Concentration difficulty",
    20: "Tiredness/fatigue",
    21: "Loss of interest in sex",
}

NUM_SYMPTOMS = 21

# DSM-5 to BDI-II mapping (from ReDSM5)
DSM5_TO_BDI: dict[str, list[int]] = {
    "depressed_mood": [1, 10, 17],
    "anhedonia": [4, 12, 21],
    "appetite_weight": [18],
    "sleep": [16],
    "psychomotor": [11],
    "fatigue": [15, 20],
    "worthlessness_guilt": [5, 6, 7, 8, 14],
    "cognitive": [13, 19],
    "suicidal": [9],
}


class SeverityBand(str, Enum):
    MINIMAL = "minimal"  # 0-13
    MILD = "mild"  # 14-19
    MODERATE = "moderate"  # 20-28
    SEVERE = "severe"  # 29-63


def score_to_band(total: int) -> SeverityBand:
    if total <= 13:
        return SeverityBand.MINIMAL
    elif total <= 19:
        return SeverityBand.MILD
    elif total <= 28:
        return SeverityBand.MODERATE
    return SeverityBand.SEVERE


# ---------------------------------------------------------------------------
# Thread / Comment models (normalized internal representation)
# ---------------------------------------------------------------------------

@dataclass
class Comment:
    comment_id: str
    author: str
    body: str
    parent_id: str  # submission_id or another comment_id
    created_utc: str  # ISO 8601 normalized
    is_target: bool = False


@dataclass
class Thread:
    submission_id: str
    title: str
    body: str
    author: str
    created_utc: str
    comments: list[Comment] = field(default_factory=list)
    target_subject: str = ""
    round_number: int = 0

    @property
    def target_is_author(self) -> bool:
        return self.author == self.target_subject

    @property
    def target_comments(self) -> list[Comment]:
        return [c for c in self.comments if c.is_target]

    @property
    def other_comments(self) -> list[Comment]:
        return [c for c in self.comments if not c.is_target]

    @property
    def target_texts(self) -> list[str]:
        texts = []
        if self.target_is_author and self.body.strip():
            texts.append(self.body)
        texts.extend(c.body for c in self.target_comments if c.body.strip())
        return texts

    @property
    def direct_replies_to_target(self) -> list[Comment]:
        target_ids = {self.submission_id} if self.target_is_author else set()
        target_ids.update(c.comment_id for c in self.target_comments)
        return [
            c for c in self.comments
            if c.parent_id in target_ids and not c.is_target
        ]

    @property
    def has_target_text(self) -> bool:
        return len(self.target_texts) > 0


# ---------------------------------------------------------------------------
# User profile (accumulated state across rounds)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    subject_id: str
    rounds_seen: int = 0
    last_active_round: int = -1

    # Layer 1: Textual features
    embedding_sum: Optional[np.ndarray] = None  # running weighted sum (1920d)
    embedding_weight: float = 0.0
    symptom_activations: list[np.ndarray] = field(default_factory=list)  # per-round 21d
    all_target_texts: list[str] = field(default_factory=list)
    text_word_counts: list[int] = field(default_factory=list)

    # Layer 2: Context features
    reply_sentiments: list[float] = field(default_factory=list)
    concern_flags: list[bool] = field(default_factory=list)
    is_author_flags: list[bool] = field(default_factory=list)
    reply_depths: list[int] = field(default_factory=list)
    target_silent_rounds: int = 0

    # Layer 3: Emotion + Topic
    emotion_distributions: list[np.ndarray] = field(default_factory=list)  # 8d per round
    topic_distributions: list[np.ndarray] = field(default_factory=list)
    rolling_text_buffer: list[str] = field(default_factory=list)  # last N texts for BERTopic

    # ToM features
    self_view_history: list[dict] = field(default_factory=list)
    observer_view_history: list[dict] = field(default_factory=list)

    # Bandit state (Thompson Sampling)
    bandit_alphas: Optional[np.ndarray] = None  # 21d Beta prior alphas
    bandit_betas: Optional[np.ndarray] = None  # 21d Beta prior betas

    # Thread topic similarities
    thread_topic_sims: list[np.ndarray] = field(default_factory=list)  # 21d per round


@dataclass
class RunUserState:
    alert_emitted: bool = False
    alert_round: Optional[int] = None
    last_score: float = 0.0
    last_probability: float = 0.0
    consecutive_positives: int = 0


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

class ClassifierType(str, Enum):
    XGBOOST = "xgboost"
    NEURAL_NET = "neural_net"
    SVM = "svm"
    ENSEMBLE = "ensemble"


@dataclass
class RunConfig:
    run_number: int
    classifier_type: ClassifierType
    theta_init: float  # starting threshold
    theta_floor: float  # minimum threshold
    erde_o: int  # ERDE parameter (5 or 50)
    t_con: int  # consecutive confirmations required
    feature_mask: Optional[list[str]] = None  # None = all features

    @property
    def name(self) -> str:
        return f"run_{self.run_number}"


# Default 5 official runs from spec Section 14
DEFAULT_RUNS = [
    RunConfig(0, ClassifierType.XGBOOST, 0.85, 0.45, 50, 2),  # FULL_XGBOOST_ERDE50
    RunConfig(1, ClassifierType.XGBOOST, 0.70, 0.35, 5, 1),   # FULL_XGBOOST_ERDE5
    RunConfig(2, ClassifierType.NEURAL_NET, 0.85, 0.45, 50, 2),  # FULL_NN_ERDE50
    RunConfig(3, ClassifierType.ENSEMBLE, 0.80, 0.40, 30, 2),  # FULL_ENSEMBLE_BALANCED
    RunConfig(4, ClassifierType.XGBOOST, 0.85, 0.45, 50, 2,   # NO_TOM_XGBOOST_ERDE50
              feature_mask=["no_tom"]),
]
