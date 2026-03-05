"""Data models for the Task 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ItemState(str, Enum):
    SCORED = "SCORED"
    NO_EVIDENCE = "NO_EVIDENCE"
    EVIDENCE_OF_ABSENCE = "EVIDENCE_OF_ABSENCE"


class SeverityBand(str, Enum):
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class PersonaProfile(str, Enum):
    ENGAGED_COPING = "Engaged-coping"
    DISMISSIVE_FLAT = "Dismissive-flat"
    HEDGING_DEFLECTING = "Hedging-deflecting"
    HOPELESS_WITHDRAWN = "Hopeless-withdrawn"
    MIXED = "Mixed"


class TopicArea(str, Enum):
    EMOTIONAL_STATE = "EMOTIONAL_STATE"
    ACTIVITIES_INTERESTS = "ACTIVITIES_INTERESTS"
    DAILY_ROUTINE = "DAILY_ROUTINE"
    SELF_PERCEPTION = "SELF_PERCEPTION"
    FUTURE_OUTLOOK = "FUTURE_OUTLOOK"
    DECISION_MAKING = "DECISION_MAKING"
    ADAPTIVE_FOLLOWUP = "ADAPTIVE_FOLLOWUP"


# BDI-II item definitions
BDI_ITEMS: dict[int, str] = {
    1: "Sadness",
    2: "Pessimism",
    3: "Past failure",
    4: "Loss of pleasure",
    5: "Guilty feelings",
    6: "Punishment feelings",
    7: "Self-dislike",
    8: "Self-criticalness",
    9: "Suicidal thoughts or wishes",
    10: "Crying",
    11: "Agitation",
    12: "Loss of interest",
    13: "Indecisiveness",
    14: "Worthlessness",
    15: "Loss of energy",
    16: "Changes in sleeping pattern",
    17: "Irritability",
    18: "Changes in appetite",
    19: "Concentration difficulty",
    20: "Tiredness or fatigue",
    21: "Loss of interest in sex",
}

# Item tier classification for Bayesian prior
TIER_1_ITEMS = {1, 2, 4, 12, 15, 17, 20}  # Naturally surface
TIER_2_ITEMS = {3, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19}  # Need steering
TIER_3_ITEMS = {6, 9, 21}  # Hard to elicit

# High-discrimination items (BDI-II Fast Screen)
FAST_SCREEN_ITEMS = {1, 2, 3, 4, 7, 8, 9}

# Assessor cluster assignments
ASSESSOR_ITEMS = {
    "AFFECTIVE": [1, 4, 10, 12, 17],
    "COGNITIVE": [2, 3, 5, 6, 7, 8, 9, 14],
    "SOMATIC": [11, 15, 16, 18, 20],
    "FUNCTIONAL": [13, 19, 21],
}


def score_to_band(total: int) -> SeverityBand:
    if total <= 13:
        return SeverityBand.MINIMAL
    elif total <= 19:
        return SeverityBand.MILD
    elif total <= 28:
        return SeverityBand.MODERATE
    else:
        return SeverityBand.SEVERE


@dataclass
class ItemScore:
    item_id: int
    item_name: str
    score: Optional[int]  # None for NO_EVIDENCE
    confidence: float
    state: ItemState
    evidence: str = ""
    source: str = "assessor"  # assessor | prior | justificator_adjusted


@dataclass
class AssessorOutput:
    assessor_name: str
    items: list[ItemScore]
    cross_observations: str = ""


@dataclass
class LinguisticFeatures:
    """Linguistic features extracted from a single persona response."""
    word_count: int = 0
    sentence_count: int = 0

    # Pronoun ratios
    first_person_singular_ratio: float = 0.0
    first_person_plural_ratio: float = 0.0

    # Absolutist language
    absolutist_count: int = 0
    absolutist_ratio: float = 0.0
    absolutist_words_found: list[str] = field(default_factory=list)

    # Emotion
    negative_emotion_count: int = 0
    positive_emotion_count: int = 0
    sadness_words: list[str] = field(default_factory=list)
    anger_words: list[str] = field(default_factory=list)

    # Cognitive style
    discrepancy_count: int = 0
    tentative_count: int = 0
    hedging_count: int = 0
    coping_count: int = 0

    # Symptom keyword hits
    sleep_keywords: list[str] = field(default_factory=list)
    appetite_keywords: list[str] = field(default_factory=list)
    energy_keywords: list[str] = field(default_factory=list)
    anhedonia_keywords: list[str] = field(default_factory=list)
    worthlessness_keywords: list[str] = field(default_factory=list)
    suicidal_keywords: list[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    role: str  # "user" (interviewer) or "assistant" (persona)
    message: str
    turn_number: int
    linguistic_features: Optional[LinguisticFeatures] = None


@dataclass
class OrchestratorGuidance:
    decision: str  # "CONTINUE" or "TERMINATE"
    next_topic: Optional[TopicArea] = None
    suggested_angle: str = ""
    exploration_gaps: list[str] = field(default_factory=list)
    priority_reasoning: str = ""
    conflict_notes: str = ""
    interviewer_adaptation: str = ""


@dataclass
class JustificatorOutput:
    patterns_detected: list[dict]
    adjustments_made: list[dict]
    final_total: int
    final_band: SeverityBand
    top_4_symptoms: list[dict]
    clinical_narrative: str
    item_scores: dict[str, int]


@dataclass
class PersonaResult:
    persona_id: str  # "00" through "19"
    persona_number: int  # 0 through 19
    conversation: list[ConversationTurn]
    assessor_outputs: dict[str, AssessorOutput]
    linguistic_features_history: list[LinguisticFeatures]
    pass1_total: int = 0
    pass2_total: int = 0
    final_total: int = 0
    final_band: SeverityBand = SeverityBand.MINIMAL
    top_4_symptoms: list[str] = field(default_factory=list)
    justificator_output: Optional[JustificatorOutput] = None
    item_scores: dict[int, ItemScore] = field(default_factory=dict)
