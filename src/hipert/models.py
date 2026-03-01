"""Core data models for the HiPerT-ADHD pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


# ---------- Enums ----------


class SymptomFactor(str, Enum):
    """Three-factor bifactor model (Panagiotidi et al. 2024)."""

    INATTENTION = "Inattention"
    MOTOR_HI = "Motor_HI"
    VERBAL_HI = "Verbal_HI"


class SymptomSubcluster(str, Enum):
    ORGANIZATION_PLANNING = "Organization/Planning"
    MEMORY_AVOIDANCE = "Memory/Avoidance"
    SUSTAINED_ATTENTION = "Sustained_Attention/Distractibility"
    FIDGETING_RESTLESSNESS = "Fidgeting/Restlessness"
    INTERNAL_DRIVE = "Internal_Drive/Settling"
    OUTPUT_CONTROL = "Output_Control"
    TURN_TAKING = "Turn-Taking/Interrupting"


class SymptomMatch(str, Enum):
    YES = "YES"
    PARTIAL = "PARTIAL"
    NO = "NO"


class SelfReference(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    NONE = "NONE"


class DetailLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class RelevanceScore(IntEnum):
    IRRELEVANT = 0
    MARGINAL = 1
    MODERATE = 2
    HIGH = 3


# ---------- First-person markers ----------

FIRST_PERSON_MARKERS = frozenset({
    "i", "me", "my", "mine", "myself",
    "i'm", "i've", "i'd", "i'll",
})


# ---------- Data classes ----------


@dataclass(frozen=True)
class Sentence:
    """A single sentence from the TREC corpus with context triplet."""

    docno: str
    pre: str
    text: str
    post: str
    file_id: str

    @property
    def user_id(self) -> str:
        """Extract userId from DOCNO (format: userId_contextId_sentenceIdx)."""
        return self.docno.rsplit("_", 2)[0]

    @property
    def has_first_person(self) -> bool:
        """Check for first-person markers in the target text."""
        tokens = set(self.text.lower().split())
        return bool(tokens & FIRST_PERSON_MARKERS)


@dataclass
class SymptomDefinition:
    """One of the 18 ASRS symptoms with all metadata."""

    item_number: int
    text: str
    factor: SymptomFactor
    subcluster: SymptomSubcluster
    clinical_definition: str        # Layer 1
    adult_manifestation: str        # Layer 2
    discussion_topics: str          # Layer 3
    differential_markers: str       # Layer 4
    token_budget: str               # "full_4", "compressed_3", "minimal_2"
    keywords: list[str] = field(default_factory=list)
    expected_reliability: str = "MEDIUM"
    symptom_weight: float = 1.0
    clinical_difficulty_prior: float = 0.5

    @property
    def is_inattention_cluster(self) -> bool:
        return self.item_number in {7, 8, 9, 10, 11}


@dataclass
class FewShotExample:
    """One few-shot example for a symptom at a given score level."""

    score: int
    pre: str
    text: str
    post: str
    symptom_match: str
    self_reference: str
    detail_level: str
    confounders: str
    confidence: int
    reasoning: str


@dataclass
class LLMOutput:
    """Structured output from LLM scoring (7 fields)."""

    symptom_match: str      # YES / PARTIAL / NO
    self_reference: str     # DIRECT / INDIRECT / NONE
    detail_level: str       # HIGH / MEDIUM / LOW / NONE
    confounders: str        # "NONE" or free text
    score: int              # 0-3
    confidence: int         # 1-5
    reasoning: str
    raw_text: str = ""      # Original LLM response for debugging


@dataclass
class ScoringResult:
    """Complete scoring result for one (sentence, symptom) pair."""

    sentence_id: str
    symptom_id: int
    llama_output: LLMOutput
    gpt_output: Optional[LLMOutput] = None
    escalated: bool = False
    escalation_triggers: list[str] = field(default_factory=list)
    final_label: int = 0
    confidence_weight: float = 0.0
    calibrated_label: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        d = {
            "sentence_id": self.sentence_id,
            "symptom_id": self.symptom_id,
            "llama_output": {
                "symptom_match": self.llama_output.symptom_match,
                "self_reference": self.llama_output.self_reference,
                "detail_level": self.llama_output.detail_level,
                "confounders": self.llama_output.confounders,
                "score": self.llama_output.score,
                "confidence": self.llama_output.confidence,
                "reasoning": self.llama_output.reasoning,
                "raw_text": self.llama_output.raw_text,
            },
            "escalated": self.escalated,
            "escalation_triggers": self.escalation_triggers,
            "final_label": self.final_label,
            "confidence_weight": self.confidence_weight,
        }
        if self.gpt_output is not None:
            d["gpt_output"] = {
                "symptom_match": self.gpt_output.symptom_match,
                "self_reference": self.gpt_output.self_reference,
                "detail_level": self.gpt_output.detail_level,
                "confounders": self.gpt_output.confounders,
                "score": self.gpt_output.score,
                "confidence": self.gpt_output.confidence,
                "reasoning": self.gpt_output.reasoning,
                "raw_text": self.gpt_output.raw_text,
            }
        else:
            d["gpt_output"] = None
        if self.calibrated_label is not None:
            d["calibrated_label"] = self.calibrated_label
        return d


@dataclass
class CandidateScore:
    """A sentence scored for a specific symptom during retrieval."""

    sentence: Sentence
    symptom_id: int
    retrieval_score: float
    keyword_boost: float = 0.0

    @property
    def combined_score(self) -> float:
        return self.retrieval_score + self.keyword_boost
