"""Candidate filters: first-person filter and keyword boost.

Applied after bi-encoder retrieval to refine candidate selection.
"""

from __future__ import annotations

from hipert.models import FIRST_PERSON_MARKERS, CandidateScore, Sentence


def apply_first_person_filter(
    candidates: list[CandidateScore],
) -> list[CandidateScore]:
    """Remove candidates without first-person markers.

    Enforces the 'writer's own state' requirement from the spec.
    First-person markers: I, me, my, mine, myself, I'm, I've, I'd, I'll.
    """
    return [c for c in candidates if c.sentence.has_first_person]


def apply_keyword_boost(
    candidates: list[CandidateScore],
    keywords: list[str],
    boost_value: float = 0.05,
) -> list[CandidateScore]:
    """Add a score bonus for keyword matches.

    Each candidate gets +boost_value if any keyword from the symptom's
    keyword cluster appears in the sentence text.

    Args:
        candidates: List of candidate scores.
        keywords: Keywords for the symptom cluster.
        boost_value: Score bonus per match (default: 0.05).

    Returns:
        Updated candidates with keyword_boost field set.
    """
    keyword_set = {kw.lower() for kw in keywords}

    boosted: list[CandidateScore] = []
    for c in candidates:
        text_lower = c.sentence.text.lower()
        if any(kw in text_lower for kw in keyword_set):
            boosted.append(CandidateScore(
                sentence=c.sentence,
                symptom_id=c.symptom_id,
                retrieval_score=c.retrieval_score,
                keyword_boost=boost_value,
            ))
        else:
            boosted.append(c)

    return boosted
