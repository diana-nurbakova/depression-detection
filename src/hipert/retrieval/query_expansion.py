"""Symptom query expansion for bi-encoder retrieval.

Combines the ASRS item text with Layer 3 naturalistic vocabulary
from symptom definitions to create expanded queries.
"""

from __future__ import annotations

from hipert.models import SymptomDefinition


def build_expanded_query(symptom: SymptomDefinition) -> str:
    """Build an expanded query for bi-encoder retrieval.

    Combines the ASRS questionnaire text with Layer 3 discussion
    topics to capture naturalistic vocabulary used online.

    Args:
        symptom: The symptom definition.

    Returns:
        Expanded query string for similarity search.
    """
    parts = [symptom.text]

    # Add discussion topics as additional query context
    if symptom.discussion_topics:
        # Take a concise excerpt — first 200 chars of discussion topics
        excerpt = symptom.discussion_topics[:200]
        # Trim at last complete sentence or phrase
        last_period = excerpt.rfind(".")
        if last_period > 50:
            excerpt = excerpt[:last_period + 1]
        parts.append(excerpt)

    return " ".join(parts)


def build_all_queries(
    symptoms: dict[int, SymptomDefinition],
) -> dict[int, str]:
    """Build expanded queries for all symptoms.

    Args:
        symptoms: Dictionary mapping item number to SymptomDefinition.

    Returns:
        Dictionary mapping item number to expanded query string.
    """
    return {
        item_num: build_expanded_query(symptom)
        for item_num, symptom in symptoms.items()
    }
