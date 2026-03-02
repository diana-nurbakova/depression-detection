"""Annotation-specific query expansion for candidate retrieval.

Extracts Layer 3 common expressions from symptom definitions
to build focused retrieval queries for annotation candidate selection.
Separate from query_expansion.py to avoid modifying the production
retrieval path.
"""

from __future__ import annotations

import re

from hipert.models import SymptomDefinition


def extract_l3_expressions(discussion_topics: str) -> list[str]:
    """Extract quoted common expressions from Layer 3 discussion topics.

    The L3_discussion field in symptoms.yaml contains lines like:
        "I can start a hundred projects but finishing the boring parts is impossible."
        "Everything is 90% done and nothing is 100% done."

    Args:
        discussion_topics: The L3 discussion topics string.

    Returns:
        List of extracted expression strings (without quotes).
    """
    return re.findall(r'"([^"]+)"', discussion_topics)


def build_annotation_query(
    symptom: SymptomDefinition,
    max_paraphrases: int = 3,
) -> str:
    """Build an expanded query for annotation candidate retrieval.

    Combines the ASRS item text with 2-3 Layer 3 common expressions
    to create a query capturing both clinical and naturalistic vocabulary.

    Args:
        symptom: The symptom definition.
        max_paraphrases: Maximum number of L3 expressions to include.

    Returns:
        Expanded query string suitable for bi-encoder retrieval.
    """
    parts = [symptom.text]

    if symptom.discussion_topics:
        expressions = extract_l3_expressions(symptom.discussion_topics)
        parts.extend(expressions[:max_paraphrases])

    return " ".join(parts)


def build_all_annotation_queries(
    symptoms: dict[int, SymptomDefinition],
    max_paraphrases: int = 3,
) -> dict[int, str]:
    """Build annotation queries for all symptoms.

    Args:
        symptoms: Dictionary mapping item number to SymptomDefinition.
        max_paraphrases: Maximum L3 expressions per query.

    Returns:
        Dictionary mapping item number to expanded query string.
    """
    return {
        item_num: build_annotation_query(symptom, max_paraphrases)
        for item_num, symptom in symptoms.items()
    }
