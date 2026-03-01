"""Confidence weight computation.

Composite weight = resolution_weight * symptom_weight.
This module provides utilities for working with confidence weights
across the pipeline.
"""

from __future__ import annotations


# Symptom weights from spec Section 6.2
SYMPTOM_WEIGHTS = {
    5: 1.0, 6: 1.0, 12: 1.0, 13: 1.0, 14: 1.0,   # Motor H/I
    15: 0.9, 16: 0.9, 17: 0.9, 18: 0.9,             # Verbal H/I
    1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7,                 # Organization/Memory
    7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,       # Sustained Attention
}


def get_symptom_weight(symptom_id: int) -> float:
    """Get the symptom reliability weight."""
    return SYMPTOM_WEIGHTS.get(symptom_id, 0.7)
