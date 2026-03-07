"""Load and filter RedSM5 depression annotations for confounder extraction.

RedSM5 contains 2,058 sentence-level DSM-5 depression annotations by a
clinical psychologist. Each annotation includes symptom label, binary status,
and a clinical explanation. We use status=1 sentences from ADHD-overlapping
categories (COGNITIVE_ISSUES, PSYCHOMOTOR, FATIGUE, SLEEP_ISSUES, ANHEDONIA)
as score-1 confounder candidates for the ASRS annotation protocol.

Data files:
    data/RedSM5/redsm5_annotations.csv — sentence-level annotations
    data/RedSM5/redsm5_posts.csv       — full post texts (not needed here)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hipert.models import FIRST_PERSON_MARKERS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedSM5Annotation:
    """One sentence-level annotation from the RedSM5 dataset."""

    post_id: str          # e.g. "s_1270_9"
    sentence_id: str      # e.g. "s_1270_9_6"
    sentence_text: str    # Full sentence text
    dsm5_symptom: str     # e.g. "COGNITIVE_ISSUES"
    status: int           # 0 = not relevant, 1 = relevant
    explanation: str      # Clinician's explanation of the annotation

    @property
    def has_first_person(self) -> bool:
        """Check for first-person markers in the sentence text."""
        tokens = set(self.sentence_text.lower().split())
        return bool(tokens & FIRST_PERSON_MARKERS)


def load_annotations(filepath: Path) -> list[RedSM5Annotation]:
    """Load all annotations from redsm5_annotations.csv.

    Args:
        filepath: Path to redsm5_annotations.csv.

    Returns:
        List of all annotation records.
    """
    annotations: list[RedSM5Annotation] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(RedSM5Annotation(
                post_id=row["post_id"].strip(),
                sentence_id=row["sentence_id"].strip(),
                sentence_text=row["sentence_text"].strip(),
                dsm5_symptom=row["DSM5_symptom"].strip(),
                status=int(row["status"].strip()),
                explanation=row["explanation"].strip(),
            ))

    logger.info("Loaded %d RedSM5 annotations from %s", len(annotations), filepath)
    return annotations


def load_annotations_by_symptom(
    filepath: Path,
) -> dict[str, list[RedSM5Annotation]]:
    """Load annotations grouped by DSM-5 symptom category.

    Args:
        filepath: Path to redsm5_annotations.csv.

    Returns:
        Dict mapping DSM5_symptom name to list of annotations.
    """
    all_annotations = load_annotations(filepath)
    grouped: dict[str, list[RedSM5Annotation]] = defaultdict(list)

    for ann in all_annotations:
        grouped[ann.dsm5_symptom].append(ann)

    for symptom, anns in sorted(grouped.items()):
        n_positive = sum(1 for a in anns if a.status == 1)
        logger.info(
            "  RedSM5 %s: %d total, %d status=1",
            symptom, len(anns), n_positive,
        )

    return dict(grouped)


def filter_confounder_candidates(
    annotations: list[RedSM5Annotation],
    dsm5_category: str,
    require_first_person: bool = True,
) -> list[RedSM5Annotation]:
    """Filter annotations to confounder candidates for a DSM-5 category.

    Applies two filters:
    1. status=1 (clinician confirmed as relevant to the DSM-5 category)
    2. Optionally, first-person markers present in sentence text

    Args:
        annotations: Full list of annotations (any category).
        dsm5_category: DSM-5 category to filter for (e.g. "COGNITIVE_ISSUES").
        require_first_person: Whether to require first-person markers.

    Returns:
        Filtered list of annotations suitable as score-1 confounders.
    """
    # Filter by category and status
    candidates = [
        ann for ann in annotations
        if ann.dsm5_symptom == dsm5_category and ann.status == 1
    ]

    if require_first_person:
        before = len(candidates)
        candidates = [ann for ann in candidates if ann.has_first_person]
        logger.debug(
            "RedSM5 %s: %d status=1, %d after first-person filter",
            dsm5_category, before, len(candidates),
        )

    return candidates
