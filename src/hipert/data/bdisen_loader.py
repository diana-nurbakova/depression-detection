"""Load and filter BDI-Sen-2.0 annotations for graded confounder extraction.

BDI-Sen-2.0 contains 2,529 unique Reddit sentences with 5,003 (sentence,
symptom) annotations across all 21 BDI-II symptoms. Each annotation includes
a severity label (0-3) on the same scale as our ADHD scoring.

Severity-1 sentences for overlapping symptoms are calibrated score-1
confounders: they sit at the mildest end of depression relevance, making
them maximally ambiguous for ADHD.

Data files:
    data/BDI-Sen/full_dataset/bdi_majority_vote.jsonl — flat annotations
    data/BDI-Sen/full_dataset/bdi_unified.jsonl       — grouped by sentence
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hipert.models import FIRST_PERSON_MARKERS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BDISenAnnotation:
    """One (sentence, symptom) annotation from BDI-Sen-2.0."""

    sentence: str         # Full sentence text
    symptom: str          # e.g. "Concentration_difficulty"
    severity: int | None  # 0, 1, 2, 3, or None
    label: int            # 0 = not relevant, 1 = relevant

    @property
    def has_first_person(self) -> bool:
        """Check for first-person markers in the sentence text."""
        tokens = set(self.sentence.lower().split())
        return bool(tokens & FIRST_PERSON_MARKERS)


def load_annotations(filepath: Path) -> list[BDISenAnnotation]:
    """Load all annotations from bdi_majority_vote.jsonl.

    Args:
        filepath: Path to bdi_majority_vote.jsonl.

    Returns:
        List of all annotation records.
    """
    annotations: list[BDISenAnnotation] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            annotations.append(BDISenAnnotation(
                sentence=row["sentence"],
                symptom=row["symptom"],
                severity=row.get("severity"),
                label=int(row["label"]),
            ))

    logger.info("Loaded %d BDI-Sen annotations from %s", len(annotations), filepath)
    return annotations


def load_annotations_by_symptom(
    filepath: Path,
) -> dict[str, list[BDISenAnnotation]]:
    """Load annotations grouped by BDI-II symptom name.

    Args:
        filepath: Path to bdi_majority_vote.jsonl.

    Returns:
        Dict mapping symptom name to list of annotations.
    """
    all_annotations = load_annotations(filepath)
    grouped: dict[str, list[BDISenAnnotation]] = defaultdict(list)

    for ann in all_annotations:
        grouped[ann.symptom].append(ann)

    for symptom, anns in sorted(grouped.items()):
        n_positive = sum(1 for a in anns if a.label == 1)
        logger.debug(
            "  BDI-Sen %s: %d total, %d label=1",
            symptom, len(anns), n_positive,
        )

    return dict(grouped)


def filter_confounder_candidates(
    annotations: list[BDISenAnnotation],
    bdisen_symptom: str,
    severity_levels: tuple[int, ...] = (1,),
    require_first_person: bool = True,
) -> list[BDISenAnnotation]:
    """Filter annotations to confounder candidates for a BDI-Sen symptom.

    Applies three filters:
    1. Matches the given BDI-Sen symptom name
    2. label=1 (annotator confirmed as relevant)
    3. Severity in the specified levels (default: severity=1 only)
    4. Optionally, first-person markers present

    Args:
        annotations: Full list of annotations (any symptom).
        bdisen_symptom: BDI-Sen symptom name (e.g. "Concentration_difficulty").
        severity_levels: Tuple of severity values to include.
            Default (1,) selects only mild — best ADHD confounders.
            Use (1, 2) to include moderate, (1, 2, 3) for all positive.
        require_first_person: Whether to require first-person markers.

    Returns:
        Filtered list suitable as graded score-1 confounders.
    """
    candidates = [
        ann for ann in annotations
        if (ann.symptom == bdisen_symptom
            and ann.label == 1
            and ann.severity in severity_levels)
    ]

    if require_first_person:
        before = len(candidates)
        candidates = [ann for ann in candidates if ann.has_first_person]
        logger.debug(
            "BDI-Sen %s (severity %s): %d label=1, %d after first-person filter",
            bdisen_symptom, severity_levels, before, len(candidates),
        )

    return candidates
