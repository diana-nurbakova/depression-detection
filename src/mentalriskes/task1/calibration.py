"""Post-assessment calibration layer for MentalRiskES Task 1.

Three strategies:
  1. Flat correction: subtract k from each item (clip to valid range)
  2. Band-aware correction: severity-band-specific corrections
  3. None: raw LLM output
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Instrument specifications
_SPECS = {
    "PHQ-9": {"n_items": 9, "max_val": 3},
    "GAD-7": {"n_items": 7, "max_val": 3},
    "CompACT-10": {"n_items": 10, "max_val": 6},
}

# PHQ-9 severity bands
_PHQ9_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately_severe"),
    (20, 27, "severe"),
]

# GAD-7 severity bands
_GAD7_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 21, "severe"),
]


def _get_band(total: int, bands: list[tuple[int, int, str]]) -> str:
    for low, high, name in bands:
        if low <= total <= high:
            return name
    return bands[-1][2]


def calibrate_flat(
    scores: list[int],
    instrument: str,
    subtract: int = 0,
) -> list[int]:
    """
    Apply flat correction: subtract k from each item.

    Special handling:
    - PHQ-9 item 9 (suicidality): only subtract if score > 1
    - CompACT-10: no correction by default
    """
    spec = _SPECS[instrument]
    result = []
    for i, s in enumerate(scores):
        if instrument == "PHQ-9" and i == 8:
            # Suicidality: conservative, only reduce if clearly over-scored
            adj = max(0, min(spec["max_val"], s - subtract)) if s > 1 else s
        else:
            adj = max(0, min(spec["max_val"], s - subtract))
        result.append(adj)
    return result


def calibrate_band_aware(
    scores: list[int],
    instrument: str,
) -> list[int]:
    """
    Apply band-aware correction based on severity level.

    Over-scoring correction is reduced for moderate-severe cases
    (where the LLM is more likely to be correct).
    """
    spec = _SPECS[instrument]
    total = sum(scores)

    if instrument == "PHQ-9":
        band = _get_band(total, _PHQ9_BANDS)
        corrections = {
            "minimal": 0,
            "mild": 0,
            "moderate": 0,
            "moderately_severe": 0,
            "severe": -1,  # reduce over-scoring at extreme end
        }
        subtract = corrections.get(band, 0)
    elif instrument == "GAD-7":
        band = _get_band(total, _GAD7_BANDS)
        corrections = {
            "minimal": 0,
            "mild": 0,
            "moderate": 0,
            "severe": -1,
        }
        subtract = corrections.get(band, 0)
    else:
        # CompACT-10: no band-aware correction yet
        return scores

    if subtract == 0:
        return scores

    return calibrate_flat(scores, instrument, abs(subtract))


def calibrate_scores(
    scores: list[int],
    instrument: str,
    strategy: str = "none",
    params: dict | None = None,
) -> list[int]:
    """
    Apply calibration to assessment scores.

    Args:
        scores: Raw assessment scores.
        instrument: "PHQ-9", "GAD-7", or "CompACT-10".
        strategy: "flat", "band_aware", or "none".
        params: Additional parameters (e.g., subtract values for flat correction).

    Returns:
        Calibrated scores.
    """
    if strategy == "none":
        return scores

    if strategy == "flat":
        params = params or {}
        if instrument == "PHQ-9":
            subtract = params.get("phq9_subtract", 0)
        elif instrument == "GAD-7":
            subtract = params.get("gad7_subtract", 0)
        else:
            subtract = 0
        return calibrate_flat(scores, instrument, subtract)

    if strategy == "band_aware":
        return calibrate_band_aware(scores, instrument)

    logger.warning("Unknown calibration strategy '%s', returning raw scores", strategy)
    return scores


# ---------------------------------------------------------------------------
# Cross-instrument consistency checks
# ---------------------------------------------------------------------------

def check_cross_instrument_consistency(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
) -> list[dict]:
    """
    Check cross-instrument consistency and return warnings.

    Implements soft constraints from the spec:
    - PHQ-9/GAD-7 comorbidity correlation
    - CompACT-10/PHQ-9 inverse relationship
    - CompACT-10 internal consistency
    """
    warnings = []

    phq9_total = sum(phq9)
    gad7_total = sum(gad7)
    compact_total = sum(compact10)

    # PHQ-9 × GAD-7 comorbidity
    phq9_norm = phq9_total / 27.0
    gad7_norm = gad7_total / 21.0
    if abs(phq9_norm - gad7_norm) > 0.4:
        warnings.append({
            "rule": "PHQ9_GAD7_comorbidity",
            "message": f"Large gap between PHQ-9 ({phq9_total}/27={phq9_norm:.2f}) "
                       f"and GAD-7 ({gad7_total}/21={gad7_norm:.2f})",
        })

    # CompACT-10 × PHQ-9 inverse
    va_items = [compact10[i] for i in [1, 3, 6, 9]]  # items 2,4,7,10 (0-indexed)
    va_mean = sum(va_items) / len(va_items)
    if phq9_total > 15 and va_mean > 4.5:
        warnings.append({
            "rule": "CompACT_PHQ9_inverse",
            "message": f"High PHQ-9 ({phq9_total}) but high Valued Action ({va_mean:.1f})",
        })

    # CompACT-10 × GAD-7 avoidance
    ote_items = [compact10[i] for i in [2, 4, 7]]  # items 3,5,8 (0-indexed)
    ote_mean = sum(ote_items) / len(ote_items)
    if gad7_total > 15 and ote_mean < 2:
        warnings.append({
            "rule": "CompACT_GAD7_avoidance",
            "message": f"High GAD-7 ({gad7_total}) but low Openness ({ote_mean:.1f})",
        })

    # CompACT-10 internal consistency (within subprocess)
    for name, indices in [("OtE", [2, 4, 7]), ("BA", [0, 5, 8]), ("VA", [1, 3, 6, 9])]:
        sub_scores = [compact10[i] for i in indices]
        if max(sub_scores) - min(sub_scores) > 2:
            warnings.append({
                "rule": "CompACT_internal_consistency",
                "message": f"CompACT-10 {name} items vary by >2: {sub_scores}",
            })

    # PHQ-9 item 9 safety check
    if phq9[8] > 0:
        warnings.append({
            "rule": "PHQ9_item9_threshold",
            "message": f"PHQ-9 item 9 (suicidality) scored {phq9[8]} — verify evidence",
        })

    return warnings


def load_calibration_config(path: str | Path) -> dict:
    """Load calibration configuration from JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Calibration config not found at %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
