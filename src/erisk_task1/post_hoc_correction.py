"""Post-hoc score correction for BDI-II totals.

Applies a deterministic correction to the raw BDI-II total score
to compensate for systematic over-scoring bias identified in the
TalkDep ablation study. See specs/posthoc_correction_spec.md.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

from .models import SeverityBand, score_to_band

logger = logging.getLogger(__name__)


class CorrectionStrategy(Enum):
    NONE = "none"
    MINUS_5 = "minus_5"
    BAND_AWARE = "band_aware"
    PROGRESSIVE = "progressive"
    FLAT_MINUS_2 = "flat_minus_2"
    FLAT_MINUS_3 = "flat_minus_3"
    PROPORTIONAL_085 = "proportional_085"


CORRECTIONS: dict[CorrectionStrategy, Callable[[int], int]] = {
    CorrectionStrategy.NONE: lambda s: s,
    CorrectionStrategy.MINUS_5: lambda s: max(0, s - 5),
    CorrectionStrategy.BAND_AWARE: lambda s: (
        max(0, s - 4) if s <= 18
        else (max(0, s - 5) if s <= 32
              else s - 1)
    ),
    CorrectionStrategy.PROGRESSIVE: lambda s: (
        max(0, s - 2) if s <= 13
        else (max(0, s - 3) if s <= 19
              else (max(0, s - 4) if s <= 28
                    else max(0, s - 2)))
    ),
    CorrectionStrategy.FLAT_MINUS_2: lambda s: max(0, s - 2),
    CorrectionStrategy.FLAT_MINUS_3: lambda s: max(0, s - 3),
    CorrectionStrategy.PROPORTIONAL_085: lambda s: round(s * 0.85),
}

# Default strategy per run ID
RUN_DEFAULTS: dict[int, CorrectionStrategy] = {
    1: CorrectionStrategy.BAND_AWARE,
    2: CorrectionStrategy.FLAT_MINUS_2,
    3: CorrectionStrategy.FLAT_MINUS_3,
}


def apply_correction(raw_total: int, strategy: CorrectionStrategy) -> dict:
    """Apply post-hoc correction and return corrected result with audit info."""
    correct_fn = CORRECTIONS[strategy]
    corrected = correct_fn(raw_total)

    raw_band = score_to_band(raw_total)
    corrected_band = score_to_band(corrected)

    result = {
        "raw_total": raw_total,
        "raw_band": raw_band.value,
        "correction_strategy": strategy.value,
        "correction_delta": corrected - raw_total,
        "corrected_total": corrected,
        "corrected_band": corrected_band.value,
        "band_changed": raw_band != corrected_band,
    }

    logger.info(
        "Post-hoc correction (%s): %d (%s) -> %d (%s) [delta=%d]",
        strategy.value, raw_total, raw_band.value,
        corrected, corrected_band.value, corrected - raw_total,
    )

    return result


def get_strategy_for_run(run_id: int, config_override: str | None = None) -> CorrectionStrategy:
    """Get the correction strategy for a given run ID.

    Args:
        run_id: The run number (1, 2, or 3).
        config_override: Optional strategy name from config (overrides default).
    """
    if config_override:
        return CorrectionStrategy(config_override)
    return RUN_DEFAULTS.get(run_id, CorrectionStrategy.NONE)
