"""Post-assessment calibration layer for MentalRiskES Task 1.

Three levels of calibration (spec: mentalriskes2026_constraints_ablation_spec.md):
  Level A: Prompt anchors embedded in assessor prompts (zero additional cost).
  Level B: Rule-based post-assessment constraints (7 rules, no LLM call).
  Level C: Iterative self-critique LLM calibration agent (~1,000 tokens/call).

Simple per-item strategies (backward compatible):
  flat:      subtract k from each item (clip to valid range)
  band_aware: severity-band-specific corrections
  none:      raw LLM output
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------

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

# CompACT-10 subscale item indices (0-indexed)
_OTE_IDX = [2, 4, 7]      # items 3, 5, 8 (Openness to Experience)
_BA_IDX = [0, 5, 8]       # items 1, 6, 9 (Behavioral Awareness)
_VA_IDX = [1, 3, 6, 9]    # items 2, 4, 7, 10 (Valued Action)

# PHQ-9 somatic items (0-indexed): sleep, fatigue, appetite, psychomotor
_PHQ9_SOMATIC_IDX = [2, 3, 4, 7]  # items 3, 4, 5, 8

# GAD-7 somatic items (0-indexed): nervousness, relaxation, restlessness
_GAD7_SOMATIC_IDX = [0, 3, 4]  # items 1, 4, 5

# Expected CompACT-10 subscale per-item mean ranges conditional on distress level.
# Source: spec section 2.3, derived from Francis et al. (2016), Golijani-Moghaddam et al. (2023).

# VA (Valued Action) expected per-item means — WEAKER constraint due to self-contradiction profile.
_VA_EXPECTED: dict[str, tuple[float, float]] = {
    "minimal":            (3.5, 6.0),
    "mild":               (3.0, 5.5),
    "moderate":           (2.0, 4.5),
    "moderately_severe":  (1.5, 4.0),
    "severe":             (1.0, 3.5),
}

# OtE (Openness to Experience) expected per-item means — STRONGEST constraint.
_OTE_EXPECTED: dict[str, tuple[float, float]] = {
    "minimal":            (0.5, 3.0),
    "mild":               (1.5, 4.0),
    "moderate":           (2.5, 5.0),
    "moderately_severe":  (3.0, 5.5),
    "severe":             (3.5, 6.0),
}

# Self-contradiction guard: if OtE per-item mean is below this threshold, the high VA
# is consistent with the self-contradiction profile (real clinical pattern, ~19% of patients).
# Do NOT correct VA in this case. Source: spec appendix A.3.
_SELF_CONTRADICTION_OTE_THRESHOLD = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_band(total: int, bands: list[tuple[int, int, str]]) -> str:
    for low, high, name in bands:
        if low <= total <= high:
            return name
    return bands[-1][2]


def _phq9_band(total: int) -> str:
    return _get_band(total, _PHQ9_BANDS)


def _gad7_band(total: int) -> str:
    return _get_band(total, _GAD7_BANDS)


def _distress_band(phq9_total: int, gad7_total: int) -> str:
    """Combined distress severity band using PHQ-9 as primary (best-calibrated).

    Falls back to GAD-7 if PHQ-9 indicates minimal but GAD-7 indicates otherwise.
    """
    phq9_b = _phq9_band(phq9_total)
    gad7_b = _gad7_band(gad7_total)
    if phq9_b == "minimal" and gad7_b not in ("minimal",):
        return gad7_b  # GAD-7 indicates more distress
    return phq9_b


def _subscale_mean(scores: list[int], indices: list[int]) -> float:
    sub = [scores[i] for i in indices]
    return sum(sub) / len(sub)


# ---------------------------------------------------------------------------
# Simple per-item calibration strategies (backward compatible)
# ---------------------------------------------------------------------------

def calibrate_flat(
    scores: list[int],
    instrument: str,
    subtract: int = 0,
) -> list[int]:
    """Apply flat correction: subtract k from each item.

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
    """Apply band-aware correction based on severity level."""
    spec = _SPECS[instrument]
    total = sum(scores)

    if instrument == "PHQ-9":
        band = _phq9_band(total)
        corrections = {
            "minimal": 0,
            "mild": 0,
            "moderate": 0,
            "moderately_severe": 0,
            "severe": -1,
        }
        subtract = corrections.get(band, 0)
    elif instrument == "GAD-7":
        band = _gad7_band(total)
        corrections = {
            "minimal": 0,
            "mild": 0,
            "moderate": 0,
            "severe": -1,
        }
        subtract = corrections.get(band, 0)
    else:
        # CompACT-10: handled by Level B constraints
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
    """Apply simple per-item calibration to assessment scores.

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
# Level B: Rule-Based Post-Assessment Constraints
# ---------------------------------------------------------------------------

@dataclass
class ConstraintViolation:
    """A single Level B constraint violation."""
    rule: str          # "C1" through "C7"
    severity: str      # "high" | "medium"
    message: str
    correction_applied: bool = False
    correction_detail: dict = field(default_factory=dict)


def apply_level_b_constraints(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
) -> tuple[list[int], list[int], list[int], list[ConstraintViolation]]:
    """Apply the 7-rule Level B psychometric constraint system.

    Rules from spec section 3.2:
      C1: PHQ-9/GAD-7 normalized discordance > 0.40  [High]
      C2: |PHQ-9 - GAD-7| > 8 points                 [Medium]
      C3: PHQ-9 somatic vs GAD-7 somatic mean diff > 1.5  [Medium]
      C4: CompACT VA mean > expected + 1.0 AND NOT self-contradiction  [High] → apply -1 to VA
      C5: CompACT OtE mean < expected - 0.5 for distress level         [Medium] → flag only
      C6: CompACT within-subprocess spread > 3                         [Medium] → flag only
      C7: PHQ-9 item 9 > 0 with total < 10                             [High] → flag only

    Returns:
        (phq9_out, gad7_out, compact10_out, violations)
        Only C4 may modify scores; all others are flag-only.
    """
    violations: list[ConstraintViolation] = []
    phq9_out = list(phq9)
    gad7_out = list(gad7)
    compact10_out = list(compact10)

    phq9_total = sum(phq9)
    gad7_total = sum(gad7)

    # C1: Normalized discordance
    phq9_norm = phq9_total / 27.0
    gad7_norm = gad7_total / 21.0
    if abs(phq9_norm - gad7_norm) > 0.40:
        violations.append(ConstraintViolation(
            rule="C1",
            severity="high",
            message=(
                f"PHQ-9/GAD-7 normalized discordance={abs(phq9_norm - gad7_norm):.2f} > 0.40 "
                f"(PHQ-9={phq9_total}/27={phq9_norm:.2f}, GAD-7={gad7_total}/21={gad7_norm:.2f}). "
                "Review both instruments for evidence consistency."
            ),
        ))

    # C2: Absolute gap
    if abs(phq9_total - gad7_total) > 8:
        violations.append(ConstraintViolation(
            rule="C2",
            severity="medium",
            message=(
                f"|PHQ-9 - GAD-7| = {abs(phq9_total - gad7_total)} > 8 "
                f"(PHQ-9={phq9_total}, GAD-7={gad7_total})"
            ),
        ))

    # C3: Somatic items tracking
    if len(phq9) >= 9 and len(gad7) >= 7:
        phq9_somatic_mean = _subscale_mean(phq9, _PHQ9_SOMATIC_IDX)
        gad7_somatic_mean = _subscale_mean(gad7, _GAD7_SOMATIC_IDX)
        if abs(phq9_somatic_mean - gad7_somatic_mean) > 1.5:
            violations.append(ConstraintViolation(
                rule="C3",
                severity="medium",
                message=(
                    f"PHQ-9 somatic mean={phq9_somatic_mean:.2f} vs "
                    f"GAD-7 somatic mean={gad7_somatic_mean:.2f} "
                    f"(diff={abs(phq9_somatic_mean - gad7_somatic_mean):.2f} > 1.5). "
                    "Somatic factors should track closely."
                ),
            ))

    # C4: CompACT VA over-scoring with self-contradiction guard
    if len(compact10) >= 10:
        distress_band = _distress_band(phq9_total, gad7_total)
        va_mean = _subscale_mean(compact10, _VA_IDX)
        ote_mean = _subscale_mean(compact10, _OTE_IDX)

        va_expected_low, va_expected_high = _VA_EXPECTED.get(distress_band, (2.0, 4.5))
        ote_expected_low, _ = _OTE_EXPECTED.get(distress_band, (2.5, 5.0))

        if va_mean > va_expected_high + 1.0:
            # Check self-contradiction profile guard
            is_self_contradiction = ote_mean < _SELF_CONTRADICTION_OTE_THRESHOLD

            if is_self_contradiction:
                # High VA + low OtE = self-contradiction profile (real clinical pattern)
                violations.append(ConstraintViolation(
                    rule="C4",
                    severity="high",
                    message=(
                        f"VA mean={va_mean:.2f} > expected_high({va_expected_high:.1f}) + 1.0, "
                        f"but OtE mean={ote_mean:.2f} < {_SELF_CONTRADICTION_OTE_THRESHOLD} → "
                        "self-contradiction profile detected. VA NOT corrected "
                        "(high VA + low OtE is a real clinical pattern ~19% of patients)."
                    ),
                ))
            else:
                # High VA + high OtE = LLM over-scoring (within-session engagement conflation)
                # Apply -1 to VA items
                corrected_va_indices = []
                for idx in _VA_IDX:
                    old = compact10_out[idx]
                    compact10_out[idx] = max(0, old - 1)
                    if compact10_out[idx] != old:
                        corrected_va_indices.append((idx + 1, old, compact10_out[idx]))

                violations.append(ConstraintViolation(
                    rule="C4",
                    severity="high",
                    message=(
                        f"VA mean={va_mean:.2f} > expected_high({va_expected_high:.1f}) + 1.0 "
                        f"AND OtE mean={ote_mean:.2f} >= {_SELF_CONTRADICTION_OTE_THRESHOLD} "
                        f"(distress_band={distress_band}). "
                        "VA over-scoring likely: within-session engagement conflated with general patterns. "
                        f"Applied -1 to VA items: {corrected_va_indices}"
                    ),
                    correction_applied=True,
                    correction_detail={
                        "items_corrected": corrected_va_indices,
                        "va_mean_before": va_mean,
                        "va_mean_after": _subscale_mean(compact10_out, _VA_IDX),
                    },
                ))

        # C5: OtE below expected (flag only, do NOT auto-correct)
        ote_expected_low_for_band, _ = _OTE_EXPECTED.get(distress_band, (2.5, 5.0))
        if ote_mean < ote_expected_low_for_band - 0.5:
            violations.append(ConstraintViolation(
                rule="C5",
                severity="medium",
                message=(
                    f"OtE mean={ote_mean:.2f} < expected_low({ote_expected_low_for_band:.1f}) - 0.5 "
                    f"for distress_band={distress_band}. "
                    "Lower avoidance than expected for this distress level. Flag only (not corrected)."
                ),
            ))

        # C6: Within-subprocess spread > 3
        for name, indices in [("OtE", _OTE_IDX), ("BA", _BA_IDX), ("VA", _VA_IDX)]:
            sub = [compact10_out[i] for i in indices]
            spread = max(sub) - min(sub)
            if spread > 3:
                violations.append(ConstraintViolation(
                    rule="C6",
                    severity="medium",
                    message=(
                        f"CompACT-10 {name} items spread={spread} > 3: {sub} "
                        f"(items {[i + 1 for i in indices]}). "
                        "Items within the same subprocess should not vary by >3."
                    ),
                ))

    # C7: PHQ-9 item 9 safety (suicidality)
    if len(phq9) >= 9 and phq9[8] > 0 and phq9_total < 10:
        violations.append(ConstraintViolation(
            rule="C7",
            severity="high",
            message=(
                f"PHQ-9 item 9 (suicidality) scored {phq9[8]} but PHQ-9 total={phq9_total} < 10. "
                "Suicidality score requires strong explicit evidence. Verify."
            ),
        ))

    for v in violations:
        log_fn = logger.warning if v.severity == "high" else logger.info
        log_fn("Level B %s [%s]: %s", v.rule, v.severity, v.message)

    return phq9_out, gad7_out, compact10_out, violations


# ---------------------------------------------------------------------------
# Level B basic check (backward compatible — no corrections applied)
# ---------------------------------------------------------------------------

def check_cross_instrument_consistency(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
) -> list[dict]:
    """Check cross-instrument consistency and return warnings (no corrections).

    Thin wrapper around Level B for backward compatibility.
    Returns list of dicts with 'rule' and 'message' keys.
    """
    _, _, _, violations = apply_level_b_constraints(phq9, gad7, compact10)
    return [{"rule": v.rule, "message": v.message} for v in violations]


# ---------------------------------------------------------------------------
# Level C: LLM Calibration Agent
# ---------------------------------------------------------------------------

_LEVEL_C_SYSTEM_PROMPT = """\
You are a psychometric calibration agent for multi-instrument clinical assessments. \
You receive raw scores from three instruments (PHQ-9, GAD-7, CompACT-10) assessed \
from a Spanish therapeutic conversation, plus rule-based constraint violations and a \
condensed assessor reasoning summary.

Your task: decide whether to apply small corrections to improve cross-instrument \
consistency. You are the last line of context-sensitive calibration.

KEY PSYCHOMETRIC FACTS:

1. PHQ-9 × GAD-7: Spearman rho=0.74; 78.4% of pairs within 4 points; 56.4% same \
severity class. Somatic factors strongly overlap: PHQ-9 items 3,4,5,8 \
(sleep/fatigue/appetite/psychomotor) and GAD-7 items 1,4,5 \
(nervousness/relaxation/restlessness) should track closely (|mean_diff| ≤ 1.5).

2. CompACT-10 × distress: r ≈ -0.55 with PHQ-9, -0.50 with GAD-7. \
OtE (Openness to Experience, items 3,5,8) is the STRONGEST constraint (r≈0.45). \
High distress → high OtE scores (high avoidance/fusion). \
VA (Valued Action, items 2,4,7,10) is a WEAKER constraint because of the \
self-contradiction profile (see below).

3. SELF-CONTRADICTION PROFILE (CRITICAL GUARD): High VA + high distress is a REAL \
clinical pattern (~19% of patients). These patients act on their values but cannot \
face difficult internal experiences. DISCRIMINATING FEATURE: In this profile, \
OtE per-item mean is LOW (< 2.5), indicating genuine acceptance/openness. \
If VA is high AND OtE < 2.5 per-item: DO NOT correct VA — this is genuine \
self-contradiction. \
If VA is high AND OtE ≥ 2.5 per-item: VA is implausible (LLM conflated \
within-session willingness with general life patterns) — apply -1 to VA items.

4. VA OVER-SCORING ROOT CAUSE: LLMs interpret within-session therapeutic engagement \
("lo intentaré", patient reads despite fear, brief compliance during session) as \
established values-aligned behavior patterns. This inflates VA items 2, 4, 7, 10. \
Only score VA ≥ 5 with strong evidence of values-aligned action OUTSIDE the session.

CORRECTION RULES (strictly follow):
- When in doubt, do NOT correct.
- Never correct more than 2 points on any single item.
- Never override the self-contradiction guard.
- Evaluate assessor reasoning consistency: if the assessor says "insufficient evidence" \
  but scores 2, trust the reasoning over the score (correct down).
- Consider clinical exceptions (comorbid PTSD, cultural context) before correcting.
- All corrections must be logged with explicit rationale.

OUTPUT FORMAT (respond with ONLY valid JSON):
{
  "concordance_assessment": {
    "phq9_gad7": "concordant|borderline|discordant",
    "compact_phq9": "concordant|borderline|discordant",
    "compact_gad7": "concordant|borderline|discordant"
  },
  "corrections": [
    {
      "instrument": "CompACT-10",
      "items_1indexed": [2, 4, 7, 10],
      "direction": "decrease",
      "amount": 1,
      "rationale": "..."
    }
  ],
  "corrected_phq9": [/* 9 integers 0-3 */],
  "corrected_gad7": [/* 7 integers 0-3 */],
  "corrected_compact10": [/* 10 integers 0-6 */]
}
If no corrections are needed, return "corrections": [] and copy input scores verbatim."""


def _build_level_c_input(
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
    violations: list[ConstraintViolation],
    assessments: dict[str, Any] | None = None,
) -> str:
    """Build the user-turn input for the Level C calibration agent."""
    phq9_total = sum(phq9)
    gad7_total = sum(gad7)
    compact10_total = sum(compact10)
    distress_band = _distress_band(phq9_total, gad7_total)

    # Subscale means
    va_mean = _subscale_mean(compact10, _VA_IDX)
    ote_mean = _subscale_mean(compact10, _OTE_IDX)
    ba_mean = _subscale_mean(compact10, _BA_IDX)
    va_expected = _VA_EXPECTED.get(distress_band, (2.0, 4.5))
    ote_expected = _OTE_EXPECTED.get(distress_band, (2.5, 5.0))

    lines = [
        "=== SCORES ===",
        f"PHQ-9: {phq9} (total={phq9_total}, severity={_phq9_band(phq9_total)})",
        f"GAD-7: {gad7} (total={gad7_total}, severity={_gad7_band(gad7_total)})",
        f"CompACT-10: {compact10} (total={compact10_total})",
        f"  OtE mean={ote_mean:.2f} (expected {ote_expected[0]:.1f}–{ote_expected[1]:.1f})",
        f"  BA  mean={ba_mean:.2f}",
        f"  VA  mean={va_mean:.2f} (expected {va_expected[0]:.1f}–{va_expected[1]:.1f})",
        f"Combined distress band: {distress_band}",
        "",
        "=== LEVEL B VIOLATIONS ===",
    ]

    if violations:
        for v in violations:
            lines.append(f"[{v.rule} {v.severity.upper()}] {v.message}")
    else:
        lines.append("None")

    if assessments:
        lines.append("")
        lines.append("=== ASSESSOR REASONING SUMMARY ===")
        for instrument, result in assessments.items():
            steps = getattr(result, "steps", {}) if hasattr(result, "steps") else {}
            labels = getattr(result, "labels", []) if hasattr(result, "labels") else []
            mismatches = getattr(result, "label_mismatches", []) if hasattr(result, "label_mismatches") else []
            lines.append(f"{instrument}:")
            if steps:
                # Include category scan summary if available
                scan_key = "step_0_category_scan" if "step_0_category_scan" in steps else "step_0_triflex_scan"
                scan = steps.get(scan_key, {})
                if scan:
                    lines.append(f"  Category scan: {json.dumps(scan, ensure_ascii=False)[:300]}")
            if mismatches:
                lines.append(f"  Label-score mismatches: {mismatches}")
            if not steps and not mismatches:
                lines.append("  (no reasoning captured)")

    lines.append("")
    lines.append(
        "Based on the above, decide if any corrections are warranted. "
        "Apply only corrections you are confident about. When in doubt, do not correct."
    )
    return "\n".join(lines)


def _should_invoke_level_c(
    violations: list[ConstraintViolation],
    compact10: list[int],
    phq9_total: int,
    gad7_total: int,
) -> bool:
    """Determine if Level C agent should be invoked.

    Invocation criteria (spec section 5.3):
    - Level B has at least one High or Medium violation, OR
    - CompACT-10 total exceeds expected range by > 5 points.
    """
    if any(v.severity in ("high", "medium") for v in violations):
        return True

    # Check CompACT-10 total vs expected
    distress_band = _distress_band(phq9_total, gad7_total)
    va_expected_high = _VA_EXPECTED.get(distress_band, (2.0, 4.5))[1]
    ote_expected_high = _OTE_EXPECTED.get(distress_band, (2.5, 5.0))[1]
    ba_expected_high = 4.0  # BA weakest link, rough estimate

    # Expected CompACT-10 total ceiling: (va_high * 4 + ote_high * 3 + ba_high * 3)
    expected_compact_ceiling = va_expected_high * 4 + ote_expected_high * 3 + ba_expected_high * 3
    if sum(compact10) > expected_compact_ceiling + 5:
        return True

    return False


def run_level_c_agent(
    client: "LLMClient",
    phq9: list[int],
    gad7: list[int],
    compact10: list[int],
    violations: list[ConstraintViolation],
    assessments: dict[str, Any] | None = None,
) -> tuple[list[int], list[int], list[int], list[dict]]:
    """Run the Level C LLM calibration agent.

    Only call this when _should_invoke_level_c() returns True.

    Returns:
        (phq9_out, gad7_out, compact10_out, agent_corrections)
        agent_corrections is a list of correction dicts from the agent.
    """
    from ..llm_client import parse_json_response

    phq9_out = list(phq9)
    gad7_out = list(gad7)
    compact10_out = list(compact10)
    agent_corrections: list[dict] = []

    user_input = _build_level_c_input(phq9, gad7, compact10, violations, assessments)

    messages = [
        {"role": "system", "content": _LEVEL_C_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    try:
        raw_response = client.complete(messages, temperature=0.05)
    except Exception as e:
        logger.error("Level C agent LLM call failed: %s", e)
        return phq9_out, gad7_out, compact10_out, []

    parsed = parse_json_response(raw_response)
    if parsed is None:
        logger.warning("Level C agent: could not parse JSON response, skipping corrections")
        logger.debug("Level C raw response: %s", raw_response[:500])
        return phq9_out, gad7_out, compact10_out, []

    # Extract corrected scores
    corrected_phq9 = parsed.get("corrected_phq9")
    corrected_gad7 = parsed.get("corrected_gad7")
    corrected_compact10 = parsed.get("corrected_compact10")
    corrections = parsed.get("corrections", [])
    concordance = parsed.get("concordance_assessment", {})

    logger.info("Level C agent concordance: %s", concordance)

    if corrections:
        logger.info("Level C agent proposed %d correction(s):", len(corrections))
        for c in corrections:
            logger.info(
                "  %s items %s %s by %s: %s",
                c.get("instrument"), c.get("items_1indexed"),
                c.get("direction"), c.get("amount"), c.get("rationale", "")[:100],
            )

    # Apply corrected scores with safety validation
    specs = {"PHQ-9": (9, 3), "GAD-7": (7, 3), "CompACT-10": (10, 6)}

    def _safe_apply(raw: list[int], corrected: list | None, max_change: int, max_val: int) -> list[int]:
        if corrected is None or not isinstance(corrected, list):
            return raw
        if len(corrected) != len(raw):
            logger.warning(
                "Level C agent returned wrong length (%d vs %d), skipping",
                len(corrected), len(raw),
            )
            return raw
        result = []
        for orig, new in zip(raw, corrected):
            # Safety: never change by more than max_change points per item
            new_clipped = max(0, min(max_val, int(new)))
            if abs(new_clipped - orig) > max_change:
                logger.warning(
                    "Level C correction capped: %d → %d (exceeds ±%d limit, using %d)",
                    orig, new_clipped, max_change,
                    orig + (max_change if new_clipped > orig else -max_change),
                )
                new_clipped = orig + (max_change if new_clipped > orig else -max_change)
                new_clipped = max(0, min(max_val, new_clipped))
            result.append(new_clipped)
        return result

    phq9_out = _safe_apply(phq9, corrected_phq9, max_change=2, max_val=3)
    gad7_out = _safe_apply(gad7, corrected_gad7, max_change=2, max_val=3)
    compact10_out = _safe_apply(compact10, corrected_compact10, max_change=2, max_val=6)

    agent_corrections = corrections

    # Log net changes
    if phq9_out != phq9:
        logger.info("Level C PHQ-9: %s → %s (delta=%s)", phq9, phq9_out, [n - o for o, n in zip(phq9, phq9_out)])
    if gad7_out != gad7:
        logger.info("Level C GAD-7: %s → %s (delta=%s)", gad7, gad7_out, [n - o for o, n in zip(gad7, gad7_out)])
    if compact10_out != compact10:
        logger.info(
            "Level C CompACT-10: %s → %s (delta=%s)",
            compact10, compact10_out,
            [n - o for o, n in zip(compact10, compact10_out)],
        )

    return phq9_out, gad7_out, compact10_out, agent_corrections


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_calibration_config(path: str | Path) -> dict:
    """Load calibration configuration from JSON."""
    path = Path(path)
    if not path.exists():
        logger.warning("Calibration config not found at %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
