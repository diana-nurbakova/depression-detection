"""Parse structured 7-field LLM output into LLMOutput dataclass.

Expected format from the LLM:
    SYMPTOM_MATCH: YES|PARTIAL|NO
    SELF_REFERENCE: DIRECT|INDIRECT|NONE
    DETAIL_LEVEL: HIGH|MEDIUM|LOW|NONE
    CONFOUNDERS: <text> or NONE
    SCORE: 0|1|2|3
    CONFIDENCE: 1|2|3|4|5
    REASONING: <text>

The parser is robust to:
- Extra text before/after the template
- Missing fields (defaults applied)
- Non-standard values (mapped to closest valid value)
- Fields in wrong order
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from hipert.models import LLMOutput

logger = logging.getLogger(__name__)

# Valid values for each field
_VALID_SYMPTOM_MATCH = {"YES", "PARTIAL", "NO"}
_VALID_SELF_REFERENCE = {"DIRECT", "INDIRECT", "NONE"}
_VALID_DETAIL_LEVEL = {"HIGH", "MEDIUM", "LOW", "NONE"}
_VALID_SCORES = {0, 1, 2, 3}
_VALID_CONFIDENCE = {1, 2, 3, 4, 5}

# Regex patterns for each field (case-insensitive, flexible whitespace)
_FIELD_PATTERNS = {
    "symptom_match": re.compile(
        r"SYMPTOM_MATCH\s*:\s*(.+)", re.IGNORECASE,
    ),
    "self_reference": re.compile(
        r"SELF_REFERENCE\s*:\s*(.+)", re.IGNORECASE,
    ),
    "detail_level": re.compile(
        r"DETAIL_LEVEL\s*:\s*(.+)", re.IGNORECASE,
    ),
    "confounders": re.compile(
        r"CONFOUNDERS?\s*:\s*(.+)", re.IGNORECASE,
    ),
    "score": re.compile(
        r"SCORE\s*:\s*(.+)", re.IGNORECASE,
    ),
    "confidence": re.compile(
        r"CONFIDENCE\s*:\s*(.+)", re.IGNORECASE,
    ),
    "reasoning": re.compile(
        r"REASONING\s*:\s*(.+)", re.IGNORECASE | re.DOTALL,
    ),
}


def _extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field value from the LLM output text."""
    pattern = _FIELD_PATTERNS[field_name]

    # For reasoning, we want everything after "REASONING:" to the end,
    # but stop at the next field if present.
    if field_name == "reasoning":
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            # Truncate at the next field marker if present
            for other_field in _FIELD_PATTERNS:
                if other_field == "reasoning":
                    continue
                cutoff = re.search(
                    rf"(?:^|\n)\s*{other_field.upper()}\s*:",
                    value,
                    re.IGNORECASE,
                )
                if cutoff:
                    value = value[:cutoff.start()].strip()
            return value
        return None

    # For other fields, find the line containing the field
    for line in text.split("\n"):
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    return None


def _normalize_enum(value: str, valid_set: set[str]) -> str:
    """Normalize a value to the closest valid enum member."""
    upper = value.upper().strip()
    if upper in valid_set:
        return upper

    # Try partial matching
    for valid in valid_set:
        if valid in upper or upper in valid:
            return valid

    # Common misspellings / variations
    mapping = {
        "PARTIALLY": "PARTIAL",
        "MAYBE": "PARTIAL",
        "N/A": "NONE",
        "NA": "NONE",
        "ABSENT": "NONE",
    }
    if upper in mapping and mapping[upper] in valid_set:
        return mapping[upper]

    return list(valid_set)[0]  # fallback to first valid value


def _parse_int(value: str, valid_range: set[int], default: int) -> int:
    """Parse an integer value, clamping to valid range."""
    # Extract first integer from the string
    match = re.search(r"(\d+)", value)
    if match:
        num = int(match.group(1))
        if num in valid_range:
            return num
        # Clamp to nearest valid value
        return min(valid_range, key=lambda x: abs(x - num))
    return default


def parse_llm_response(raw_text: str) -> tuple[LLMOutput, list[str]]:
    """Parse structured 7-field response from LLM output.

    Args:
        raw_text: Raw text output from the LLM.

    Returns:
        Tuple of (LLMOutput, list of warning messages).
        Warnings are generated for missing or invalid fields.
    """
    warnings: list[str] = []

    # Extract each field
    symptom_match_raw = _extract_field(raw_text, "symptom_match")
    self_reference_raw = _extract_field(raw_text, "self_reference")
    detail_level_raw = _extract_field(raw_text, "detail_level")
    confounders_raw = _extract_field(raw_text, "confounders")
    score_raw = _extract_field(raw_text, "score")
    confidence_raw = _extract_field(raw_text, "confidence")
    reasoning_raw = _extract_field(raw_text, "reasoning")

    # Normalize symptom_match
    if symptom_match_raw is None:
        warnings.append("SYMPTOM_MATCH field missing, defaulting to NO")
        symptom_match = "NO"
    else:
        symptom_match = _normalize_enum(symptom_match_raw, _VALID_SYMPTOM_MATCH)
        if symptom_match_raw.upper() != symptom_match:
            warnings.append(
                f"SYMPTOM_MATCH normalized: '{symptom_match_raw}' -> '{symptom_match}'",
            )

    # Normalize self_reference
    if self_reference_raw is None:
        warnings.append("SELF_REFERENCE field missing, defaulting to NONE")
        self_reference = "NONE"
    else:
        self_reference = _normalize_enum(
            self_reference_raw, _VALID_SELF_REFERENCE,
        )

    # Normalize detail_level
    if detail_level_raw is None:
        warnings.append("DETAIL_LEVEL field missing, defaulting to NONE")
        detail_level = "NONE"
    else:
        detail_level = _normalize_enum(detail_level_raw, _VALID_DETAIL_LEVEL)

    # Confounders (free text field)
    if confounders_raw is None:
        warnings.append("CONFOUNDERS field missing, defaulting to NONE")
        confounders = "NONE"
    else:
        confounders = confounders_raw.strip()
        if not confounders:
            confounders = "NONE"

    # Score
    if score_raw is None:
        warnings.append("SCORE field missing, defaulting to 0")
        score = 0
    else:
        score = _parse_int(score_raw, _VALID_SCORES, 0)

    # Confidence
    if confidence_raw is None:
        warnings.append("CONFIDENCE field missing, defaulting to 1")
        confidence = 1
    else:
        confidence = _parse_int(confidence_raw, _VALID_CONFIDENCE, 1)

    # Reasoning
    if reasoning_raw is None:
        warnings.append("REASONING field missing")
        reasoning = ""
    else:
        # Clean up reasoning text
        reasoning = reasoning_raw.strip()
        # Remove trailing field-like patterns that might have leaked
        reasoning = re.sub(
            r"\n\s*(SYMPTOM_MATCH|SELF_REFERENCE|DETAIL_LEVEL|"
            r"CONFOUNDERS?|SCORE|CONFIDENCE)\s*:.*$",
            "",
            reasoning,
            flags=re.IGNORECASE | re.DOTALL,
        )
        reasoning = reasoning.strip()

    if warnings:
        logger.debug("Parse warnings for response: %s", "; ".join(warnings))

    output = LLMOutput(
        symptom_match=symptom_match,
        self_reference=self_reference,
        detail_level=detail_level,
        confounders=confounders,
        score=score,
        confidence=confidence,
        reasoning=reasoning,
        raw_text=raw_text,
    )

    return output, warnings
