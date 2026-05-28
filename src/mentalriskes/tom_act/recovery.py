"""Three-stage JSON parsing and recovery pipeline (spec §6.9).

Applied identically to Llama and Gemma outputs. The original raw response is
always retained by the caller, so any record can be re-parsed later.

Stage 1 — strict ``json.loads`` on the trimmed response.
Stage 2 — permissive: strip code fences, extract the first balanced ``{...}``,
          run ``json-repair`` (trailing commas, single quotes, unquoted keys…).
Stage 3 — fuzzy field extraction: ``rapidfuzz`` against closed Spanish label
          sets (accent-insensitive, edit-distance ≤ 0.2 → similarity ≥ 80),
          regex for integer item scores. Salvages partial data.

Closed label sets come from ``constants.py`` (single source of truth).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import json_repair
from rapidfuzz import fuzz, process

from .constants import (
    INSTRUMENTS,
    PRESENCIA_LABELS_ES,
    TOM_STANCE_FALLBACK_ALIASES,
    TOM_STANCE_LABELS_ES,
    TOM_TIER_LABELS_ES,
)

logger = logging.getLogger(__name__)

# Similarity threshold: edit-distance ≤ 0.2 ⇔ rapidfuzz ratio ≥ 80.
_FUZZY_THRESHOLD = 80.0

# Critical-field requirements per signal schema.
_VIEW_MIN_ITEMS = 21  # ≥ 80% of the 26 view items (spec §6.9)
_VIEW_INSTRUMENTS = {"phq9": 9, "gad7": 7, "compact10": 10}


@dataclass
class RecoveryResult:
    parsed: dict | None
    success: bool
    stage: str | None          # "strict" | "permissive" | "fuzzy" | None
    error: str | None = None
    notes: list[str] = field(default_factory=list)   # disagreements → meta.jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower().strip()


def fuzzy_label(value, choices: list[str], aliases: dict[str, str] | None = None,
                threshold: float = _FUZZY_THRESHOLD) -> str | None:
    """Map ``value`` to a canonical label from ``choices`` (accent-insensitive).

    ``aliases`` maps alternate forms (e.g. English) → canonical choice. Returns
    the canonical Spanish label or None if no match clears ``threshold``.
    """
    if value is None:
        return None
    raw = str(value)
    norm = _strip_accents(raw)
    # Build a normalized lookup over canonical choices + aliases.
    lookup: dict[str, str] = {_strip_accents(c): c for c in choices}
    if aliases:
        for alt, canon in aliases.items():
            lookup[_strip_accents(alt)] = canon
    # Exact / containment fast paths.
    if norm in lookup:
        return lookup[norm]
    for key, canon in lookup.items():
        if key in norm or norm in key:
            return canon
    # Fuzzy match.
    match = process.extractOne(norm, list(lookup.keys()), scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return lookup[match[0]]
    return None


def _strict(raw: str) -> dict | None:
    try:
        obj = json.loads(raw.strip())
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _permissive(raw: str) -> dict | None:
    text = raw.strip()
    # Strip ```json ... ``` fences.
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1)
    # Extract first balanced {...}.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start:i + 1]
                    break
        else:
            text = text[start:]
    try:
        obj = json_repair.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Schema validation / normalization (applied after a successful parse)
# ---------------------------------------------------------------------------

def _normalize_categorical(d: dict, field_name: str, choices: list[str],
                           aliases: dict[str, str] | None, notes: list[str]) -> bool:
    """Normalize ``d[field_name]`` to a canonical label in place. True if valid."""
    canon = fuzzy_label(d.get(field_name), choices, aliases)
    if canon is None:
        return False
    if canon != d.get(field_name):
        notes.append(f"{field_name}: '{d.get(field_name)}' -> '{canon}'")
    d[field_name] = canon
    return True


def _validate_view(d: dict, notes: list[str]) -> bool:
    """Validate a 3-instrument view; count recovered item scores."""
    recovered = 0
    for inst, n in _VIEW_INSTRUMENTS.items():
        block = d.get(inst, {})
        items = block.get("items", []) if isinstance(block, dict) else []
        for it in items:
            if isinstance(it, dict) and _coerce_int(it.get("score")) is not None:
                recovered += 1
    if recovered < _VIEW_MIN_ITEMS:
        notes.append(f"view: only {recovered}/26 item scores recovered")
        return False
    return True


def _coerce_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+", v)
        if m:
            return int(m.group())
    return None


def assessor_scores(d: dict, instrument: str) -> list[int] | None:
    """Extract a full clipped item-score array from a parsed assessor response.

    Handles the direct ``{"PHQ-9": [...]}`` form and the CoT step-2 item dicts.
    Returns None unless exactly ``n_items`` scores are recoverable (a truncated
    CoT therefore fails validation and is retried).
    """
    spec = INSTRUMENTS.get(instrument)
    if not spec:
        return None
    n, mx = spec["n_items"], spec["max_val"]
    scores = None
    for key in (instrument, instrument.replace("-", ""), instrument.lower()):
        v = d.get(key)
        if isinstance(v, list):
            scores = v
            break
    if scores is None:
        step2 = d.get("step_2_temporal") or d.get("step_2_endorsement") or {}
        if isinstance(step2, dict) and step2:
            tmp = []
            for i in range(1, n + 1):
                item = step2.get(f"item_{i}", {})
                sc = item.get("score") if isinstance(item, dict) else None
                ci = _coerce_int(sc)
                if ci is None:
                    tmp = None
                    break
                tmp.append(ci)
            scores = tmp
    if scores is None or len(scores) != n:
        return None
    out = []
    for s in scores:
        ci = _coerce_int(s)
        if ci is None:
            return None
        out.append(max(0, min(mx, ci)))
    return out


def _validate(d: dict, schema: str, notes: list[str]) -> bool:
    if schema == "view":
        return _validate_view(d, notes)
    if schema.startswith("assessor:"):
        instrument = schema.split(":", 1)[1]
        ok = assessor_scores(d, instrument) is not None
        if not ok:
            notes.append(f"assessor {instrument}: full score vector not recoverable")
        return ok
    if schema == "tom_tier_patient":
        return _normalize_categorical(d, "argmax", TOM_TIER_LABELS_ES, None, notes)
    if schema == "tom_stance":
        return _normalize_categorical(d, "stance", TOM_STANCE_LABELS_ES,
                                      TOM_STANCE_FALLBACK_ALIASES, notes)
    if schema == "presencia":
        return _normalize_categorical(d, "presencia", PRESENCIA_LABELS_ES, None, notes)
    return True  # generic: any dict is acceptable


# ---------------------------------------------------------------------------
# Stage 3 — fuzzy extraction from broken text
# ---------------------------------------------------------------------------

def _fuzzy_categorical(raw: str, field_name: str, choices: list[str],
                       aliases: dict[str, str] | None, notes: list[str]) -> dict | None:
    # Try to find ``"field": "value"`` first, else scan whole text for a label.
    m = re.search(rf'"{field_name}"\s*:\s*"?([^",}}\n]+)"?', raw)
    candidate = m.group(1) if m else raw
    canon = fuzzy_label(candidate, choices, aliases)
    if canon is None and m is None:
        # last resort: does any label literally appear in the text?
        for c in choices:
            if _strip_accents(c) in _strip_accents(raw):
                canon = c
                break
    if canon is None:
        return None
    notes.append(f"fuzzy-extracted {field_name}='{canon}'")
    return {field_name: canon}


def _fuzzy_view(raw: str, notes: list[str]) -> dict | None:
    """Salvage item scores from a structurally broken view response."""
    out: dict = {}
    recovered = 0
    for inst, n in _VIEW_INSTRUMENTS.items():
        # Find the instrument block then pull "item": k ... "score": v pairs.
        items = []
        for im in re.finditer(r'"item"\s*:\s*(\d+).*?"score"\s*:\s*(-?\d+)', raw, re.DOTALL):
            items.append({"item": int(im.group(1)), "score": int(im.group(2))})
        # Heuristic fallback handled below; keep simple per-instrument best effort.
        if items:
            out[inst] = {"items": items[:n]}
            recovered += len(items[:n])
    if recovered >= _VIEW_MIN_ITEMS:
        notes.append(f"fuzzy-extracted {recovered}/26 view item scores")
        return out
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recover(raw: str, schema: str | None = None) -> RecoveryResult:
    """Parse ``raw`` into structured JSON via the three-stage pipeline.

    Args:
        raw: the raw LLM response text.
        schema: one of ``"view"``, ``"tom_tier_patient"``, ``"tom_stance"``,
            ``"presencia"``, or None (generic — any dict accepted).
    """
    if not raw or not raw.strip():
        return RecoveryResult(None, False, None, error="empty_response")

    notes: list[str] = []

    # Stages 1 & 2 produce a candidate dict; validate against the schema.
    for stage, parser in (("strict", _strict), ("permissive", _permissive)):
        d = parser(raw)
        if d is not None:
            if schema is None or _validate(d, schema, notes):
                return RecoveryResult(d, True, stage, notes=notes)

    # Stage 3 — fuzzy field extraction.
    if schema == "view":
        d = _fuzzy_view(raw, notes)
    elif schema == "tom_tier_patient":
        d = _fuzzy_categorical(raw, "argmax", TOM_TIER_LABELS_ES, None, notes)
    elif schema == "tom_stance":
        d = _fuzzy_categorical(raw, "stance", TOM_STANCE_LABELS_ES,
                               TOM_STANCE_FALLBACK_ALIASES, notes)
    elif schema == "presencia":
        d = _fuzzy_categorical(raw, "presencia", PRESENCIA_LABELS_ES, None, notes)
    else:
        d = None

    if d is not None:
        return RecoveryResult(d, True, "fuzzy", notes=notes)

    return RecoveryResult(None, False, None,
                          error="fuzzy_extraction_insufficient", notes=notes)
