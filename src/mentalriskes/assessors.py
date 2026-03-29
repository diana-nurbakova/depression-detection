"""LLM-based assessors for PHQ-9, GAD-7, and CompACT-10.

Each assessor uses chain-of-thought prompting with three steps:
  Step 0: Category-level evidence scan
  Step 1: Per-item detection with disambiguation
  Step 2: Temporal/severity inference with behavioral anchors

Prompts are imported from the spec's assessor_prompts_v2 module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import LLMClient, parse_json_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "PHQ-9": {"n_items": 9, "max_val": 3, "key": "PHQ-9"},
    "GAD-7": {"n_items": 7, "max_val": 3, "key": "GAD-7"},
    "CompACT-10": {"n_items": 10, "max_val": 6, "key": "CompACT-10"},
}


@dataclass
class AssessmentResult:
    """Result of a single instrument assessment."""
    instrument: str
    scores: list[int]
    raw_response: str = ""
    steps: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def total(self) -> int:
        return sum(self.scores)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

# Import prompts from the spec file (assessor_prompts_v2.py)
# These are loaded at module level for efficiency.
_PROMPTS_LOADED = False
_PHQ9_SYSTEM = ""
_GAD7_SYSTEM = ""
_COMPACT10_SYSTEM = ""
_PHQ9_FEW_SHOT = ""
_GAD7_FEW_SHOT = ""
_COMPACT10_FEW_SHOT = ""


def _load_prompts() -> None:
    """Load prompts from the assessor_prompts_v2 spec module."""
    global _PROMPTS_LOADED, _PHQ9_SYSTEM, _GAD7_SYSTEM, _COMPACT10_SYSTEM
    global _PHQ9_FEW_SHOT, _GAD7_FEW_SHOT, _COMPACT10_FEW_SHOT

    if _PROMPTS_LOADED:
        return

    # Try importing from the spec location (added to sys.path)
    import importlib.util
    spec_path = Path("specs/MentalRiskES/assessor_prompts_v2.py")
    if spec_path.exists():
        spec = importlib.util.spec_from_file_location("assessor_prompts_v2", spec_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore

        _PHQ9_SYSTEM = mod.PHQ9_SYSTEM_PROMPT
        _GAD7_SYSTEM = mod.GAD7_SYSTEM_PROMPT
        _COMPACT10_SYSTEM = mod.COMPACT10_SYSTEM_PROMPT
        _PHQ9_FEW_SHOT = mod.PHQ9_FEW_SHOT
        _GAD7_FEW_SHOT = mod.GAD7_FEW_SHOT
        _COMPACT10_FEW_SHOT = mod.COMPACT10_FEW_SHOT
        logger.info("Loaded assessor prompts from %s", spec_path)
    else:
        logger.warning("Assessor prompts file not found at %s, using minimal prompts", spec_path)
        _PHQ9_SYSTEM = _MINIMAL_PHQ9
        _GAD7_SYSTEM = _MINIMAL_GAD7
        _COMPACT10_SYSTEM = _MINIMAL_COMPACT10

    _PROMPTS_LOADED = True


# Minimal fallback prompts (if spec file not found)
_MINIMAL_PHQ9 = """You are a clinical psychologist assessing depression. Read the therapeutic conversation in Spanish and estimate PHQ-9 scores.

PHQ-9 items (0=not at all, 1=several days, 2=more than half the days, 3=nearly every day):
1. Little interest or pleasure
2. Feeling down, depressed, hopeless
3. Sleep problems
4. Feeling tired
5. Poor appetite or overeating
6. Feeling bad about yourself
7. Trouble concentrating
8. Psychomotor changes
9. Thoughts of self-harm

{few_shot_examples}

CONVERSATION:
{conversation_history}

Respond with ONLY a JSON object:
{{"PHQ-9": [item1, item2, item3, item4, item5, item6, item7, item8, item9]}}"""

_MINIMAL_GAD7 = """You are a clinical psychologist assessing anxiety. Read the therapeutic conversation in Spanish and estimate GAD-7 scores.

GAD-7 items (0=never, 1=several days, 2=more than half, 3=nearly every day):
1. Feeling nervous
2. Can't stop worrying
3. Worrying too much
4. Difficulty relaxing
5. Restlessness
6. Irritability
7. Feeling afraid

{few_shot_examples}

CONVERSATION:
{conversation_history}

Respond with ONLY a JSON object:
{{"GAD-7": [item1, item2, item3, item4, item5, item6, item7]}}"""

_MINIMAL_COMPACT10 = """You are a psychologist trained in ACT assessing psychological flexibility. Read the conversation in Spanish and estimate CompACT-10 scores.

CompACT-10 items (0=strongly disagree ... 6=strongly agree):
1. I rush through meaningful activities [Behavioral Awareness - reverse]
2. I act coherently with how I want to live [Valued Action]
3. I tell myself I shouldn't have certain thoughts [Openness - reverse]
4. I behave according to my personal values [Valued Action]
5. I work hard to avoid difficult situations [Openness - reverse]
6. Even doing important things, I'm not paying attention [Behavioral Awareness - reverse]
7. I pursue meaningful things even when difficult [Valued Action]
8. I work hard to keep upsetting feelings away [Openness - reverse]
9. I'm on autopilot [Behavioral Awareness - reverse]
10. I can keep going when it matters [Valued Action]

{few_shot_examples}

CONVERSATION:
{conversation_history}

Respond with ONLY a JSON object:
{{"CompACT-10": [item1, item2, item3, item4, item5, item6, item7, item8, item9, item10]}}"""


def build_prompt(
    instrument: str,
    conversation_history: str,
    use_few_shot: bool = True,
) -> str:
    """Build the full assessor prompt for a given instrument."""
    _load_prompts()

    templates = {"PHQ-9": _PHQ9_SYSTEM, "GAD-7": _GAD7_SYSTEM, "CompACT-10": _COMPACT10_SYSTEM}
    few_shots = {"PHQ-9": _PHQ9_FEW_SHOT, "GAD-7": _GAD7_FEW_SHOT, "CompACT-10": _COMPACT10_FEW_SHOT}

    template = templates.get(instrument)
    if template is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    few_shot_text = few_shots.get(instrument, "") if use_few_shot else ""

    return template.format(
        conversation_history=conversation_history,
        few_shot_examples=few_shot_text,
    )


# ---------------------------------------------------------------------------
# Assessment execution
# ---------------------------------------------------------------------------

def assess_instrument(
    client: LLMClient,
    instrument: str,
    conversation_history: str,
    use_few_shot: bool = True,
) -> AssessmentResult:
    """
    Run a single instrument assessment via LLM.

    Args:
        client: LLM client instance.
        instrument: "PHQ-9", "GAD-7", or "CompACT-10".
        conversation_history: Formatted conversation string.
        use_few_shot: Whether to include few-shot examples.

    Returns:
        AssessmentResult with scores and reasoning steps.
    """
    spec = INSTRUMENTS[instrument]
    prompt = build_prompt(instrument, conversation_history, use_few_shot)

    messages = [{"role": "user", "content": prompt}]

    try:
        raw_response = client.complete(messages)
    except Exception as e:
        logger.error("LLM call failed for %s: %s", instrument, e)
        # Graceful degradation: return defaults
        defaults = _default_scores(instrument)
        return AssessmentResult(
            instrument=instrument, scores=defaults,
            error=str(e), raw_response="",
        )

    # Parse response
    parsed = parse_json_response(raw_response)
    if parsed is None:
        logger.warning("Failed to parse JSON for %s, using defaults", instrument)
        return AssessmentResult(
            instrument=instrument,
            scores=_default_scores(instrument),
            raw_response=raw_response,
            error="No JSON found in response",
        )

    scores = parsed.get(instrument)
    if scores is None:
        # Try alternative keys
        for key in [instrument, instrument.replace("-", ""), instrument.lower()]:
            scores = parsed.get(key)
            if scores is not None:
                break

    if scores is None:
        logger.warning("No scores found for %s in response, using defaults", instrument)
        return AssessmentResult(
            instrument=instrument,
            scores=_default_scores(instrument),
            raw_response=raw_response,
            steps=parsed,
            error="Scores key not found",
        )

    # Validate and clip
    if len(scores) != spec["n_items"]:
        logger.warning("%s: expected %d scores, got %d", instrument, spec["n_items"], len(scores))
        # Pad or truncate
        if len(scores) < spec["n_items"]:
            mid = spec["max_val"] // 2
            scores.extend([mid] * (spec["n_items"] - len(scores)))
        else:
            scores = scores[:spec["n_items"]]

    scores = [max(0, min(spec["max_val"], int(s))) for s in scores]

    # Extract reasoning steps
    steps = {}
    for step_key in ["step_0_category_scan", "step_0_triflex_scan",
                     "step_1_detection", "step_2_temporal", "step_2_endorsement"]:
        if step_key in parsed:
            steps[step_key] = parsed[step_key]

    return AssessmentResult(
        instrument=instrument,
        scores=scores,
        raw_response=raw_response,
        steps=steps,
    )


def assess_all_instruments(
    client: LLMClient,
    conversation_history: str,
    use_few_shot: bool = True,
) -> dict[str, AssessmentResult]:
    """
    Run assessments for all three instruments.

    Returns dict mapping instrument name -> AssessmentResult.
    """
    results = {}
    for instrument in ["PHQ-9", "GAD-7", "CompACT-10"]:
        logger.info("Assessing %s...", instrument)
        results[instrument] = assess_instrument(
            client, instrument, conversation_history, use_few_shot
        )
        logger.info(
            "%s scores: %s (total=%d)",
            instrument, results[instrument].scores, results[instrument].total,
        )

    return results


def _default_scores(instrument: str) -> list[int]:
    """Return default scores for graceful degradation."""
    spec = INSTRUMENTS[instrument]
    if instrument == "PHQ-9":
        return [1, 1, 1, 1, 1, 1, 1, 1, 0]  # default 1 except suicidality
    elif instrument == "GAD-7":
        return [1, 1, 1, 1, 1, 1, 1]
    elif instrument == "CompACT-10":
        return [3] * 10  # midpoint
    return [spec["max_val"] // 2] * spec["n_items"]
