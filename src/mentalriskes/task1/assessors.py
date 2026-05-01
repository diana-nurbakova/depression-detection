"""LLM-based assessors for PHQ-9, GAD-7, and CompACT-10.

Each assessor uses chain-of-thought prompting with three steps:
  Step 0: Category-level evidence scan
  Step 1: Per-item detection with disambiguation
  Step 2: Temporal/severity inference with behavioral anchors

Prompts are imported from the spec's assessor_prompts_v2 module.
Verbalizer system (v2.1) replaces numeric tags with clinically meaningful
Spanish labels and provides label-score consistency checking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient, parse_json_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verbalizer system (loaded from verbalizer_update_v2_1.py)
# ---------------------------------------------------------------------------

_VERBALIZERS_LOADED = False
_PHQ9_VERBALIZERS: dict = {}
_GAD7_VERBALIZERS: dict = {}
_COMPACT10_VERBALIZERS: dict = {}
_EVIDENCE_LEVEL_VERBALIZERS: dict = {}
_DETECTION_VERBALIZERS: dict = {}
_VERBALIZER_CONSISTENCY_FN = None
_VERBALIZER_RESOLVE_FN = None


def _load_verbalizers() -> None:
    """Load verbalizer definitions from the spec module."""
    global _VERBALIZERS_LOADED
    global _PHQ9_VERBALIZERS, _GAD7_VERBALIZERS, _COMPACT10_VERBALIZERS
    global _EVIDENCE_LEVEL_VERBALIZERS, _DETECTION_VERBALIZERS
    global _VERBALIZER_CONSISTENCY_FN, _VERBALIZER_RESOLVE_FN

    if _VERBALIZERS_LOADED:
        return

    import importlib.util
    spec_path = Path("specs/MentalRiskES/verbalizer_update_v2_1.py")
    if spec_path.exists():
        spec = importlib.util.spec_from_file_location("verbalizer_update_v2_1", spec_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore

        _PHQ9_VERBALIZERS = mod.PHQ9_VERBALIZERS
        _GAD7_VERBALIZERS = mod.GAD7_VERBALIZERS
        _COMPACT10_VERBALIZERS = mod.COMPACT10_VERBALIZERS
        _EVIDENCE_LEVEL_VERBALIZERS = mod.EVIDENCE_LEVEL_VERBALIZERS
        _DETECTION_VERBALIZERS = mod.DETECTION_VERBALIZERS
        _VERBALIZER_CONSISTENCY_FN = mod.check_label_score_consistency
        _VERBALIZER_RESOLVE_FN = mod.resolve_mismatches
        logger.info("Loaded verbalizers from %s", spec_path)
    else:
        logger.warning("Verbalizer file not found at %s — verbalizer features disabled", spec_path)

    _VERBALIZERS_LOADED = True

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
    labels: list[str] = field(default_factory=list)  # verbalizer labels from LLM
    label_mismatches: list[dict] = field(default_factory=list)  # label-score disagreements

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


def _build_verbalizer_instructions(instrument: str) -> str:
    """
    Build verbalizer instructions to append to the assessor prompt.

    These instruct the model to use clinically meaningful labels instead of
    raw integers, enabling label-score consistency checking.
    """
    _load_verbalizers()

    if not _PHQ9_VERBALIZERS:
        return ""  # verbalizers not available

    lines = ["\n## MEANINGFUL LABELS (use these instead of raw numbers)\n"]

    if instrument == "PHQ-9":
        verbs = _PHQ9_VERBALIZERS
        lines.append("For each item in Step 2, select a frequency_label from:")
        for score, label in verbs["labels"].items():
            lines.append(f"  {score} = \"{label}\"")
        lines.append("\nAlso select an item_specific_descriptor from the item's set.")
        lines.append("The score MUST match the frequency_label.")
        lines.append("\nFor Step 1, use detection_label from these process-specific options:")
        det = _DETECTION_VERBALIZERS.get("phq9", {})
        for item_key in ["item_1", "item_2", "item_3", "item_4", "item_5",
                         "item_6", "item_7", "item_8", "item_9"]:
            item_det = det.get(item_key, {})
            present = item_det.get("present_labels", [])
            insuf = item_det.get("insufficient_label", "")
            if present:
                lines.append(f"  {item_key}: {' | '.join(present)} | {insuf}")

    elif instrument == "GAD-7":
        verbs = _GAD7_VERBALIZERS
        lines.append("For each item in Step 2, select a frequency_label from:")
        for score, label in verbs["labels"].items():
            lines.append(f"  {score} = \"{label}\"")
        lines.append("\nThe score MUST match the frequency_label.")

    elif instrument == "CompACT-10":
        verbs = _COMPACT10_VERBALIZERS
        lines.append("For each item in Step 2, select an agreement_label from:")
        for score, label in verbs["labels"].items():
            lines.append(f"  {score} = \"{label}\"")
        lines.append("\nAlso select a construct_descriptor from the item's construct-specific set.")
        lines.append("The score MUST match the agreement_label.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Level A prompt anchors (spec section 3.1)
# ---------------------------------------------------------------------------

# Per-instrument anchor texts injected when prompt_anchors=True (Level A).
# Source: mentalriskes2026_constraints_ablation_spec.md section 3.1.
_LEVEL_A_ANCHORS: dict[str, list[str]] = {
    "PHQ-9": [
        "CROSS-INSTRUMENT ANCHOR: PHQ-9 and GAD-7 totals are typically within 4 points of "
        "each other (Spearman rho=0.74 in clinical populations). If your PHQ-9 total would "
        "be much higher or lower than you would expect for GAD-7, double-check that the "
        "evidence clearly supports the discrepancy.",
    ],
    "GAD-7": [
        "ITEM DISAMBIGUATION ANCHOR — Item 2 vs Item 3: Item 2 asks about the LOOP "
        "(can the patient stop worrying once it starts?). Item 3 asks about the BREADTH "
        "(does the patient worry about many different things?). These are different constructs. "
        "A patient who ruminates intensely about ONE topic scores high on Item 2 but low on Item 3.",
        "CROSS-INSTRUMENT ANCHOR: PHQ-9 and GAD-7 totals are typically within 4 points of "
        "each other (Spearman rho=0.74 in clinical populations). If your GAD-7 total would "
        "be much higher or lower than the PHQ-9 would imply, double-check the evidence.",
    ],
    "CompACT-10": [
        "VALUED ACTION ANCHOR: For a moderately distressed patient (PHQ-9 10-14), typical "
        "Valued Action per-item scores are 3-4. Score 5+ ONLY with strong behavioral evidence "
        "of values-aligned action OUTSIDE the therapy session — not just within-session "
        "willingness ('lo intentaré') or brief compliance. Within-session engagement does not "
        "constitute established values-aligned behavior.",
        "OPENNESS TO EXPERIENCE ANCHOR: If the therapist teaches acceptance, defusion, or "
        "mindfulness techniques, this implies the patient currently struggles with avoidance or "
        "fusion. Score OtE items (3, 5, 8) at 3-4 (moderate avoidance), NOT 0-1. Low OtE "
        "scores (0-1) would indicate the patient has essentially no avoidance patterns, which "
        "is inconsistent with seeking ACT-based therapy.",
    ],
}


def _build_anchor_block(instrument: str) -> str:
    """Build the Level A anchor text block for injection into a prompt."""
    anchors = _LEVEL_A_ANCHORS.get(instrument, [])
    if not anchors:
        return ""
    lines = ["\n## PSYCHOMETRIC CALIBRATION ANCHORS (read carefully before scoring)\n"]
    for i, anchor in enumerate(anchors, 1):
        lines.append(f"{i}. {anchor}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Recency bias anchor + GAD-7 severity anchors (spec: gad7_severe_examples.py)
# ---------------------------------------------------------------------------

_RECENCY_BIAS_ANCHOR = """
## CRITICAL: RECENCY BIAS WARNING

Therapeutic conversations often show improvement within a session: the
patient starts distressed, engages with techniques, and ends calmer.
This within-session arc does NOT change the past-two-weeks assessment.

When scoring, give EQUAL WEIGHT to evidence from ALL rounds, not just
the most recent ones. A patient who says "estoy agobiada todo el tiempo"
in round 1 but "me siento mas tranquila" in round 15 still has the same
past-two-weeks anxiety pattern. The therapy session is a snapshot, not a cure.

Specifically:
- Symptom evidence from early rounds (R1-R5) reflects the patient's
  PRESENTING STATE and general patterns. Weight this HEAVILY.
- Within-session improvement (R8-R15) reflects EMERGING skills, not
  established change. Do NOT reduce scores based on within-session calm.
- Temporal markers like "todo el tiempo", "siempre", "llevo semanas"
  describe the PAST TWO WEEKS, regardless of when in the session they appear.
"""

_GAD7_SEVERITY_ANCHOR = """
## SEVERITY CALIBRATION

If the patient describes anxiety as their PRIMARY presenting problem
(first topic raised, most discussed, strongest emotional language),
expect GAD-7 total of 10-21 (moderate to severe).

Score distributions by severity:

  SEVERE anxiety (GAD-7 15-21):
    - Most items at 2-3 (more than half the days to nearly every day)
    - Rarely any item at 0
    - Multiple worry domains (health, relationships, future, performance)
    - Physical symptoms described as constant or near-constant
    - Patient uses words like "todo el tiempo", "siempre", "no puedo parar"

  MODERATE anxiety (GAD-7 8-14):
    - Most items at 1-2 (several days to more than half)
    - Some items may be 0 (not all anxiety symptoms present)
    - Worry focused on 1-2 specific domains
    - Physical symptoms intermittent

  MILD anxiety (GAD-7 5-7):
    - Most items at 0-1
    - Anxiety is present but manageable

Do NOT compress all items into the 0-1 range for a clearly anxious patient.
If anxiety is the PRIMARY reason for therapy, total should be >= 10.
"""

_GAD7_SEVERE_EXAMPLES = """
### SEVERE ANXIETY EXAMPLE — Health Anxiety (GAD-7 gold = 20)

Patient profile: Health anxiety, somatic symptoms, catastrophizing, need for control/certainty.

Key patient quotes across the conversation:

R1: "Todo el tiempo estoy preocupada por mi salud. Me duele algo y ya pienso que es algo grave. Me siento agobiada todo el tiempo."

R2: "Tengo miedo de que me pase algo malo y no me de cuenta a tiempo. Mi abuela murio de cancer y siempre pienso que podria ser como ella. Leo cosas en internet y me entra el panico. Me siento como si tuviera que estar siempre vigilante."

R3: "Es como si me estuviera cayendo por un precipicio, sin nada a que agarrarme. Me siento muy insegura, como si no tuviera control sobre nada. Me da miedo admitir que no se lo que va a pasar. Me aterra."

Assessment at final round:
  Step 0 — Category scan:
    Somatic anxiety: STRONG — multiple physical symptoms, constant hypervigilance.
    Cognitive anxiety: STRONG — catastrophic thinking, uncontrollable worry loop.
    Emotional reactivity: MODERATE — pervasive fear/dread.

  Step 2 — Temporal:
    Item 1: "todo el tiempo", "siempre" — constant. Score: 3
    Item 2: "no puedo evitarlo" — daily uncontrollable worry. Score: 3
    Item 3: 4+ worry domains — pervasive. Score: 3
    Item 4: "siempre en guardia" — cannot relax. Score: 3
    Item 5: physical symptoms persistent. Score: 3
    Item 6: frustration with dismissiveness — intermittent. Score: 2
    Item 7: "me aterra", "precipicio" — constant dread. Score: 3

  GAD-7: [3, 3, 3, 3, 3, 2, 3] = 20 (severe)

NOTE: Although this patient shows IMPROVEMENT within the therapy session,
the GAD-7 asks about the LAST TWO WEEKS. Within-session improvement does
NOT change the past-two-weeks score.

---

### SEVERE ANXIETY EXAMPLE — Social Anxiety (GAD-7 gold = 20)

Patient profile: Social anxiety, fear of judgment/rejection, hypervigilance to evaluation.

Key patient quotes:

R1: "Me da mucho miedo que la gente me juzgue o me rechace. Me siento como si estuviera bajo un microscopio y todos me estuvieran observando."

R2: "Siempre estoy esperando a que alguien me critique. Pienso mucho en que diran los demas."

Assessment:
  Item 1: "muy ansiosa", "bajo un microscopio" — constant. Score: 3
  Item 2: "me paraliza" — worry takes over. Score: 3
  Item 3: judgment, rejection, self-worth, performance — 4+ domains. Score: 3
  Item 4: "me cuesta mucho relajarme" — cannot relax. Score: 3
  Item 5: "batalla constante", physical agitation. Score: 3
  Item 6: self-directed anger. Score: 3
  Item 7: anticipatory fear. Score: 2

  GAD-7: [3, 3, 3, 3, 3, 3, 2] = 20 (severe)

---

### MODERATE ANXIETY CONTRAST — Academic Anxiety (GAD-7 gold = 10)

Patient profile: Academic pressure, perfectionism.

R1: "Me siento muy estresada con mis estudios. Mi familia siempre me presiona."

Assessment:
  Item 1: "muy nerviosa" — but situational. Score: 2
  Item 2: "no puedo pararla" — rumination about grades. Score: 2
  Item 3: grades, family — 2 domains only. Score: 1
  Item 4: physical tension present but not constant. Score: 1
  Item 5: intermittent. Score: 1
  Item 6: minimal evidence. Score: 1
  Item 7: fear of failure — moderate. Score: 2

  GAD-7: [2, 2, 1, 1, 1, 1, 2] = 10 (moderate)

NOTE: Severe anxiety = PERVASIVE symptoms across MULTIPLE domains, CONSTANT
("todo el tiempo", "siempre"). Moderate = anxiety about SPECIFIC situations.
"""


def build_prompt(
    instrument: str,
    conversation_history: str,
    use_few_shot: bool = True,
    use_verbalizers: bool = True,
    use_prompt_anchors: bool = False,
) -> str:
    """Build the full assessor prompt for a given instrument."""
    _load_prompts()

    templates = {"PHQ-9": _PHQ9_SYSTEM, "GAD-7": _GAD7_SYSTEM, "CompACT-10": _COMPACT10_SYSTEM}
    few_shots = {"PHQ-9": _PHQ9_FEW_SHOT, "GAD-7": _GAD7_FEW_SHOT, "CompACT-10": _COMPACT10_FEW_SHOT}

    template = templates.get(instrument)
    if template is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    few_shot_text = few_shots.get(instrument, "") if use_few_shot else ""

    # Inject verbalizer instructions before the output format section
    verbalizer_text = _build_verbalizer_instructions(instrument) if use_verbalizers else ""
    if verbalizer_text:
        few_shot_text = verbalizer_text + "\n" + few_shot_text

    # Level A prompt anchors: injected after few-shot examples, before conversation
    if use_prompt_anchors:
        anchor_block = _build_anchor_block(instrument)
        few_shot_text = few_shot_text + anchor_block

        # Recency bias anchor for ALL instruments (defense-in-depth)
        few_shot_text = few_shot_text + _RECENCY_BIAS_ANCHOR

        # GAD-7-specific: severity anchor + severe anxiety examples
        if instrument == "GAD-7":
            few_shot_text = few_shot_text + _GAD7_SEVERITY_ANCHOR + _GAD7_SEVERE_EXAMPLES

    return template.format(
        conversation_history=conversation_history,
        few_shot_examples=few_shot_text,
    )


# ---------------------------------------------------------------------------
# Label extraction from LLM output
# ---------------------------------------------------------------------------

def _extract_labels(parsed: dict, instrument: str) -> list[str]:
    """
    Extract frequency/agreement labels from the Step 2 output.

    Looks for 'frequency_label' (PHQ-9/GAD-7) or 'agreement_label' (CompACT-10)
    in the step_2_temporal or step_2_endorsement fields.
    """
    _load_verbalizers()

    step2 = parsed.get("step_2_temporal") or parsed.get("step_2_endorsement") or {}
    if not step2:
        return []

    spec = INSTRUMENTS.get(instrument)
    if not spec:
        return []

    label_key = "agreement_label" if instrument == "CompACT-10" else "frequency_label"
    labels = []

    for i in range(spec["n_items"]):
        item_key = f"item_{i + 1}"
        item_data = step2.get(item_key, {})
        if isinstance(item_data, dict):
            label = item_data.get(label_key, "")
            labels.append(label)
        else:
            labels.append("")

    return labels


# ---------------------------------------------------------------------------
# Score extraction fallbacks
# ---------------------------------------------------------------------------

def _extract_bare_scores(text: str, instrument: str) -> list[int] | None:
    """Extract a scores array from prose output when JSON parsing fails.

    Handles patterns like:
      - "CompACT-10: [3, 3, 4, 3, ...]"
      - "### PHQ-9 Scores\n[1, 2, 1, ...]"
      - "PHQ-9: [1, 2, 1, 2, 1, 2, 2, 2, 0]"
      - Bare "[3, 3, 4, ...]" near end of response

    Returns the scores list if found and valid, else None.
    """
    import re

    spec = INSTRUMENTS.get(instrument)
    if not spec:
        return None

    n_items = spec["n_items"]
    max_val = spec["max_val"]

    # Pattern: look for the instrument name followed by a scores array
    # Handles "PHQ-9", "GAD-7", "CompACT-10", "CompACT10", case-insensitive
    inst_pattern = re.escape(instrument).replace(r"\-", r"[\-\s]?")
    # Match: instrument name ... [scores]
    pattern = rf'(?:{inst_pattern})\s*(?:Scores?|:)?\s*\[([0-9,\s]+)\]'
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        # Fallback: find the LAST bracket array in the text with the right number of elements
        all_arrays = re.findall(r'\[([0-9,\s]+)\]', text)
        for arr_str in reversed(all_arrays):
            nums = [int(x.strip()) for x in arr_str.split(",") if x.strip().isdigit()]
            if len(nums) == n_items:
                match_str = arr_str
                break
        else:
            return None
    else:
        match_str = match.group(1)

    # Parse the numbers
    try:
        scores = [int(x.strip()) for x in match_str.split(",") if x.strip().isdigit()]
    except ValueError:
        return None

    if len(scores) != n_items:
        return None

    # Validate range
    scores = [max(0, min(max_val, s)) for s in scores]
    return scores


def _extract_scores_from_steps(parsed: dict, instrument: str) -> list[int] | None:
    """
    Fallback: extract scores from Step 2 item-level data when the final
    scores array is missing (truncated response or different key name).

    Looks for 'score' field in step_2_temporal or step_2_endorsement items.
    """
    spec = INSTRUMENTS.get(instrument)
    if not spec:
        return None

    step2 = parsed.get("step_2_temporal") or parsed.get("step_2_endorsement") or {}
    if not step2:
        return None

    scores = []
    for i in range(spec["n_items"]):
        item_key = f"item_{i + 1}"
        item_data = step2.get(item_key, {})
        if isinstance(item_data, dict) and "score" in item_data:
            try:
                s = int(item_data["score"])
                scores.append(max(0, min(spec["max_val"], s)))
            except (ValueError, TypeError):
                scores.append(spec["max_val"] // 2)  # midpoint fallback
        else:
            return None  # incomplete step 2, can't reliably extract

    return scores if len(scores) == spec["n_items"] else None


# ---------------------------------------------------------------------------
# Assessment execution
# ---------------------------------------------------------------------------

def _minimal_retry_prompt(instrument: str, conversation_history: str) -> str:
    """Build a calibrated retry prompt when CoT fails to produce valid JSON.

    Includes item descriptions, conservative scoring guidance, and anchors
    to prevent the over-scoring bias seen with bare prompts.
    """
    if instrument == "PHQ-9":
        return f"""You are a clinical psychologist. Read this Spanish therapeutic conversation and predict PHQ-9 scores.

PHQ-9 items (0=not at all, 1=several days, 2=more than half, 3=nearly every day):
1. Little interest or pleasure in doing things
2. Feeling down, depressed, hopeless
3. Trouble sleeping or sleeping too much
4. Feeling tired or little energy
5. Poor appetite or overeating
6. Feeling bad about yourself or failure
7. Trouble concentrating
8. Moving/speaking slowly or being restless
9. Thoughts of self-harm (default to 0 unless clear evidence)

RULES: Score based on evidence in the conversation only. If a symptom is not discussed, score 1 for a clearly distressed patient, 0 otherwise. Most patients score 0-2 per item. Score 3 only with strong explicit evidence. Item 9 defaults to 0.

CONVERSATION:
{conversation_history}

Respond with ONLY this JSON:
{{"{instrument}": [item1, item2, item3, item4, item5, item6, item7, item8, item9]}}"""

    elif instrument == "GAD-7":
        return f"""You are a clinical psychologist. Read this Spanish therapeutic conversation and predict GAD-7 scores.

GAD-7 items (0=never, 1=several days, 2=more than half, 3=nearly every day):
1. Feeling nervous, anxious, on edge
2. Not being able to stop or control worrying
3. Worrying too much about different things
4. Trouble relaxing
5. Being so restless it's hard to sit still
6. Becoming easily annoyed or irritable
7. Feeling afraid as if something awful might happen

RULES: Score based on evidence only. If a symptom is not discussed, score 1 for a clearly anxious patient, 0 otherwise. Distinguish: item 1=general nervousness, items 2-3=worry patterns, item 4=physical tension, item 5=motor restlessness, item 6=irritability, item 7=anticipatory fear.

CONVERSATION:
{conversation_history}

Respond with ONLY this JSON:
{{"{instrument}": [item1, item2, item3, item4, item5, item6, item7]}}"""

    else:  # CompACT-10
        return f"""You are an ACT psychologist. Read this Spanish therapeutic conversation and predict CompACT-10 scores.

CompACT-10 items (0=strongly disagree, 3=neutral, 6=strongly agree):
OPENNESS (reverse: high=more avoidance): 3.Thought suppression, 5.Situational avoidance, 8.Emotional suppression
AWARENESS (reverse: high=more autopilot): 1.Rushing meaningful activities, 6.Inattentive engagement, 9.Autopilot
VALUED ACTION (direct: high=more aligned): 2.Coherent living, 4.Values-aligned, 7.Persistence, 10.Perseverance

RULES: Score the patient's GENERAL TENDENCY, not just this session. For a moderately distressed patient: Openness items typically 3-4, Awareness items 3, Valued Action items 3-4. Do NOT score all items at extremes. Use the full 0-6 range.

CONVERSATION:
{conversation_history}

Respond with ONLY this JSON:
{{"{instrument}": [item1, item2, item3, item4, item5, item6, item7, item8, item9, item10]}}"""


def assess_instrument(
    client: LLMClient,
    instrument: str,
    conversation_history: str,
    use_few_shot: bool = True,
    use_prompt_anchors: bool = False,
) -> AssessmentResult:
    """
    Run a single instrument assessment via LLM.

    Uses chain-of-thought prompt first. If JSON parsing fails, retries
    with a minimal prompt that asks for scores only.
    """
    spec = INSTRUMENTS[instrument]
    prompt = build_prompt(instrument, conversation_history, use_few_shot,
                          use_prompt_anchors=use_prompt_anchors)

    messages = [{"role": "user", "content": prompt}]

    try:
        raw_response = client.complete(messages)
    except Exception as e:
        logger.error("LLM call failed for %s: %s", instrument, e)
        defaults = _default_scores(instrument)
        return AssessmentResult(
            instrument=instrument, scores=defaults,
            error=str(e), raw_response="",
        )

    # Parse response: try JSON first, then extract scores from prose
    parsed = parse_json_response(raw_response)
    if parsed is None:
        # Attempt 2: extract bare scores array from prose response
        # The model often outputs markdown with scores as [1, 2, 3, ...]
        bare_scores = _extract_bare_scores(raw_response, instrument)
        if bare_scores is not None:
            logger.info("%s: extracted scores from prose: %s", instrument, bare_scores)
            parsed = {instrument: bare_scores}
        else:
            # Attempt 3: retry with minimal prompt
            logger.info("CoT parse failed for %s, retrying with minimal prompt", instrument)
            retry_prompt = _minimal_retry_prompt(instrument, conversation_history)
            try:
                retry_response = client.complete([{"role": "user", "content": retry_prompt}])
                parsed = parse_json_response(retry_response)
                if parsed is None:
                    bare_scores = _extract_bare_scores(retry_response, instrument)
                    if bare_scores is not None:
                        parsed = {instrument: bare_scores}
                if parsed is not None:
                    raw_response = retry_response
                    logger.info("Minimal retry succeeded for %s", instrument)
            except Exception:
                pass

    if parsed is None:
        logger.warning("All parse attempts failed for %s, using defaults", instrument)
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
        # Fallback: extract scores from Step 2 item-level data
        scores = _extract_scores_from_steps(parsed, instrument)
        if scores:
            logger.info("%s: extracted scores from Step 2 items: %s", instrument, scores)
        else:
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
        if len(scores) < spec["n_items"]:
            mid = spec["max_val"] // 2
            scores.extend([mid] * (spec["n_items"] - len(scores)))
        else:
            scores = scores[:spec["n_items"]]

    def _safe_int(val, default: int = 0) -> int:
        """Convert to int, falling back to default for non-numeric values."""
        try:
            return int(val)
        except (ValueError, TypeError):
            logger.warning("%s: non-numeric score '%s', using default %d", instrument, val, default)
            return default

    mid = spec["max_val"] // 2
    scores = [max(0, min(spec["max_val"], _safe_int(s, mid))) for s in scores]

    # Extract reasoning steps
    steps = {}
    for step_key in ["step_0_category_scan", "step_0_triflex_scan",
                     "step_1_detection", "step_2_temporal", "step_2_endorsement"]:
        if step_key in parsed:
            steps[step_key] = parsed[step_key]

    # Extract verbalizer labels from Step 2 output
    labels = _extract_labels(parsed, instrument)

    # Apply label-score consistency checking
    label_mismatches = []
    if labels and _VERBALIZER_CONSISTENCY_FN:
        label_mismatches = _VERBALIZER_CONSISTENCY_FN(instrument, scores, labels)
        if label_mismatches:
            logger.warning(
                "%s: %d label-score mismatch(es): %s",
                instrument, len(label_mismatches),
                [(m["item"], m["label"], m["expected_score"], m["actual_score"])
                 for m in label_mismatches],
            )
            # Trust labels over scores
            if _VERBALIZER_RESOLVE_FN:
                scores = _VERBALIZER_RESOLVE_FN(scores, labels, instrument)
                logger.info("%s: scores corrected via verbalizer labels → %s", instrument, scores)

    return AssessmentResult(
        instrument=instrument,
        scores=scores,
        raw_response=raw_response,
        steps=steps,
        labels=labels,
        label_mismatches=label_mismatches,
    )


def assess_all_instruments(
    client: LLMClient,
    conversation_history: str,
    use_few_shot: bool = True,
    use_prompt_anchors: bool = False,
) -> dict[str, AssessmentResult]:
    """
    Run assessments for all three instruments.

    Returns dict mapping instrument name -> AssessmentResult.
    """
    results = {}
    for instrument in ["PHQ-9", "GAD-7", "CompACT-10"]:
        logger.info("Assessing %s...", instrument)
        results[instrument] = assess_instrument(
            client, instrument, conversation_history, use_few_shot,
            use_prompt_anchors=use_prompt_anchors,
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
