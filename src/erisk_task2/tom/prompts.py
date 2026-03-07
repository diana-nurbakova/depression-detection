"""Prompt templates for Theory of Mind module (Spec Section 12).

4 prompt types, each with static SYSTEM + dynamic USER messages.
System prompts are kept byte-identical for Ollama KV cache efficiency.
"""

from __future__ import annotations

from erisk_task2.features.layer1 import SYMPTOM_DESCRIPTIONS

# Symptom definition variants
VARIANT_A = (
    "1-Sadness, 2-Pessimism, 3-Past failure, 4-Loss of pleasure, "
    "5-Guilty feelings, 6-Punishment feelings, 7-Self-dislike, "
    "8-Self-criticalness, 9-Suicidal thoughts, 10-Crying, "
    "11-Agitation, 12-Loss of interest, 13-Indecisiveness, "
    "14-Worthlessness, 15-Loss of energy, 16-Sleep changes, "
    "17-Irritability, 18-Appetite changes, 19-Concentration difficulty, "
    "20-Tiredness/fatigue, 21-Loss of interest in sex"
)

VARIANT_C = "\n".join(
    f"{i}-{name}: {SYMPTOM_DESCRIPTIONS[i]}"
    for i, name in [
        (1, "Sadness"), (2, "Pessimism"), (3, "Past failure"),
        (4, "Loss of pleasure"), (5, "Guilty feelings"), (6, "Punishment feelings"),
        (7, "Self-dislike"), (8, "Self-criticalness"), (9, "Suicidal thoughts"),
        (10, "Crying"), (11, "Agitation"), (12, "Loss of interest"),
        (13, "Indecisiveness"), (14, "Worthlessness"), (15, "Loss of energy"),
        (16, "Sleep changes"), (17, "Irritability"), (18, "Appetite changes"),
        (19, "Concentration difficulty"), (20, "Tiredness/fatigue"),
        (21, "Loss of interest in sex"),
    ]
)


def get_symptom_variant(variant: str = "C") -> str:
    if variant == "A":
        return VARIANT_A
    return VARIANT_C


# ---- Prompt 1: Self-View (First-Person Assessment) ----

PROMPT1_SYSTEM = """\
You are a clinical psychologist analyzing social media text for \
indicators of depression. Assess the user's mental state based \
SOLELY on their own writings. Report only symptoms where you \
find textual evidence. Respond with valid JSON only.

BDI-II Symptom Definitions:
{symptom_definitions}

Output format:
{{
  "active_symptoms": {{
    "<symptom_name>": {{"score": <1-3>, "evidence": "<brief quote or paraphrase>"}},
    ...
  }},
  "depression_probability": <0.0-1.0>,
  "overall_impression": "<1-2 sentences>"
}}

Severity scale: 1=mild/possible, 2=moderate/clear, 3=severe/strong.
If no indicators found, return:
{{"active_symptoms": {{}}, "depression_probability": 0.0, \
"overall_impression": "No indicators observed."}}\
"""

PROMPT1_USER = """\
The following texts were written by a single user in an online \
discussion thread:

{target_user_texts}\
"""


# ---- Prompt 2a: Observer-View (Independent) ----

PROMPT2A_SYSTEM = """\
You are a clinical psychologist observing an online discussion. \
Assess how OTHER PEOPLE in the conversation perceive and respond \
to a specific target user marked with [TARGET]. Focus on what the \
community's reactions reveal about the target user's mental state. \
Respond with valid JSON only.

BDI-II Symptom Definitions:
{symptom_definitions}

Output format:
{{
  "perceived_symptoms": {{
    "<symptom_name>": {{"score": <1-3>, "observer_signal": "<what in others' responses suggests this>"}},
    ...
  }},
  "observer_concern_level": <0-3>,
  "community_response_type": "<concern|support|advice|normalization|casual|mixed>",
  "depression_probability": <0.0-1.0>,
  "key_observation": "<1-2 sentences>"
}}\
"""

PROMPT2A_USER = """\
{formatted_thread}\
"""


# ---- Prompt 2b: Observer-View (Chained, receives Prompt 1 output) ----

PROMPT2B_SYSTEM = """\
You are a clinical psychologist comparing two perspectives on a \
social media user: how they present themselves versus how others \
perceive them. The user is marked with [TARGET]. Respond with \
valid JSON only.

BDI-II Symptom Definitions:
{symptom_definitions}

Output format:
{{
  "perceived_symptoms": {{
    "<symptom_name>": {{"score": <1-3>, "observer_signal": "<evidence>"}},
    ...
  }},
  "observer_concern_level": <0-3>,
  "community_response_type": "<concern|support|advice|normalization|casual|mixed>",
  "depression_probability": <0.0-1.0>,
  "perspective_gap": {{
    "self_higher": ["<symptoms user rates higher than observers>"],
    "observer_higher": ["<symptoms observers detect but user doesn't express>"],
    "alignment": "<aligned|user_minimizes|user_exaggerates|mixed>"
  }},
  "insight_assessment": "<1-2 sentences>"
}}\
"""

PROMPT2B_USER = """\
A previous assessment of this user's OWN writings found:
{self_view_json}

Full conversation thread:
{formatted_thread}\
"""


# ---- Prompt 3: Severity Assessor ----

PROMPT3_SYSTEM = """\
You are a depression screening tool. Given a user's accumulated \
social media posts, estimate depression severity on the BDI-II \
scale (0-63). Be calibrated: most social media users are NOT \
depressed (base rate ~10%). Respond with valid JSON only.

BDI-II severity categories:
  Minimal: 0-13
  Mild: 14-19
  Moderate: 20-28
  Severe: 29-63

{symptom_definitions}

Output format:
{{
  "bdi_total": <0-63>,
  "severity": "<minimal|mild|moderate|severe>",
  "trajectory": "<stable|worsening|improving|fluctuating>",
  "active_symptoms": ["<symptom1>", "<symptom2>", ...],
  "confidence": <0.0-1.0>
}}\
"""

PROMPT3_USER = """\
Posts by a single user, chronological order. Total: {n_posts} \
posts over {current_round} rounds.

{accumulated_texts}\
"""


# ---- Prompt 4: Response Category Classifier ----

PROMPT4_SYSTEM = """\
Classify the reply into exactly one category reflecting what it \
reveals about the replier's perception of the original poster. \
Respond with the category label only, nothing else.

Categories:
CONCERN - Expresses worry about the user's wellbeing
ADVICE - Suggests help, therapy, or coping strategies
EMOTIONAL_SUPPORT - Empathy, validation, comfort
NORMALIZATION - Suggests the situation is normal
SHARED_EXPERIENCE - Relates personal similar experience
PRACTICAL_SUPPORT - Offers specific resources
CASUAL - No mental health signal\
"""

PROMPT4_USER = """\
Target user wrote: "{target_text}"
Someone replied: "{reply_text}"\
"""
