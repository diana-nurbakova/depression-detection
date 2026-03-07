"""Static mapping tables between external datasets and ASRS symptoms.

Maps RedSM5 DSM-5 depression categories, BDI-Sen-2.0 symptom names,
eRisk 2023 BDI-II queries, and eRisk 2025 T1 BDI-II queries to
ASRS-v1.1 item numbers. These are fixed domain knowledge from the
annotation protocol spec v3 (Sections 2.2–2.4).
"""

from __future__ import annotations

from collections import defaultdict

# ---------------------------------------------------------------------------
# RedSM5 DSM-5 category -> ASRS item numbers
# Source: annotation_protocol_spec_v2.md Section 2.2
# ---------------------------------------------------------------------------

REDSM5_TO_ASRS: dict[str, list[int]] = {
    "COGNITIVE_ISSUES": [7, 8, 9, 10, 11],   # -> Sustained Attention/Distractibility
    "PSYCHOMOTOR": [5, 6, 12, 13],            # -> Motor H/I
    "FATIGUE": [4, 7],                        # -> Avoidance / Unwinding
    "SLEEP_ISSUES": [7],                      # -> Difficulty unwinding
    "ANHEDONIA": [4, 11],                     # -> Avoidance / Selective attention
}

# RedSM5 category descriptions for logging
REDSM5_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "COGNITIVE_ISSUES": "Depression-related cognitive difficulty (concentration, decision-making)",
    "PSYCHOMOTOR": "Depression-related psychomotor agitation/retardation",
    "FATIGUE": "Depression fatigue vs. ADHD avoidance/understimulation",
    "SLEEP_ISSUES": "Depression sleep disruption vs. ADHD racing mind at bedtime",
    "ANHEDONIA": "Depression amotivation vs. ADHD selective attention",
}

# All DSM-5 categories in RedSM5 (including those without ASRS mapping)
ALL_REDSM5_CATEGORIES = [
    "ANHEDONIA", "APPETITE_CHANGE", "COGNITIVE_ISSUES", "DEPRESSED_MOOD",
    "FATIGUE", "PSYCHOMOTOR", "SLEEP_ISSUES", "SPECIAL_CASE",
    "SUICIDAL_THOUGHTS", "WORTHLESSNESS",
]

# ---------------------------------------------------------------------------
# BDI-II query number -> ASRS item numbers
# Source: annotation_protocol_spec_v2.md Section 2.3
# ---------------------------------------------------------------------------

BDIII_TO_ASRS: dict[int, list[int]] = {
    15: [4, 7],          # Loss of Energy -> Avoidance, Unwinding
    16: [7],             # Changes in Sleep -> Difficulty unwinding
    19: [8, 9, 10, 11],  # Concentration Difficulty -> Sustained Attention items
    11: [5, 6, 12, 13],  # Agitation -> Motor H/I items
    13: [1, 2],          # Indecisiveness -> Organization/Planning items
}

# BDI-II query names for logging and display
BDIII_QUERY_NAMES: dict[int, str] = {
    1: "Sadness",
    2: "Pessimism",
    3: "Past Failure",
    4: "Loss of Pleasure",
    5: "Guilty Feelings",
    6: "Punishment Feelings",
    7: "Self-Dislike",
    8: "Self-Criticalness",
    9: "Suicidal Thoughts",
    10: "Crying",
    11: "Agitation",
    12: "Loss of Interest",
    13: "Indecisiveness",
    14: "Worthlessness",
    15: "Loss of Energy",
    16: "Changes in Sleep",
    17: "Irritability",
    18: "Changes in Appetite",
    19: "Concentration Difficulty",
    20: "Tiredness/Fatigue",
    21: "Loss of Interest in Sex",
}

# BDI-II queries that overlap with ASRS (subset of all 21)
BDIII_OVERLAPPING_QUERIES: list[int] = sorted(BDIII_TO_ASRS.keys())

# ---------------------------------------------------------------------------
# BDI-Sen-2.0 symptom name -> ASRS item numbers
# Source: annotation_protocol_spec_v3.md Section 2.4
# ---------------------------------------------------------------------------

BDISEN_TO_ASRS: dict[str, list[int]] = {
    "Concentration_difficulty": [8, 9, 10, 11],  # -> Sustained Attention items
    "Agitation": [5, 6, 12, 13],                 # -> Motor H/I items
    "Loss_of_energy": [4, 7],                    # -> Avoidance / Unwinding
    "Tiredness_or_fatigue": [4, 7],              # -> Avoidance / Unwinding
    "Indecision": [1, 2],                        # -> Organization/Planning items
    "Change_of_sleep": [7],                      # -> Difficulty unwinding
}

# BDI-Sen symptom descriptions for logging
BDISEN_SYMPTOM_DESCRIPTIONS: dict[str, str] = {
    "Concentration_difficulty": "Graded depression concentration difficulty (severity 0-3)",
    "Agitation": "Graded depression agitation (severity 0-3)",
    "Loss_of_energy": "Graded depression energy loss (severity 0-3)",
    "Tiredness_or_fatigue": "Graded depression fatigue (severity 0-3)",
    "Indecision": "Graded depression indecisiveness (severity 0-3)",
    "Change_of_sleep": "Graded depression sleep changes (severity 0-3)",
}

# All BDI-Sen symptom names (including those without ASRS mapping)
ALL_BDISEN_SYMPTOMS = [
    "Agitation", "Change_of_sleep", "Changes_in_appetite",
    "Concentration_difficulty", "Crying", "Feelings_of_worthlessness",
    "Guilty_feelings", "Indecision", "Irritability",
    "Loss_of_energy", "Loss_of_interest_in_sex", "Loss_of_Pleasure",
    "Pessimism", "Sadness", "Self-dislike", "Self-incrimination",
    "Sense_of_failure", "Sense_of_punishment", "Social_withdrawal",
    "Suicidal_ideas", "Tiredness_or_fatigue",
]

# BDI-Sen symptoms that overlap with ASRS
BDISEN_OVERLAPPING_SYMPTOMS: list[str] = sorted(BDISEN_TO_ASRS.keys())

# ---------------------------------------------------------------------------
# Computed inverses
# ---------------------------------------------------------------------------


def _build_inverse_map(forward: dict[str | int, list[int]]) -> dict[int, list]:
    """Build inverse mapping: ASRS item -> list of source keys."""
    inverse: dict[int, list] = defaultdict(list)
    for source_key, asrs_items in forward.items():
        for item in asrs_items:
            inverse[item].append(source_key)
    return dict(inverse)


# ASRS item -> list of RedSM5 DSM-5 categories that map to it
ASRS_TO_REDSM5: dict[int, list[str]] = _build_inverse_map(REDSM5_TO_ASRS)

# ASRS item -> list of BDI-II query numbers that map to it
ASRS_TO_BDIII: dict[int, list[int]] = _build_inverse_map(BDIII_TO_ASRS)

# ASRS item -> list of BDI-Sen symptom names that map to it
ASRS_TO_BDISEN: dict[int, list[str]] = _build_inverse_map(BDISEN_TO_ASRS)

# ASRS items with ANY external data source (RedSM5, BDI-Sen, eRisk2023/2025)
ASRS_ITEMS_WITH_EXTERNAL_DATA: set[int] = (
    set(ASRS_TO_REDSM5.keys())
    | set(ASRS_TO_BDIII.keys())
    | set(ASRS_TO_BDISEN.keys())
)

# ASRS items with NO external data source (Verbal H/I items 15-18 mostly)
ALL_ASRS_ITEMS = set(range(1, 19))
ASRS_ITEMS_WITHOUT_EXTERNAL_DATA: set[int] = ALL_ASRS_ITEMS - ASRS_ITEMS_WITH_EXTERNAL_DATA
