"""Linguistic feature extraction for persona responses."""

from __future__ import annotations

import re
from collections import Counter

from .models import LinguisticFeatures, SeverityBand

# --- Word lists ---

ABSOLUTIST_WORDS = frozenset({
    "always", "never", "nothing", "must", "every", "totally", "completely",
    "constantly", "entirely", "all", "definitely", "full", "everything",
    "absolutely", "cannot", "impossible",
})

FIRST_PERSON_SINGULAR = frozenset({"i", "me", "my", "myself", "mine"})
FIRST_PERSON_PLURAL = frozenset({"we", "us", "our", "ourselves", "ours"})

NEGATIVE_EMOTION_WORDS = frozenset({
    "sad", "unhappy", "miserable", "down", "blue", "low", "empty", "hopeless",
    "gloomy", "hurt", "pain", "suffer", "awful", "terrible", "horrible",
    "dreadful", "depressed", "heartbroken", "devastated", "lonely", "anxious",
    "worried", "scared", "afraid", "overwhelmed", "stressed", "numb",
})

POSITIVE_EMOTION_WORDS = frozenset({
    "happy", "good", "great", "wonderful", "amazing", "excellent", "love",
    "enjoy", "fun", "glad", "pleased", "grateful", "thankful", "excited",
    "hopeful", "proud", "satisfied", "content", "cheerful", "joyful",
    "delighted", "optimistic", "confident",
})

SADNESS_WORDS = frozenset({
    "sad", "unhappy", "miserable", "gloomy", "heartbroken", "devastated",
    "crying", "tears", "depressed", "hopeless", "despair", "sorrow",
    "grief", "mourning", "melancholy", "despondent",
})

ANGER_WORDS = frozenset({
    "angry", "frustrated", "annoyed", "irritated", "furious", "mad",
    "rage", "snap", "snapped", "bothered", "agitated", "hostile",
    "resentful", "bitter",
})

DISCREPANCY_WORDS = frozenset({
    "should", "would", "could", "want", "need", "hope", "try",
    "ought", "wish", "desire", "expect",
})

TENTATIVE_WORDS = frozenset({
    "maybe", "perhaps", "guess", "probably", "possibly", "might",
    "somewhat", "sort of", "kind of", "i think", "not sure",
})

HEDGING_PHRASES = [
    "i guess", "maybe", "i don't know", "not sure", "i suppose",
    "sort of", "kind of", "i could try", "i might", "probably",
    "not really", "i think", "perhaps",
]

COPING_PHRASES = [
    "i can try", "i could try", "i will try", "i'll try",
    "that might help", "good idea", "i should", "i want to",
    "i plan to", "looking forward", "i'm going to", "i hope to",
    "i'll give it a shot", "maybe i could",
]

# Symptom-specific keyword lists
SLEEP_KEYWORDS = frozenset({
    "sleep", "insomnia", "awake", "wake", "waking", "bed", "rest",
    "nightmare", "tossing", "turning", "nap", "drowsy", "restless",
    "oversleep", "oversleeping",
})

APPETITE_KEYWORDS = frozenset({
    "eat", "eating", "food", "hungry", "appetite", "meal", "meals",
    "weight", "snack", "binge", "starving", "skip", "skipping",
})

ENERGY_KEYWORDS = frozenset({
    "tired", "exhausted", "energy", "drained", "fatigue", "fatigued",
    "wiped", "sluggish", "dragging", "lethargic", "weary",
})

ANHEDONIA_KEYWORDS = frozenset({
    "enjoy", "fun", "pleasure", "interest", "boring", "bored",
    "hobby", "hobbies", "motivation", "passionate", "excited",
    "enthusiasm", "pointless",
})

WORTHLESSNESS_KEYWORDS = frozenset({
    "worthless", "useless", "failure", "blame", "fault", "guilty",
    "burden", "deserve", "pathetic", "incompetent", "inadequate",
    "good enough",
})

SUICIDAL_KEYWORDS = frozenset({
    "end it", "no point", "burden", "disappear", "give up",
    "better off without", "can't go on", "no way out", "sleep forever",
    "not be here", "don't want to wake",
})


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r"\b[a-z']+\b", text.lower())


def _count_matches(words: list[str], word_set: frozenset) -> tuple[int, list[str]]:
    """Count how many words match the given set, return count and matched words."""
    matched = [w for w in words if w in word_set]
    return len(matched), matched


def _count_phrase_matches(text_lower: str, phrases: list[str]) -> int:
    """Count how many phrases appear in the text."""
    return sum(1 for p in phrases if p in text_lower)


def _count_keyword_hits(text_lower: str, keywords: frozenset) -> list[str]:
    """Find keyword hits (supports multi-word keywords)."""
    hits = []
    for kw in keywords:
        if " " in kw:
            if kw in text_lower:
                hits.append(kw)
        else:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                hits.append(kw)
    return hits


def extract_features(text: str) -> LinguisticFeatures:
    """Extract linguistic features from a single persona response."""
    text_lower = text.lower()
    words = _tokenize(text)
    total_words = len(words) or 1  # avoid division by zero

    # Sentence count (approximate)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Pronoun ratios
    fps_count, _ = _count_matches(words, FIRST_PERSON_SINGULAR)
    fpp_count, _ = _count_matches(words, FIRST_PERSON_PLURAL)

    # Absolutist language
    abs_count, abs_found = _count_matches(words, ABSOLUTIST_WORDS)

    # Emotion words
    neg_count, _ = _count_matches(words, NEGATIVE_EMOTION_WORDS)
    pos_count, _ = _count_matches(words, POSITIVE_EMOTION_WORDS)
    _, sad_found = _count_matches(words, SADNESS_WORDS)
    _, anger_found = _count_matches(words, ANGER_WORDS)

    # Cognitive style
    disc_count, _ = _count_matches(words, DISCREPANCY_WORDS)
    tent_count, _ = _count_matches(words, TENTATIVE_WORDS)

    # Hedging and coping
    hedging = _count_phrase_matches(text_lower, HEDGING_PHRASES)
    coping = _count_phrase_matches(text_lower, COPING_PHRASES)

    # Symptom keywords
    sleep_hits = _count_keyword_hits(text_lower, SLEEP_KEYWORDS)
    appetite_hits = _count_keyword_hits(text_lower, APPETITE_KEYWORDS)
    energy_hits = _count_keyword_hits(text_lower, ENERGY_KEYWORDS)
    anhedonia_hits = _count_keyword_hits(text_lower, ANHEDONIA_KEYWORDS)
    worthlessness_hits = _count_keyword_hits(text_lower, WORTHLESSNESS_KEYWORDS)
    suicidal_hits = _count_keyword_hits(text_lower, SUICIDAL_KEYWORDS)

    return LinguisticFeatures(
        word_count=total_words,
        sentence_count=sentence_count,
        first_person_singular_ratio=fps_count / total_words,
        first_person_plural_ratio=fpp_count / total_words,
        absolutist_count=abs_count,
        absolutist_ratio=abs_count / total_words,
        absolutist_words_found=abs_found,
        negative_emotion_count=neg_count,
        positive_emotion_count=pos_count,
        sadness_words=sad_found,
        anger_words=anger_found,
        discrepancy_count=disc_count,
        tentative_count=tent_count,
        hedging_count=hedging,
        coping_count=coping,
        sleep_keywords=sleep_hits,
        appetite_keywords=appetite_hits,
        energy_keywords=energy_hits,
        anhedonia_keywords=anhedonia_hits,
        worthlessness_keywords=worthlessness_hits,
        suicidal_keywords=suicidal_hits,
    )


def compute_cumulative_features(
    features_history: list[LinguisticFeatures],
) -> dict:
    """Compute cumulative linguistic statistics across all turns."""
    if not features_history:
        return {
            "absolutist_density": 0.0,
            "absolutist_band": SeverityBand.MINIMAL,
            "total_absolutist": 0,
            "total_words": 0,
            "avg_response_length": 0,
            "total_hedging": 0,
            "total_coping": 0,
            "total_negative_emotion": 0,
            "total_positive_emotion": 0,
        }

    total_words = sum(f.word_count for f in features_history)
    total_abs = sum(f.absolutist_count for f in features_history)
    density = total_abs / total_words if total_words > 0 else 0

    # Calibrated thresholds from TalkDep validation
    if density < 0.005:
        abs_band = SeverityBand.MINIMAL
    elif density < 0.012:
        abs_band = SeverityBand.MILD
    elif density < 0.025:
        abs_band = SeverityBand.MODERATE
    else:
        abs_band = SeverityBand.SEVERE

    return {
        "absolutist_density": density,
        "absolutist_band": abs_band,
        "total_absolutist": total_abs,
        "total_words": total_words,
        "avg_response_length": total_words / len(features_history),
        "total_hedging": sum(f.hedging_count for f in features_history),
        "total_coping": sum(f.coping_count for f in features_history),
        "total_negative_emotion": sum(f.negative_emotion_count for f in features_history),
        "total_positive_emotion": sum(f.positive_emotion_count for f in features_history),
    }


def estimate_engagement_band(features_history: list[LinguisticFeatures]) -> SeverityBand:
    """Estimate severity band from response engagement patterns."""
    if not features_history:
        return SeverityBand.MINIMAL

    cum = compute_cumulative_features(features_history)
    avg_len = cum["avg_response_length"]
    total_coping = cum["total_coping"]
    total_hedging = cum["total_hedging"]

    # High engagement + solution-oriented → Minimal
    if avg_len > 30 and total_coping >= 3:
        return SeverityBand.MINIMAL
    # Moderate engagement + some hedging → Mild
    if avg_len > 20 and total_hedging <= 5:
        return SeverityBand.MILD
    # Low engagement + resignation → Moderate
    if avg_len > 10:
        return SeverityBand.MODERATE
    # Very short responses + flat → Severe
    return SeverityBand.SEVERE


def detect_persona_profile(features_history: list[LinguisticFeatures]) -> str:
    """Detect persona behavioural profile from linguistic features."""
    cum = compute_cumulative_features(features_history)

    total_coping = cum["total_coping"]
    total_hedging = cum["total_hedging"]
    avg_len = cum["avg_response_length"]
    abs_density = cum["absolutist_density"]

    if total_coping >= 3 and avg_len > 25:
        return "Engaged-coping"
    if total_hedging >= 8 and avg_len > 20:
        return "Hedging-deflecting"
    if avg_len < 15 and abs_density >= 0.02:
        return "Hopeless-withdrawn"
    if avg_len < 20 and total_coping == 0:
        return "Dismissive-flat"
    return "Mixed"
