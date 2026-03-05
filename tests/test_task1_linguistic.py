"""Tests for Task 1 linguistic feature extraction."""

import pytest

from erisk_task1.linguistic import (
    extract_features,
    compute_cumulative_features,
    estimate_engagement_band,
    detect_persona_profile,
)
from erisk_task1.models import SeverityBand


class TestExtractFeatures:
    def test_basic_word_count(self):
        feats = extract_features("I feel sad today")
        assert feats.word_count == 4

    def test_absolutist_words(self):
        feats = extract_features("I always feel sad and I never feel happy")
        assert feats.absolutist_count >= 2
        assert "always" in feats.absolutist_words_found
        assert "never" in feats.absolutist_words_found

    def test_absolutist_ratio(self):
        feats = extract_features("I always feel nothing will change")
        assert feats.absolutist_ratio > 0

    def test_first_person_singular(self):
        feats = extract_features("I think I am doing okay but I feel tired")
        assert feats.first_person_singular_ratio > 0.2

    def test_first_person_plural(self):
        feats = extract_features("We went out and our friends joined us")
        assert feats.first_person_plural_ratio > 0

    def test_negative_emotion(self):
        feats = extract_features("I feel sad and hopeless about everything")
        assert feats.negative_emotion_count >= 2

    def test_positive_emotion(self):
        feats = extract_features("I am happy and excited about the trip")
        assert feats.positive_emotion_count >= 2

    def test_sadness_words(self):
        feats = extract_features("I feel sad and miserable all the time")
        assert "sad" in feats.sadness_words
        assert "miserable" in feats.sadness_words

    def test_anger_words(self):
        feats = extract_features("I am frustrated and annoyed at everything")
        assert "frustrated" in feats.anger_words
        assert "annoyed" in feats.anger_words

    def test_hedging_count(self):
        feats = extract_features("I guess maybe I don't know if it will help")
        assert feats.hedging_count >= 2

    def test_coping_count(self):
        feats = extract_features("I can try to do that and maybe I could go for a walk")
        assert feats.coping_count >= 1

    def test_sleep_keywords(self):
        feats = extract_features("I can't sleep and I keep waking up at night")
        assert len(feats.sleep_keywords) >= 1

    def test_appetite_keywords(self):
        feats = extract_features("I haven't been eating much and I skip meals")
        assert len(feats.appetite_keywords) >= 1

    def test_energy_keywords(self):
        feats = extract_features("I feel exhausted and drained all the time")
        assert len(feats.energy_keywords) >= 1

    def test_empty_text(self):
        feats = extract_features("")
        assert feats.word_count == 1  # Avoid div-by-zero: min 1
        assert feats.absolutist_count == 0

    def test_sentence_count(self):
        feats = extract_features("I am fine. Everything is okay. Nothing to worry about.")
        assert feats.sentence_count == 3


class TestCumulativeFeatures:
    def test_empty_history(self):
        cum = compute_cumulative_features([])
        assert cum["absolutist_density"] == 0.0
        assert cum["absolutist_band"] == SeverityBand.MINIMAL

    def test_minimal_absolutist_density(self):
        feats = [extract_features("I am doing fine today")]
        cum = compute_cumulative_features(feats)
        assert cum["absolutist_density"] < 0.005
        assert cum["absolutist_band"] == SeverityBand.MINIMAL

    def test_severe_absolutist_density(self):
        feats = [
            extract_features("Nothing always everything never always nothing completely totally"),
        ]
        cum = compute_cumulative_features(feats)
        assert cum["absolutist_density"] >= 0.025
        assert cum["absolutist_band"] == SeverityBand.SEVERE

    def test_accumulates_across_turns(self):
        feats = [
            extract_features("I am fine"),
            extract_features("Everything is okay"),
            extract_features("Nothing to worry about"),
        ]
        cum = compute_cumulative_features(feats)
        assert cum["total_words"] > 0
        assert cum["avg_response_length"] > 0


class TestEngagementBand:
    def test_empty_history_is_minimal(self):
        assert estimate_engagement_band([]) == SeverityBand.MINIMAL

    def test_long_responses_with_coping_is_minimal(self):
        # Simulate long, engaged responses with coping language
        text = "I can try to go for a walk. " * 15 + "That might help me feel better. I'll give it a shot."
        feats = [extract_features(text)] * 3
        band = estimate_engagement_band(feats)
        assert band == SeverityBand.MINIMAL

    def test_very_short_responses_is_severe(self):
        feats = [extract_features("No.")] * 5
        band = estimate_engagement_band(feats)
        assert band == SeverityBand.SEVERE


class TestPersonaProfile:
    def test_engaged_coping(self):
        text = "I can try that. " * 5 + "Maybe I could go for a walk. " * 5
        feats = [extract_features(text)] * 3
        profile = detect_persona_profile(feats)
        assert profile == "Engaged-coping"

    def test_hopeless_withdrawn(self):
        text = "Nothing. Always bad. Everything fails. Never good."
        feats = [extract_features(text)] * 5
        profile = detect_persona_profile(feats)
        assert profile == "Hopeless-withdrawn"
