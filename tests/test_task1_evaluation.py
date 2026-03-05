"""Tests for Task 1 evaluation harness and metrics."""

import pytest
from pathlib import Path

from erisk_task1.evaluation import (
    GOLDEN_SCORES,
    GOLDEN_KEY_SYMPTOMS,
    BOUNDARY_PERSONAS,
    AblationResult,
    PersonaEvaluation,
    TalkDepConversation,
    _compute_symptom_hit_rate,
    _parse_talkdep_transcript,
    canonicalize_symptom,
    compute_component_contribution,
    evaluate_persona,
    format_comparison_table,
    format_error_analysis,
    load_talkdep_conversations,
)
from erisk_task1.models import SeverityBand, score_to_band


class TestGoldenData:
    def test_12_personas(self):
        assert len(GOLDEN_SCORES) == 12

    def test_all_personas_have_symptoms(self):
        for name in GOLDEN_SCORES:
            assert name in GOLDEN_KEY_SYMPTOMS

    def test_severity_distribution(self):
        bands = [score_to_band(s).value for s in GOLDEN_SCORES.values()]
        assert bands.count("minimal") == 5  # Noah(5), Maya(6), Priya(7), Ethan(12), Gabriel(13)
        assert bands.count("mild") == 1  # Alex(15)
        assert bands.count("moderate") == 3  # James(22), Laura(23), Linda(28)
        assert bands.count("severe") == 3  # Elena(35), Marco(38), Maria(40)

    def test_boundary_personas(self):
        assert BOUNDARY_PERSONAS == {"Ethan", "Gabriel", "Alex", "Linda"}

    def test_symptoms_are_canonical(self):
        """All golden key symptoms should use canonical BDI-II names."""
        from erisk_task1.models import BDI_ITEMS

        canonical_names = set(BDI_ITEMS.values())
        for name, symptoms in GOLDEN_KEY_SYMPTOMS.items():
            for symptom in symptoms:
                assert symptom in canonical_names, (
                    f"{name}: '{symptom}' not in canonical BDI-II names"
                )


class TestCanonicalizeSymptom:
    def test_direct_match(self):
        assert canonicalize_symptom("Sadness") == "Sadness"

    def test_hopelessness_maps_to_pessimism(self):
        assert canonicalize_symptom("Hopelessness") == "Pessimism"
        assert canonicalize_symptom("Feelings of Hopelessness") == "Pessimism"

    def test_self_criticism_maps(self):
        assert canonicalize_symptom("Self-criticism") == "Self-criticalness"
        assert canonicalize_symptom("Self-Criticism") == "Self-criticalness"

    def test_fatigue_maps(self):
        assert canonicalize_symptom("Tiredness") == "Tiredness or fatigue"
        assert canonicalize_symptom("Extreme Fatigue") == "Tiredness or fatigue"
        assert canonicalize_symptom("Fatigue") == "Tiredness or fatigue"

    def test_sleep_maps(self):
        assert canonicalize_symptom("Sleep Disturbances") == "Changes in sleeping pattern"

    def test_unknown_passthrough(self):
        assert canonicalize_symptom("Unknown Symptom") == "Unknown Symptom"


class TestSymptomHitRate:
    def test_perfect_match(self):
        golden = ["Sadness", "Pessimism", "Loss of interest", "Loss of energy"]
        predicted = ["Sadness", "Pessimism", "Loss of interest", "Loss of energy"]
        assert _compute_symptom_hit_rate(golden, predicted) == 1.0

    def test_no_overlap(self):
        golden = ["Sadness", "Pessimism"]
        predicted = ["Crying", "Agitation"]
        assert _compute_symptom_hit_rate(golden, predicted) == 0.0

    def test_partial_overlap(self):
        golden = ["Sadness", "Pessimism", "Loss of interest", "Loss of energy"]
        predicted = ["Sadness", "Pessimism", "Crying", "Agitation"]
        assert _compute_symptom_hit_rate(golden, predicted) == 0.5  # 2/4

    def test_both_empty(self):
        assert _compute_symptom_hit_rate([], []) == 1.0

    def test_golden_empty_predicted_nonempty(self):
        assert _compute_symptom_hit_rate([], ["Sadness"]) == 0.0

    def test_canonical_mapping_applied(self):
        golden = ["Self-criticism"]  # Maps to Self-criticalness
        predicted = ["Self-criticalness"]
        assert _compute_symptom_hit_rate(golden, predicted) == 1.0


class TestEvaluatePersona:
    def test_perfect_prediction(self):
        result = evaluate_persona("Noah", 5, [])
        assert result.band_correct
        assert result.absolute_deviation == 0
        assert result.cr_score == 1.0

    def test_wrong_band(self):
        result = evaluate_persona("Gabriel", 20, ["Sadness"])
        assert not result.band_correct  # 13=Minimal, 20=Moderate
        assert result.absolute_deviation == 7

    def test_cr_score(self):
        # CR = (63 - deviation) / 63
        result = evaluate_persona("Maria", 35, [])
        assert result.absolute_deviation == 5
        assert abs(result.cr_score - (63 - 5) / 63) < 0.001

    def test_boundary_persona_alex(self):
        # Alex: golden=15 (Mild), predict 14 -> still Mild
        result = evaluate_persona("Alex", 14, [])
        assert result.band_correct  # 14 is still Mild

    def test_boundary_persona_alex_miss(self):
        # Alex: golden=15 (Mild), predict 13 -> Minimal (wrong)
        result = evaluate_persona("Alex", 13, [])
        assert not result.band_correct


class TestAblationResult:
    def _make_results(self, predictions: list[tuple[str, int, list[str]]]):
        evals = []
        for name, total, top4 in predictions:
            evals.append(evaluate_persona(name, total, top4))
        return evals

    def test_perfect_dchr(self):
        predictions = [
            ("Noah", 5, []),
            ("Alex", 15, []),
            ("Linda", 28, []),
            ("Maria", 40, []),
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.dchr == 1.0

    def test_half_dchr(self):
        predictions = [
            ("Noah", 5, []),   # Correct: Minimal
            ("Alex", 13, []),  # Wrong: 15->13 crosses to Minimal
            ("Linda", 28, []), # Correct: Moderate
            ("Maria", 20, []), # Wrong: 40->20 crosses to Moderate
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.dchr == 0.5

    def test_mad(self):
        predictions = [
            ("Noah", 7, []),   # deviation=2
            ("Maria", 38, []), # deviation=2
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.mad == 2.0

    def test_adodl(self):
        predictions = [
            ("Noah", 5, []),   # CR = (63-0)/63 = 1.0
            ("Maria", 40, []), # CR = (63-0)/63 = 1.0
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.adodl == 1.0

    def test_adodl_partial(self):
        predictions = [
            ("Noah", 5, []),   # CR = 1.0
            ("Maria", 35, []), # CR = (63-5)/63 ≈ 0.921
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert 0.9 < r.adodl < 1.0

    def test_boundary_accuracy(self):
        predictions = [
            ("Ethan", 12, []),   # Correct
            ("Gabriel", 14, []), # Wrong: 13->14 crosses to Mild
            ("Alex", 15, []),    # Correct
            ("Linda", 28, []),   # Correct
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.boundary_accuracy == 0.75  # 3/4

    def test_ashr_proxy(self):
        predictions = [
            ("Maria", 40, ["Sadness", "Self-criticalness", "Loss of interest", "Tiredness or fatigue"]),
        ]
        r = AblationResult("test", self._make_results(predictions))
        assert r.ashr_proxy == 1.0


class TestParseTalkDepTranscript:
    def test_basic_parsing(self):
        raw = (
            "**Therapist:** How are you?\n"
            "**Noah:** I'm doing okay.\n"
            "\n"
            "**Therapist:** Good to hear.\n"
        )
        result = _parse_talkdep_transcript(raw)
        assert "Interviewer: How are you?" in result
        assert "Person: I'm doing okay." in result
        assert "Interviewer: Good to hear." in result

    def test_empty_text(self):
        assert _parse_talkdep_transcript("") == ""

    def test_skips_non_dialogue(self):
        raw = "Patient name: Noah\n\n**Therapist:** Hello.\n"
        result = _parse_talkdep_transcript(raw)
        assert "Patient name" not in result
        assert "Interviewer: Hello." in result


class TestLoadTalkDep:
    def test_loads_conversations(self):
        talkdep_dir = Path("data/TalkDep")
        if not talkdep_dir.exists():
            pytest.skip("TalkDep data not available")

        convs = load_talkdep_conversations(talkdep_dir)
        assert len(convs) == 12

        # Should be sorted by golden score ascending
        scores = [c.golden_total for c in convs]
        assert scores == sorted(scores)

        # Each conversation should have non-empty transcript
        for c in convs:
            assert len(c.transcript) > 100
            assert c.name in GOLDEN_SCORES

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_talkdep_conversations("/nonexistent/path")


class TestComponentContribution:
    def test_improvement(self):
        baseline_evals = [
            evaluate_persona("Noah", 8, []),  # off by 3
            evaluate_persona("Maria", 35, []),  # off by 5
        ]
        enhanced_evals = [
            evaluate_persona("Noah", 5, []),  # perfect
            evaluate_persona("Maria", 40, []),  # perfect
        ]
        baseline = AblationResult("A0", baseline_evals)
        enhanced = AblationResult("A1", enhanced_evals)

        contrib = compute_component_contribution(baseline, enhanced)
        assert contrib["improved"] == 2
        assert contrib["worsened"] == 0
        assert contrib["mad_delta"] < 0  # MAD should decrease


class TestFormatting:
    def test_comparison_table_not_empty(self):
        evals = [evaluate_persona("Noah", 5, [])]
        r = AblationResult("A0", evals)
        table = format_comparison_table([r])
        assert "A0" in table
        assert "DCHR" in table

    def test_error_analysis_all_correct(self):
        evals = [evaluate_persona("Noah", 5, [])]
        r = AblationResult("A0", evals)
        analysis = format_error_analysis(r)
        assert "All personas correctly classified" in analysis

    def test_error_analysis_with_errors(self):
        evals = [evaluate_persona("Alex", 13, [])]  # Wrong band
        r = AblationResult("A0", evals)
        analysis = format_error_analysis(r)
        assert "Alex" in analysis
        assert "BOUNDARY" in analysis
