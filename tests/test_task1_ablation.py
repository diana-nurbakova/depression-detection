"""Tests for Task 1 ablation study framework."""

import pytest

from erisk_task1.ablation import (
    ABLATION_CONFIGS,
    GENERAL_ASSESSOR_PROMPT,
    AblationConfig,
)


class TestAblationConfigs:
    def test_all_configs_defined(self):
        expected = {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"}
        assert set(ABLATION_CONFIGS.keys()) == expected

    def test_a0_baseline(self):
        a0 = ABLATION_CONFIGS["A0"]
        assert not a0.use_specialized_assessors
        assert not a0.use_linguistic_features
        assert not a0.use_bayesian_prior
        assert not a0.use_justificator

    def test_a1_adds_specialized(self):
        a1 = ABLATION_CONFIGS["A1"]
        assert a1.use_specialized_assessors
        assert not a1.use_linguistic_features
        assert not a1.use_bayesian_prior
        assert not a1.use_justificator

    def test_a2_adds_linguistic(self):
        a2 = ABLATION_CONFIGS["A2"]
        assert a2.use_specialized_assessors
        assert a2.use_linguistic_features
        assert not a2.use_bayesian_prior
        assert not a2.use_justificator

    def test_a3_adds_prior(self):
        a3 = ABLATION_CONFIGS["A3"]
        assert a3.use_specialized_assessors
        assert a3.use_linguistic_features
        assert a3.use_bayesian_prior
        assert not a3.use_justificator

    def test_a4_full_pipeline(self):
        a4 = ABLATION_CONFIGS["A4"]
        assert a4.use_specialized_assessors
        assert a4.use_linguistic_features
        assert a4.use_bayesian_prior
        assert a4.use_justificator

    def test_cumulative_progression(self):
        """Each config should add one component to the previous."""
        configs = [ABLATION_CONFIGS[f"A{i}"] for i in range(5)]

        # A0 -> A1: adds specialized assessors
        assert not configs[0].use_specialized_assessors
        assert configs[1].use_specialized_assessors

        # A1 -> A2: adds linguistic features
        assert not configs[1].use_linguistic_features
        assert configs[2].use_linguistic_features

        # A2 -> A3: adds prior
        assert not configs[2].use_bayesian_prior
        assert configs[3].use_bayesian_prior

        # A3 -> A4: adds justificator
        assert not configs[3].use_justificator
        assert configs[4].use_justificator

    def test_temperature_sweep_configs(self):
        a5 = ABLATION_CONFIGS["A5"]
        a6 = ABLATION_CONFIGS["A6"]
        assert a5.assessor_temperature == 0.05
        assert a6.assessor_temperature == 0.3

    def test_a7_no_prior_variant(self):
        a7 = ABLATION_CONFIGS["A7"]
        assert a7.use_justificator
        assert not a7.use_bayesian_prior

    def test_all_configs_have_names(self):
        for key, cfg in ABLATION_CONFIGS.items():
            assert cfg.name.startswith(key)
            assert len(cfg.description) > 10


class TestGeneralAssessorPrompt:
    def test_covers_all_21_items(self):
        assert "21 BDI-II items" in GENERAL_ASSESSOR_PROMPT
        assert "1-Sadness" in GENERAL_ASSESSOR_PROMPT
        assert "21-Loss of interest in sex" in GENERAL_ASSESSOR_PROMPT

    def test_conservative_suicidal(self):
        assert "Item 9" in GENERAL_ASSESSOR_PROMPT
        assert "require strong evidence" in GENERAL_ASSESSOR_PROMPT

    def test_json_output_format(self):
        assert "valid JSON" in GENERAL_ASSESSOR_PROMPT
        assert '"items"' in GENERAL_ASSESSOR_PROMPT


class TestAblationConfig:
    def test_default_values(self):
        cfg = AblationConfig(name="test", description="test config")
        assert cfg.use_specialized_assessors
        assert cfg.use_linguistic_features
        assert cfg.use_bayesian_prior
        assert cfg.use_justificator
        assert cfg.assessor_temperature == 0.1
        assert cfg.prior_confidence == 0.3
        assert cfg.absolutist_thresholds == (0.005, 0.012, 0.025)
