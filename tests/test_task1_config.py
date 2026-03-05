"""Tests for Task 1 configuration."""

import pytest
from pathlib import Path

from erisk_task1.config import load_config, PipelineConfig


class TestDefaultConfig:
    def test_default_config_loads(self):
        cfg = PipelineConfig()
        assert cfg.hardware_scenario == "high_vram"
        assert cfg.execution.max_turns == 10
        assert cfg.execution.min_turns == 5

    def test_persona_defaults(self):
        cfg = PipelineConfig()
        assert "Meta-Llama-3-8B-Instruct" in cfg.persona.base_model
        assert cfg.persona.temperature == 0.6
        assert cfg.persona.top_p == 0.9
        assert cfg.persona.max_new_tokens == 256

    def test_persona_system_prompt_is_verbatim(self):
        cfg = PipelineConfig()
        assert "simulated patient" in cfg.persona.system_prompt
        assert "contextual realism" in cfg.persona.system_prompt
        assert "Do not mention you are an AI" in cfg.persona.system_prompt

    def test_interviewer_defaults(self):
        cfg = PipelineConfig()
        assert cfg.interviewer.provider == "openai"
        assert cfg.interviewer.temperature == 0.7

    def test_assessor_defaults(self):
        cfg = PipelineConfig()
        assert cfg.assessor.provider == "ollama"
        assert cfg.assessor.temperature == 0.1

    def test_severity_bands(self):
        from erisk_task1.models import score_to_band, SeverityBand
        assert score_to_band(0) == SeverityBand.MINIMAL
        assert score_to_band(13) == SeverityBand.MINIMAL
        assert score_to_band(14) == SeverityBand.MILD
        assert score_to_band(28) == SeverityBand.MODERATE
        assert score_to_band(29) == SeverityBand.SEVERE


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
hardware_scenario: "limited_vram"
execution:
  max_turns: 8
  min_turns: 4
models:
  assessor:
    provider: "ollama"
    model: "qwen2.5:32b"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        cfg = load_config(yaml_file)
        assert cfg.hardware_scenario == "limited_vram"
        assert cfg.execution.max_turns == 8
        assert cfg.execution.min_turns == 4
        assert cfg.assessor.model == "qwen2.5:32b"

    def test_load_nonexistent_uses_defaults(self):
        cfg = load_config("nonexistent.yaml")
        assert cfg.hardware_scenario == "high_vram"
        assert cfg.execution.max_turns == 10
