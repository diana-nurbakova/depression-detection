"""Configuration management for the Task 1 pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    provider: str  # "ollama", "openai", "local"
    model: str
    temperature: float = 0.1
    max_tokens: int = 2000
    fallback_model: str = ""


@dataclass
class PersonaModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_pattern: str = "Anxo/erisk26-task1-patient-{id:02d}-adapter"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    system_prompt: str = (
        "You are a simulated patient. Act realistically based on your "
        "internal training. Ensure contextual realism. Avoid overly detailed "
        "or formal speech. Keep natural speaking style (e.g., short answers, "
        "hesitations, casual expressions). Do not mention you are an AI."
    )


@dataclass
class ExecutionConfig:
    assess_every_n_turns: int = 1
    parallel_assessors: bool = True
    max_turns: int = 10
    min_turns: int = 5
    termination_confidence: float = 0.5


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    timeout_seconds: int = 300
    retry_attempts: int = 2


@dataclass
class OpenAIConfig:
    api_key: str = ""
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class TogetherConfig:
    base_url: str = "https://api.together.xyz/v1"
    api_key: str = ""
    timeout_seconds: int = 120
    retry_attempts: int = 3


@dataclass
class SentenceTransformerConfig:
    """Config for Tier 2 symptom sentence transformer."""
    enabled: bool = False
    base_model: str = "all-mpnet-base-v2"
    model_path: str = ""  # path to fine-tuned model; empty = use base
    max_seq_length: int = 128
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class CorrectionConfig:
    """Post-hoc score correction config per run."""
    run1: str = "none"
    run2: str = "flat_minus_2"
    run3: str = "proportional_085"


@dataclass
class LoggingConfig:
    output_dir: str = "./runs/task1"
    log_level: str = "INFO"
    save_raw_llm_responses: bool = True
    save_linguistic_features: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the Task 1 pipeline."""
    hardware_scenario: str = "high_vram"

    persona: PersonaModelConfig = field(default_factory=PersonaModelConfig)
    interviewer: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="openai", model="gpt-5-nano", temperature=0.7, max_tokens=1000
        )
    )
    assessor: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            model="llama3.3:70b",
            temperature=0.1,
            max_tokens=2000,
            fallback_model="qwen3:32b",
        )
    )
    orchestrator_llm: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            model="llama3.3:70b",
            temperature=0.3,
            max_tokens=1500,
        )
    )
    justificator: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider="ollama",
            model="llama3.3:70b",
            temperature=0.2,
            max_tokens=3000,
        )
    )

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    sentence_transformer: SentenceTransformerConfig = field(
        default_factory=SentenceTransformerConfig
    )
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    together: TogetherConfig = field(default_factory=TogetherConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)

    # Run configuration
    run_id: int = 1
    run_type: str = "automated"  # "automated" or "manual"
    persona_ids: list[int] = field(default_factory=lambda: list(range(20)))


def load_config(config_path: str | Path = "config/task1.yaml") -> PipelineConfig:
    """Load pipeline configuration from YAML file + environment."""
    load_dotenv()

    cfg = PipelineConfig()

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        if "hardware_scenario" in raw:
            cfg.hardware_scenario = raw["hardware_scenario"]

        # Models
        models = raw.get("models", {})
        if "persona" in models:
            p = models["persona"]
            cfg.persona.base_model = p.get("base_model", cfg.persona.base_model)
            cfg.persona.adapter_pattern = p.get("adapter_pattern", cfg.persona.adapter_pattern)
            cfg.persona.torch_dtype = p.get("torch_dtype", cfg.persona.torch_dtype)
            cfg.persona.device_map = p.get("device_map", cfg.persona.device_map)
            gen = p.get("generation", {})
            cfg.persona.temperature = gen.get("temperature", cfg.persona.temperature)
            cfg.persona.top_p = gen.get("top_p", cfg.persona.top_p)
            cfg.persona.max_new_tokens = gen.get("max_new_tokens", cfg.persona.max_new_tokens)
            if "system_prompt" in p:
                cfg.persona.system_prompt = p["system_prompt"]

        for name in ("interviewer", "assessor", "orchestrator_llm", "justificator"):
            if name in models:
                m = models[name]
                mc = getattr(cfg, name)
                mc.provider = m.get("provider", mc.provider)
                mc.model = m.get("model", mc.model)
                mc.temperature = m.get("temperature", mc.temperature)
                mc.max_tokens = m.get("max_tokens", mc.max_tokens)
                mc.fallback_model = m.get("fallback_model", mc.fallback_model)

        # Execution
        if "execution" in raw:
            e = raw["execution"]
            cfg.execution.assess_every_n_turns = e.get(
                "assess_every_n_turns", cfg.execution.assess_every_n_turns
            )
            cfg.execution.parallel_assessors = e.get(
                "parallel_assessors", cfg.execution.parallel_assessors
            )
            cfg.execution.max_turns = e.get("max_turns", cfg.execution.max_turns)
            cfg.execution.min_turns = e.get("min_turns", cfg.execution.min_turns)
            cfg.execution.termination_confidence = e.get(
                "termination_confidence", cfg.execution.termination_confidence
            )

        # Ollama
        if "ollama" in raw:
            o = raw["ollama"]
            cfg.ollama.base_url = o.get("base_url", cfg.ollama.base_url)
            cfg.ollama.timeout_seconds = o.get("timeout_seconds", cfg.ollama.timeout_seconds)
            cfg.ollama.retry_attempts = o.get("retry_attempts", cfg.ollama.retry_attempts)

        # OpenAI
        if "openai" in raw:
            o = raw["openai"]
            cfg.openai.timeout_seconds = o.get("timeout_seconds", cfg.openai.timeout_seconds)
            cfg.openai.retry_attempts = o.get("retry_attempts", cfg.openai.retry_attempts)

        # Together
        if "together" in raw:
            t = raw["together"]
            cfg.together.base_url = t.get("base_url", cfg.together.base_url)
            cfg.together.timeout_seconds = t.get("timeout_seconds", cfg.together.timeout_seconds)
            cfg.together.retry_attempts = t.get("retry_attempts", cfg.together.retry_attempts)

        # Sentence transformer
        if "sentence_transformer" in raw:
            st = raw["sentence_transformer"]
            cfg.sentence_transformer.enabled = st.get(
                "enabled", cfg.sentence_transformer.enabled
            )
            cfg.sentence_transformer.base_model = st.get(
                "base_model", cfg.sentence_transformer.base_model
            )
            cfg.sentence_transformer.model_path = st.get(
                "model_path", cfg.sentence_transformer.model_path
            )
            cfg.sentence_transformer.max_seq_length = st.get(
                "max_seq_length", cfg.sentence_transformer.max_seq_length
            )
            cfg.sentence_transformer.device = st.get(
                "device", cfg.sentence_transformer.device
            )
            cfg.sentence_transformer.batch_size = st.get(
                "batch_size", cfg.sentence_transformer.batch_size
            )

        # Correction
        if "correction" in raw:
            c = raw["correction"]
            cfg.correction.run1 = c.get("run1", cfg.correction.run1)
            cfg.correction.run2 = c.get("run2", cfg.correction.run2)
            cfg.correction.run3 = c.get("run3", cfg.correction.run3)

        # Logging
        if "logging" in raw:
            lg = raw["logging"]
            cfg.logging.output_dir = lg.get("output_dir", cfg.logging.output_dir)
            cfg.logging.log_level = lg.get("log_level", cfg.logging.log_level)
            cfg.logging.save_raw_llm_responses = lg.get(
                "save_raw_llm_responses", cfg.logging.save_raw_llm_responses
            )
            cfg.logging.save_linguistic_features = lg.get(
                "save_linguistic_features", cfg.logging.save_linguistic_features
            )

        # Run config
        if "run" in raw:
            r = raw["run"]
            cfg.run_id = r.get("id", cfg.run_id)
            cfg.run_type = r.get("type", cfg.run_type)
            cfg.persona_ids = r.get("persona_ids", cfg.persona_ids)

    # Environment overrides
    cfg.ollama.base_url = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)
    cfg.ollama.api_key = os.getenv("OLLAMA_API_KEY", cfg.ollama.api_key)
    cfg.openai.api_key = os.getenv("OPENAI_API_KEY", cfg.openai.api_key)
    cfg.together.api_key = os.getenv("TOGETHER_API_KEY", cfg.together.api_key)

    return cfg
