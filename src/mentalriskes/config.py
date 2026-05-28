"""Configuration loading for MentalRiskES Task 1."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    base_url: str = ""
    token: str = ""
    user: str = ""
    use_trial: bool = True
    retries: int = 5
    backoff: float = 0.1


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = ""
    api_key: str = ""
    model: str = "llama3.3:70b"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 180


@dataclass
class DataConfig:
    trial_dir: Path = field(default_factory=lambda: Path("data/MentalRiskES-2026/task1_trial/data"))
    primate_path: Path = field(default_factory=lambda: Path("data/MentalRiskES-2026/primate_dataset.json"))
    output_dir: Path = field(default_factory=lambda: Path("output/mentalriskes"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("output/mentalriskes/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("output/mentalriskes/logs"))


@dataclass
class ResourcesConfig:
    calibration_config: Path = field(default_factory=lambda: Path("specs/MentalRiskES/calibration_config.json"))
    act_vocabulary: Path = field(default_factory=lambda: Path("specs/MentalRiskES/act_vocabulary_es.json"))
    hexaflex_quotes: Path = field(default_factory=lambda: Path("specs/MentalRiskES/hexaflex_quotes_fallback.json"))


@dataclass
class RunConfig:
    name: str = "run0"
    description: str = ""
    model: str = "llama3.3:70b"
    prompt_language: str = "english"
    few_shot: bool = True
    calibration: str = "flat"  # flat | band_aware | none (simple per-item strategy)
    calibration_params: dict = field(default_factory=dict)

    # Three-tier calibration (spec: mentalriskes2026_constraints_ablation_spec.md)
    # A1 = prompt_anchors=True, level_b=False, level_c=False
    # A3 = prompt_anchors=True, level_b=True,  level_c=False  (Run 1 — safety hedge)
    # A5 = prompt_anchors=True, level_b=True,  level_c=True   (Run 0 — best RMSE)
    prompt_anchors: bool = False   # Level A: inject psychometric anchors into assessor prompts
    level_b: bool = False          # Level B: apply 7-rule rule-based constraint system
    level_c: bool = False          # Level C: LLM calibration agent (conditional on violations)

    # Temporal aggregation (spec: mentalriskes2026_wasserstein_temporal_spec.md)
    # Methods: T0=last-round, T1=uniform median, T2=early-weighted, T3=stability-adaptive
    temporal_phq9: str = "T0"      # aggregation method for PHQ-9
    temporal_gad7: str = "T0"      # aggregation method for GAD-7
    temporal_compact10: str = "T0"  # aggregation method for CompACT-10
    temporal_decay: str = "step"    # decay type for T2 (step/inverse/linear)
    temporal_stability_threshold: float = 0.5  # std threshold for T3
    temporal_w1_threshold: float = 2.0  # anomaly detection threshold factor
    temporal_discard_anomalous: bool = True  # discard anomalous rounds for PHQ-9/GAD-7


@dataclass
class PipelineSettings:
    log_level: str = "INFO"
    log_llm_outputs: bool = True
    parallel_assessment: bool = False
    codecarbon_enabled: bool = True
    codecarbon_country: str = "FRA"


@dataclass
class MentalRiskESConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    simulation_llm: LLMConfig = field(default_factory=LLMConfig)
    runs: list[RunConfig] = field(default_factory=list)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)


def _resolve_env(raw: dict, key: str, env_key: str | None = None) -> str:
    """Resolve a value from dict, falling back to env var."""
    if env_key and env_key in raw:
        return os.environ.get(raw[env_key], "")
    return raw.get(key, "")


def load_config(config_path: str | Path) -> MentalRiskESConfig:
    """Load configuration from YAML file, resolving env vars."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = MentalRiskESConfig()

    # Server
    srv = raw.get("server", {})
    cfg.server = ServerConfig(
        base_url=os.environ.get(srv.get("base_url_env", ""), ""),
        token=os.environ.get(srv.get("token_env", ""), ""),
        user=os.environ.get(srv.get("user_env", ""), ""),
        use_trial=srv.get("use_trial", True),
        retries=srv.get("retries", 5),
        backoff=srv.get("backoff", 0.1),
    )

    # LLM
    llm = raw.get("llm", {})
    provider = llm.get("provider", "ollama")
    # Resolve API key from env var (provider-aware fallbacks)
    api_key_env = llm.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "")
    if not api_key and provider == "huggingface":
        api_key = os.environ.get("HF_TOKEN", "")
    if not api_key and provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key and provider == "deepinfra":
        api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key and provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    # Resolve base URL (provider-aware fallback for Together / DeepInfra / OpenRouter)
    base_url = os.environ.get(llm.get("base_url_env", ""), "")
    if not base_url and provider == "together":
        base_url = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    if not base_url and provider == "deepinfra":
        base_url = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
    if not base_url and provider == "openrouter":
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    cfg.llm = LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=llm.get("model", "llama3.3:70b"),
        temperature=llm.get("temperature", 0.1),
        max_tokens=llm.get("max_tokens", 4096),
        timeout=llm.get("timeout", 180),
    )

    # Data
    data = raw.get("data", {})
    cfg.data = DataConfig(
        trial_dir=Path(data.get("trial_dir", "data/MentalRiskES-2026/task1_trial/data")),
        primate_path=Path(data.get("primate_path", "data/MentalRiskES-2026/primate_dataset.json")),
        output_dir=Path(data.get("output_dir", "output/mentalriskes")),
        checkpoint_dir=Path(data.get("checkpoint_dir", "output/mentalriskes/checkpoints")),
        log_dir=Path(data.get("log_dir", "output/mentalriskes/logs")),
    )

    # Simulation LLM (for data_prep — defaults to main LLM if not specified)
    sim = raw.get("simulation_llm", {})
    if sim:
        sim_provider = sim.get("provider", "openai")
        sim_api_key_env = sim.get("api_key_env", "")
        sim_api_key = os.environ.get(sim_api_key_env, "")
        cfg.simulation_llm = LLMConfig(
            provider=sim_provider,
            base_url=os.environ.get(sim.get("base_url_env", ""), ""),
            api_key=sim_api_key,
            model=sim.get("model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            temperature=sim.get("temperature", 0.7),
            max_tokens=sim.get("max_tokens", 512),
            timeout=sim.get("timeout", 60),
        )
    else:
        # Fall back to main LLM config
        cfg.simulation_llm = cfg.llm

    # Resources
    res = raw.get("resources", {})
    cfg.resources = ResourcesConfig(
        calibration_config=Path(res.get("calibration_config", "specs/MentalRiskES/calibration_config.json")),
        act_vocabulary=Path(res.get("act_vocabulary", "specs/MentalRiskES/act_vocabulary_es.json")),
        hexaflex_quotes=Path(res.get("hexaflex_quotes", "specs/MentalRiskES/hexaflex_quotes_fallback.json")),
    )

    # Runs
    for run_raw in raw.get("runs", []):
        cfg.runs.append(RunConfig(
            name=run_raw.get("name", "run0"),
            description=run_raw.get("description", ""),
            model=run_raw.get("model", cfg.llm.model),
            prompt_language=run_raw.get("prompt_language", "english"),
            few_shot=run_raw.get("few_shot", True),
            calibration=run_raw.get("calibration", "none"),
            calibration_params=run_raw.get("calibration_params", {}),
            prompt_anchors=run_raw.get("prompt_anchors", False),
            level_b=run_raw.get("level_b", False),
            level_c=run_raw.get("level_c", False),
            temporal_phq9=run_raw.get("temporal_phq9", "T0"),
            temporal_gad7=run_raw.get("temporal_gad7", "T0"),
            temporal_compact10=run_raw.get("temporal_compact10", "T0"),
            temporal_decay=run_raw.get("temporal_decay", "step"),
            temporal_stability_threshold=run_raw.get("temporal_stability_threshold", 0.5),
            temporal_w1_threshold=run_raw.get("temporal_w1_threshold", 2.0),
            temporal_discard_anomalous=run_raw.get("temporal_discard_anomalous", True),
        ))

    # Pipeline
    pipe = raw.get("pipeline", {})
    cc = pipe.get("codecarbon", {})
    cfg.pipeline = PipelineSettings(
        log_level=pipe.get("log_level", "INFO"),
        log_llm_outputs=pipe.get("log_llm_outputs", True),
        parallel_assessment=pipe.get("parallel_assessment", False),
        codecarbon_enabled=cc.get("enabled", True),
        codecarbon_country=cc.get("country_iso_code", "FRA"),
    )

    return cfg
