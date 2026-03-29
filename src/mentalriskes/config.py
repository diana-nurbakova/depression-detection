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
    calibration: str = "flat"  # flat | band_aware | none
    calibration_params: dict = field(default_factory=dict)


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
    cfg.llm = LLMConfig(
        provider=llm.get("provider", "ollama"),
        base_url=os.environ.get(llm.get("base_url_env", ""), ""),
        api_key=os.environ.get(llm.get("api_key_env", ""), ""),
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
