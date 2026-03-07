"""Configuration for eRisk 2026 Task 2 pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class EmbeddingConfig:
    models: list[str] = field(default_factory=lambda: [
        "all-mpnet-base-v2",       # 768d
        "all-MiniLM-L12-v2",      # 384d
        "all-distilroberta-v1",    # 768d
    ])
    total_dim: int = 1920
    decay_lambda: float = 0.95
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class SymptomConfig:
    variant: str = "C"  # A, B, C, or D
    use_depresym_embeddings: bool = True
    activation_threshold: float = 0.3


@dataclass
class BERTopicConfig:
    n_topics: int = 40
    n_neighbors: int = 15
    n_components: int = 5
    min_cluster_size: int = 50
    min_samples: int = 10
    rolling_buffer_size: int = 5


@dataclass
class EmotionConfig:
    model: str = "j-hartmann/emotion-english-distilroberta-base"
    min_words: int = 10  # below this, use neutral distribution


@dataclass
class WassersteinConfig:
    short_window: int = 5
    medium_window: int = 25
    n_projections: int = 50  # for sliced Wasserstein


@dataclass
class MahalanobisConfig:
    n_pca_components: int = 50
    regularization: str = "ledoit_wolf"


@dataclass
class ToMConfig:
    enabled: bool = True
    method: str = "option_c"  # option_a, option_b, option_c
    chained: bool = False  # True = Prompt 2b (chained), False = Prompt 2a (independent)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.3:70b"
    num_ctx: int = 8192
    keep_alive: str = "24h"
    temperature: float = 0.1
    timeout_seconds: int = 120
    retry_attempts: int = 3


@dataclass
class ServerConfig:
    base_url: str = "https://erisk.irlab.org/challenge-t2"
    team_token: str = ""
    max_retries: int = 5
    initial_delay: float = 2.0
    backoff_factor: float = 2.0
    timeout: int = 60


@dataclass
class LoggingConfig:
    output_dir: str = "./runs/task2"
    log_level: str = "INFO"


@dataclass
class ThreadFormatConfig:
    max_tokens: int = 2000
    priority1_full: bool = True  # target posts + direct replies always full
    truncate_length: int = 100  # chars for Priority 3


@dataclass
class Task2Config:
    # Data paths
    training_data_dir: str = ""
    labels_path: str = ""

    # Components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    symptom: SymptomConfig = field(default_factory=SymptomConfig)
    bertopic: BERTopicConfig = field(default_factory=BERTopicConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    wasserstein: WassersteinConfig = field(default_factory=WassersteinConfig)
    mahalanobis: MahalanobisConfig = field(default_factory=MahalanobisConfig)
    tom: ToMConfig = field(default_factory=ToMConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    thread_format: ThreadFormatConfig = field(default_factory=ThreadFormatConfig)

    # Classifier training
    cv_folds: int = 5
    xgboost_params: dict = field(default_factory=lambda: {
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.1,
    })


def load_config(config_path: str | Path = "config/task2.yaml") -> Task2Config:
    load_dotenv()
    cfg = Task2Config()

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        _apply_dict(cfg, raw)

    # Environment overrides
    cfg.ollama.base_url = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)
    cfg.server.team_token = os.getenv("ERISK_TOKEN", cfg.server.team_token)

    return cfg


def _apply_dict(obj, d: dict) -> None:
    """Recursively apply dict values to dataclass fields."""
    for key, val in d.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
            _apply_dict(current, val)
        else:
            setattr(obj, key, val)
