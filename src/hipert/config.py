"""Configuration loader for the HiPerT-ADHD pipeline.

Loads YAML config files and resolves environment variables from .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from hipert.models import SymptomDefinition, SymptomFactor, SymptomSubcluster

# Map string names to enums
_FACTOR_MAP = {
    "Inattention": SymptomFactor.INATTENTION,
    "Motor_HI": SymptomFactor.MOTOR_HI,
    "Verbal_HI": SymptomFactor.VERBAL_HI,
}

_SUBCLUSTER_MAP = {
    "Organization/Planning": SymptomSubcluster.ORGANIZATION_PLANNING,
    "Memory/Avoidance": SymptomSubcluster.MEMORY_AVOIDANCE,
    "Sustained_Attention/Distractibility": SymptomSubcluster.SUSTAINED_ATTENTION,
    "Sustained_Attention": SymptomSubcluster.SUSTAINED_ATTENTION,
    "Fidgeting/Restlessness": SymptomSubcluster.FIDGETING_RESTLESSNESS,
    "Fidgeting_Restlessness": SymptomSubcluster.FIDGETING_RESTLESSNESS,
    "Internal_Drive/Settling": SymptomSubcluster.INTERNAL_DRIVE,
    "Internal_Drive": SymptomSubcluster.INTERNAL_DRIVE,
    "Output_Control": SymptomSubcluster.OUTPUT_CONTROL,
    "Turn-Taking/Interrupting": SymptomSubcluster.TURN_TAKING,
    "Turn_Taking": SymptomSubcluster.TURN_TAKING,
}


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Data paths
    project_root: Path
    corpus_dir: Path
    output_dir: Path
    checkpoint_dir: Path

    # Retrieval
    retrieval_models: list[str]
    retrieval_top_k: int
    keyword_boost: float
    first_person_filter: bool

    # LLM providers
    primary_provider: LLMProviderConfig
    escalation_provider: LLMProviderConfig
    llm_max_retries: int
    llm_rate_limit_delay: float
    llm_read_timeout: int

    # Scoring
    escalation_confidence_threshold: int
    escalation_max_rate: float

    # Processing
    batch_size: int
    num_workers: int

    # Logging
    log_level: str
    log_dir: Path

    # External datasets (annotation protocol v3)
    redsm5_dir: Path | None = None
    erisk2023_dir: Path | None = None
    erisk2023_trec_dir: Path | None = None
    bdisen_dir: Path | None = None
    erisk2025_dir: Path | None = None
    erisk2025_trec_dir: Path | None = None
    candidates_dir: Path | None = None

    # Symptom definitions
    symptoms: list[SymptomDefinition] = field(default_factory=list)

    # Hierarchy metadata
    hierarchy: dict = field(default_factory=dict)
    keyword_clusters: dict[str, list[str]] = field(default_factory=dict)


def _resolve_provider(
    raw: dict, name: str,
) -> LLMProviderConfig:
    """Build an LLMProviderConfig, resolving env vars for URL and key."""
    if "base_url_env" in raw:
        base_url = os.getenv(raw["base_url_env"], "")
    else:
        base_url = raw.get("base_url", "")

    api_key_env = raw.get("api_key_env", "")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""

    return LLMProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=raw.get("model", ""),
        temperature=raw.get("temperature", 0.3),
        max_tokens=raw.get("max_tokens", 512),
    )


def _parse_symptom(raw: dict) -> SymptomDefinition:
    """Parse a single symptom definition from YAML."""
    layers = raw.get("layers", {})
    return SymptomDefinition(
        item_number=raw["item_number"],
        text=raw["text"],
        factor=_FACTOR_MAP[raw["factor"]],
        subcluster=_SUBCLUSTER_MAP[raw["subcluster"]],
        clinical_definition=layers.get("L1_clinical", "").strip(),
        adult_manifestation=layers.get("L2_adult", "").strip(),
        discussion_topics=layers.get("L3_discussion", "").strip(),
        differential_markers=layers.get("L4_differential", "").strip(),
        token_budget=raw.get("token_budget", "compressed_3"),
        keywords=raw.get("keywords", []),
        expected_reliability=raw.get("expected_reliability", "MEDIUM"),
        symptom_weight=raw.get("symptom_weight", 1.0),
        clinical_difficulty_prior=raw.get("clinical_difficulty_prior", 0.5),
    )


def load_config(
    pipeline_path: str | Path = "config/pipeline.yaml",
    symptoms_path: str | Path = "config/symptoms.yaml",
    project_root: str | Path | None = None,
) -> PipelineConfig:
    """Load pipeline and symptom configurations.

    Args:
        pipeline_path: Path to the pipeline YAML config.
        symptoms_path: Path to the symptoms YAML config.
        project_root: Project root directory. If None, inferred from
            pipeline_path location.
    """
    pipeline_path = Path(pipeline_path)
    symptoms_path = Path(symptoms_path)

    if project_root is None:
        project_root = pipeline_path.parent.parent
    project_root = Path(project_root)

    # Load .env from project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load pipeline YAML
    with open(pipeline_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    data_cfg = raw.get("data", {})
    retrieval_cfg = raw.get("retrieval", {})
    llm_cfg = raw.get("llm", {})
    scoring_cfg = raw.get("scoring", {})
    processing_cfg = raw.get("processing", {})
    logging_cfg = raw.get("logging", {})

    # Resolve providers
    providers_raw = llm_cfg.get("providers", {})
    primary_name = scoring_cfg.get("primary_provider", "ollama")
    escalation_name = scoring_cfg.get("escalation_provider", "openai")

    primary_provider = _resolve_provider(
        providers_raw.get(primary_name, {}), primary_name,
    )
    escalation_provider = _resolve_provider(
        providers_raw.get(escalation_name, {}), escalation_name,
    )

    # Resolve paths relative to project root
    corpus_dir = project_root / data_cfg.get("corpus_dir", "data")
    output_dir = project_root / data_cfg.get("output_dir", "output")
    checkpoint_dir = project_root / data_cfg.get("checkpoint_dir", "output/checkpoints")
    log_dir = project_root / logging_cfg.get("dir", "output/logs")

    # External dataset paths (optional)
    redsm5_dir = (
        project_root / data_cfg["redsm5_dir"]
        if "redsm5_dir" in data_cfg else None
    )
    erisk2023_dir = (
        project_root / data_cfg["erisk2023_dir"]
        if "erisk2023_dir" in data_cfg else None
    )
    erisk2023_trec_dir = (
        project_root / data_cfg["erisk2023_trec_dir"]
        if "erisk2023_trec_dir" in data_cfg else None
    )
    bdisen_dir = (
        project_root / data_cfg["bdisen_dir"]
        if "bdisen_dir" in data_cfg else None
    )
    erisk2025_dir = (
        project_root / data_cfg["erisk2025_dir"]
        if "erisk2025_dir" in data_cfg else None
    )
    erisk2025_trec_dir = (
        project_root / data_cfg["erisk2025_trec_dir"]
        if "erisk2025_trec_dir" in data_cfg else None
    )
    candidates_dir = (
        project_root / data_cfg["candidates_dir"]
        if "candidates_dir" in data_cfg else None
    )

    # Load symptoms YAML
    symptoms: list[SymptomDefinition] = []
    hierarchy: dict = {}
    keyword_clusters: dict[str, list[str]] = {}

    if symptoms_path.exists():
        with open(symptoms_path, "r", encoding="utf-8") as f:
            symptoms_raw: dict = yaml.safe_load(f)

        for s in symptoms_raw.get("symptoms", []):
            symptoms.append(_parse_symptom(s))

        hierarchy = symptoms_raw.get("hierarchy", {})
        keyword_clusters = symptoms_raw.get("keyword_clusters", {})

    return PipelineConfig(
        project_root=project_root,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        redsm5_dir=redsm5_dir,
        erisk2023_dir=erisk2023_dir,
        erisk2023_trec_dir=erisk2023_trec_dir,
        bdisen_dir=bdisen_dir,
        erisk2025_dir=erisk2025_dir,
        erisk2025_trec_dir=erisk2025_trec_dir,
        candidates_dir=candidates_dir,
        retrieval_models=retrieval_cfg.get("models", ["all-mpnet-base-v2"]),
        retrieval_top_k=retrieval_cfg.get("top_k", 5000),
        keyword_boost=retrieval_cfg.get("keyword_boost", 0.05),
        first_person_filter=retrieval_cfg.get("first_person_filter", True),
        primary_provider=primary_provider,
        escalation_provider=escalation_provider,
        llm_max_retries=llm_cfg.get("max_retries", 5),
        llm_rate_limit_delay=llm_cfg.get("rate_limit_delay", 0.5),
        llm_read_timeout=llm_cfg.get("read_timeout", 300),
        escalation_confidence_threshold=scoring_cfg.get(
            "escalation_confidence_threshold", 2,
        ),
        escalation_max_rate=scoring_cfg.get("escalation_max_rate", 0.40),
        batch_size=processing_cfg.get("batch_size", 50),
        num_workers=processing_cfg.get("num_workers", 4),
        log_level=logging_cfg.get("level", "INFO"),
        log_dir=log_dir,
        symptoms=symptoms,
        hierarchy=hierarchy,
        keyword_clusters=keyword_clusters,
    )
