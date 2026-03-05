#!/usr/bin/env python3
"""Extract confounder and boundary candidates from external datasets.

Implements Steps A–D of annotation_protocol_spec_v3.md Section 3:
  Step A: Extract RedSM5 confounder candidates (depression sentences that
          mimic ADHD symptoms) for score-1 few-shot examples.
  Step B: Extract BDI-Sen graded confounder candidates (severity-graded
          depression sentences) for calibrated score-1 examples.
  Step C: Extract eRisk 2025 T1 boundary candidates (majority-vs-consensus
          disagreement sentences) for score-1 boundary examples.
  Step D: Extract eRisk 2023 boundary candidates (legacy, same logic as C).

Produces pool TSV files consumed by merge_candidates.py (Step F).

Usage:
    uv run python scripts/extract_external_candidates.py
    uv run python scripts/extract_external_candidates.py --symptoms 7,8,9,10,11
    uv run python scripts/extract_external_candidates.py --redsm5-only
    uv run python scripts/extract_external_candidates.py --bdisen-only
    uv run python scripts/extract_external_candidates.py --erisk2025-only
    uv run python scripts/extract_external_candidates.py --erisk2023-only
    uv run python scripts/extract_external_candidates.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hipert.config import load_config
from hipert.data.cross_dataset_mappings import (
    ASRS_TO_BDIII,
    ASRS_TO_BDISEN,
    ASRS_TO_REDSM5,
    BDIII_QUERY_NAMES,
    BDIII_TO_ASRS,
    BDISEN_SYMPTOM_DESCRIPTIONS,
    BDISEN_TO_ASRS,
    REDSM5_CATEGORY_DESCRIPTIONS,
    REDSM5_TO_ASRS,
)
from hipert.data.bdisen_loader import (
    filter_confounder_candidates as bdisen_filter_confounders,
    load_annotations as bdisen_load_annotations,
)
from hipert.data.erisk2023_loader import (
    build_sentence_lookup as erisk2023_build_lookup,
    extract_boundary_candidates_cached as erisk2023_extract_boundary,
    load_qrels_by_query as erisk2023_load_qrels_by_query,
    resolve_sentences as erisk2023_resolve_sentences,
)
from hipert.data.erisk2025_loader import (
    build_sentence_lookup as erisk2025_build_lookup,
    build_user_index as erisk2025_build_user_index,
    extract_boundary_candidates_cached as erisk2025_extract_boundary,
    load_qrels_by_query as erisk2025_load_qrels_by_query,
    resolve_sentences as erisk2025_resolve_sentences,
)
from hipert.data.redsm5_loader import (
    filter_confounder_candidates as redsm5_filter_confounders,
    load_annotations as redsm5_load_annotations,
)
from hipert.models import SymptomDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL event logger for this script
# ---------------------------------------------------------------------------


class ExtractionEventLogger:
    """JSONL event logger for the extraction pipeline."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.filepath = log_dir / f"extract_external_{run_id}.jsonl"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")

    def log(self, event_type: str, **kwargs) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            **kwargs,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Intermediate result saving/loading
# ---------------------------------------------------------------------------


def _save_intermediate(filepath: Path, data: object) -> None:
    """Save intermediate result as JSON for resumability."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.debug("Saved intermediate: %s", filepath)


def _load_intermediate(filepath: Path) -> object | None:
    """Load intermediate result if it exists."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Step A: RedSM5 confounder extraction
# ---------------------------------------------------------------------------


def extract_redsm5_confounders(
    redsm5_dir: Path,
    symptoms: dict[int, SymptomDefinition],
    symptom_ids: list[int] | None = None,
    intermediate_dir: Path | None = None,
    event_logger: ExtractionEventLogger | None = None,
    resume: bool = False,
) -> dict[int, list[dict]]:
    """Step A: Extract RedSM5 confounder candidates per ASRS symptom.

    For each symptom with a RedSM5 mapping:
    1. Look up mapped DSM-5 categories from ASRS_TO_REDSM5
    2. Load annotations, filter to status=1 + first-person
    3. Return candidates with clinical explanation attached

    Args:
        redsm5_dir: Path to data/RedSM5/ directory.
        symptoms: Dict of symptom_id -> SymptomDefinition.
        symptom_ids: Subset of symptoms to process (default: all with mapping).
        intermediate_dir: Directory for intermediate JSON checkpoints.
        event_logger: JSONL event logger.
        resume: If True, skip steps with existing intermediate files.

    Returns:
        Dict mapping symptom_id to list of candidate dicts.
    """
    annotations_path = redsm5_dir / "redsm5_annotations.csv"

    # Load all annotations
    logger.info("Loading RedSM5 annotations from %s", annotations_path)
    all_annotations = redsm5_load_annotations(annotations_path)

    if event_logger:
        # Count per category
        category_counts: dict[str, dict[str, int]] = {}
        for ann in all_annotations:
            if ann.dsm5_symptom not in category_counts:
                category_counts[ann.dsm5_symptom] = {"total": 0, "status_1": 0}
            category_counts[ann.dsm5_symptom]["total"] += 1
            if ann.status == 1:
                category_counts[ann.dsm5_symptom]["status_1"] += 1

        event_logger.log(
            "redsm5_load",
            total_annotations=len(all_annotations),
            category_counts=category_counts,
        )

    # Save raw annotation summary
    if intermediate_dir:
        summary = {
            "total": len(all_annotations),
            "categories": {},
        }
        for ann in all_annotations:
            cat = ann.dsm5_symptom
            if cat not in summary["categories"]:
                summary["categories"][cat] = {"total": 0, "status_1": 0, "first_person_and_status_1": 0}
            summary["categories"][cat]["total"] += 1
            if ann.status == 1:
                summary["categories"][cat]["status_1"] += 1
                if ann.has_first_person:
                    summary["categories"][cat]["first_person_and_status_1"] += 1
        _save_intermediate(intermediate_dir / "redsm5_annotation_summary.json", summary)

    # Determine which symptoms to process
    if symptom_ids is None:
        symptom_ids = sorted(ASRS_TO_REDSM5.keys())
    else:
        symptom_ids = [s for s in symptom_ids if s in ASRS_TO_REDSM5]

    logger.info(
        "Extracting RedSM5 confounders for %d symptoms: %s",
        len(symptom_ids), symptom_ids,
    )

    results: dict[int, list[dict]] = {}

    for symptom_id in symptom_ids:
        # Check for existing intermediate
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"redsm5_filtered_{symptom_id:02d}.json",
            )
            if cached is not None:
                results[symptom_id] = cached
                logger.info("Symptom %02d: loaded %d cached RedSM5 candidates", symptom_id, len(cached))
                continue

        dsm5_categories = ASRS_TO_REDSM5[symptom_id]
        symptom_candidates: list[dict] = []

        for dsm5_cat in dsm5_categories:
            filtered = redsm5_filter_confounders(
                all_annotations, dsm5_cat, require_first_person=True,
            )

            for ann in filtered:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "sentence_id": ann.sentence_id,
                    "sentence_text": ann.sentence_text,
                    "dsm5_symptom": ann.dsm5_symptom,
                    "explanation": ann.explanation,
                    "source": "redsm5",
                })

        results[symptom_id] = symptom_candidates

        logger.info(
            "Symptom %02d: %d RedSM5 confounder candidates (from %s)",
            symptom_id, len(symptom_candidates),
            ", ".join(dsm5_categories),
        )

        if event_logger:
            event_logger.log(
                "redsm5_symptom_candidates",
                symptom_id=symptom_id,
                dsm5_categories=dsm5_categories,
                candidate_count=len(symptom_candidates),
                sample_texts=[c["sentence_text"][:100] for c in symptom_candidates[:3]],
            )

        # Save per-symptom intermediate
        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"redsm5_filtered_{symptom_id:02d}.json",
                symptom_candidates,
            )

    return results


# ---------------------------------------------------------------------------
# Step B: eRisk 2023 boundary extraction
# ---------------------------------------------------------------------------


def extract_erisk2023_boundary(
    erisk2023_dir: Path,
    trec_dir: Path,
    symptoms: dict[int, SymptomDefinition],
    symptom_ids: list[int] | None = None,
    intermediate_dir: Path | None = None,
    event_logger: ExtractionEventLogger | None = None,
    resume: bool = False,
) -> dict[int, list[dict]]:
    """Step B: Extract eRisk 2023 boundary candidates per ASRS symptom.

    For each symptom with a BDI-II mapping:
    1. Load majority and consensus qrels
    2. Find disagreement set (majority=1, consensus=0)
    3. Find agreement set (majority=1, consensus=1)
    4. Resolve sentence text from TREC files
    5. Apply first-person filter

    Args:
        erisk2023_dir: Path to data/eRisk2023_T1/ directory.
        trec_dir: Path to data/eRisk2023_T1/new_data/ directory.
        symptoms: Dict of symptom_id -> SymptomDefinition.
        symptom_ids: Subset of symptoms to process (default: all with mapping).
        intermediate_dir: Directory for intermediate JSON checkpoints.
        event_logger: JSONL event logger.
        resume: If True, skip steps with existing intermediate files.

    Returns:
        Dict mapping symptom_id to list of candidate dicts.
    """
    majority_path = erisk2023_dir / "g_qrels_majority_2.csv"
    consenso_path = erisk2023_dir / "g_rels_consenso.csv"

    # Load qrels once (used across all queries)
    logger.info("Loading eRisk 2023 qrels...")
    majority_by_query = erisk2023_load_qrels_by_query(majority_path)
    consenso_by_query = erisk2023_load_qrels_by_query(consenso_path)

    if event_logger:
        maj_total = sum(len(v) for v in majority_by_query.values())
        con_total = sum(len(v) for v in consenso_by_query.values())
        maj_relevant = sum(
            sum(1 for q in v if q.rel == 1)
            for v in majority_by_query.values()
        )
        con_relevant = sum(
            sum(1 for q in v if q.rel == 1)
            for v in consenso_by_query.values()
        )
        event_logger.log(
            "erisk2023_qrels_load",
            majority_total=maj_total,
            majority_relevant=maj_relevant,
            consensus_total=con_total,
            consensus_relevant=con_relevant,
            query_count=len(majority_by_query),
        )

    # Save qrel summary
    if intermediate_dir:
        qrel_summary = {
            "majority": {
                "total": sum(len(v) for v in majority_by_query.values()),
                "relevant": sum(
                    sum(1 for q in v if q.rel == 1)
                    for v in majority_by_query.values()
                ),
                "queries": len(majority_by_query),
            },
            "consensus": {
                "total": sum(len(v) for v in consenso_by_query.values()),
                "relevant": sum(
                    sum(1 for q in v if q.rel == 1)
                    for v in consenso_by_query.values()
                ),
                "queries": len(consenso_by_query),
            },
        }
        _save_intermediate(intermediate_dir / "erisk2023_qrels_summary.json", qrel_summary)

    # Determine which BDI-II queries we need
    if symptom_ids is None:
        symptom_ids = sorted(ASRS_TO_BDIII.keys())
    else:
        symptom_ids = [s for s in symptom_ids if s in ASRS_TO_BDIII]

    needed_bdi_queries = set()
    for sid in symptom_ids:
        needed_bdi_queries.update(ASRS_TO_BDIII[sid])

    logger.info(
        "Processing %d BDI-II queries: %s (for %d ASRS symptoms)",
        len(needed_bdi_queries), sorted(needed_bdi_queries), len(symptom_ids),
    )

    # Extract boundary sets per BDI-II query
    boundary_sets: dict[int, dict] = {}

    for bdi_query in sorted(needed_bdi_queries):
        # Check cache
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"erisk2023_boundary_{bdi_query:02d}.json",
            )
            if cached is not None:
                boundary_sets[bdi_query] = cached
                logger.info(
                    "BDI-II query %d (%s): loaded cached boundary set",
                    bdi_query, BDIII_QUERY_NAMES.get(bdi_query, "?"),
                )
                continue

        disagreement, agreement = erisk2023_extract_boundary(
            majority_by_query, consenso_by_query, bdi_query,
        )

        boundary_sets[bdi_query] = {
            "bdi_query": bdi_query,
            "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "Unknown"),
            "disagreement_docids": disagreement,
            "agreement_docids": agreement,
        }

        if event_logger:
            event_logger.log(
                "erisk2023_boundary_set",
                bdi_query=bdi_query,
                bdi_name=BDIII_QUERY_NAMES.get(bdi_query, "?"),
                disagreement_count=len(disagreement),
                agreement_count=len(agreement),
            )

        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"erisk2023_boundary_{bdi_query:02d}.json",
                boundary_sets[bdi_query],
            )

    # Collect all docids we need to resolve
    all_docids: set[str] = set()
    for bs in boundary_sets.values():
        all_docids.update(bs["disagreement_docids"])
        all_docids.update(bs["agreement_docids"])

    logger.info("Resolving %d unique docids from TREC files...", len(all_docids))

    # Build sentence lookup (only for needed users)
    sentence_lookup = erisk2023_build_lookup(trec_dir, docids=all_docids)

    # Map to ASRS symptoms
    results: dict[int, list[dict]] = {}

    for symptom_id in symptom_ids:
        # Check cache
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"erisk2023_resolved_{symptom_id:02d}.json",
            )
            if cached is not None:
                results[symptom_id] = cached
                logger.info("Symptom %02d: loaded %d cached eRisk2023 candidates", symptom_id, len(cached))
                continue

        bdi_queries = ASRS_TO_BDIII[symptom_id]
        symptom_candidates: list[dict] = []

        for bdi_query in bdi_queries:
            bs = boundary_sets.get(bdi_query, {})

            # Disagreement candidates (score-1 material)
            disagree_docids = bs.get("disagreement_docids", [])
            disagree_sents = erisk2023_resolve_sentences(
                disagree_docids, sentence_lookup, require_first_person=True,
            )
            for sent in disagree_sents:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "docid": sent.docno,
                    "text": sent.text,
                    "bdi_query": bdi_query,
                    "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "?"),
                    "majority": 1,
                    "consensus": 0,
                    "type": "disagreement",
                    "source": "erisk2023",
                })

            # Agreement candidates (score-2 material)
            agree_docids = bs.get("agreement_docids", [])
            agree_sents = erisk2023_resolve_sentences(
                agree_docids, sentence_lookup, require_first_person=True,
            )
            for sent in agree_sents:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "docid": sent.docno,
                    "text": sent.text,
                    "bdi_query": bdi_query,
                    "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "?"),
                    "majority": 1,
                    "consensus": 1,
                    "type": "agreement",
                    "source": "erisk2023",
                })

        results[symptom_id] = symptom_candidates

        n_disagree = sum(1 for c in symptom_candidates if c["type"] == "disagreement")
        n_agree = sum(1 for c in symptom_candidates if c["type"] == "agreement")
        logger.info(
            "Symptom %02d: %d eRisk2023 candidates (%d disagreement, %d agreement) from BDI queries %s",
            symptom_id, len(symptom_candidates), n_disagree, n_agree, bdi_queries,
        )

        if event_logger:
            event_logger.log(
                "erisk2023_symptom_candidates",
                symptom_id=symptom_id,
                bdi_queries=bdi_queries,
                total=len(symptom_candidates),
                disagreement=n_disagree,
                agreement=n_agree,
                sample_texts=[c["text"][:100] for c in symptom_candidates[:3]],
            )

        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"erisk2023_resolved_{symptom_id:02d}.json",
                symptom_candidates,
            )

    return results


# ---------------------------------------------------------------------------
# Step B: BDI-Sen graded confounder extraction (NEW in v3)
# ---------------------------------------------------------------------------


def extract_bdisen_confounders(
    bdisen_dir: Path,
    symptoms: dict[int, SymptomDefinition],
    symptom_ids: list[int] | None = None,
    intermediate_dir: Path | None = None,
    event_logger: ExtractionEventLogger | None = None,
    resume: bool = False,
) -> dict[int, list[dict]]:
    """Step B: Extract BDI-Sen graded confounder candidates per ASRS symptom.

    For each symptom with a BDI-Sen mapping:
    1. Look up mapped BDI-Sen symptom names from ASRS_TO_BDISEN
    2. Load annotations, filter to label=1 + severity in (1, 2, 3) + first-person
    3. Return candidates with severity attached

    Args:
        bdisen_dir: Path to data/BDI-Sen/full_dataset/ directory.
        symptoms: Dict of symptom_id -> SymptomDefinition.
        symptom_ids: Subset of symptoms to process (default: all with mapping).
        intermediate_dir: Directory for intermediate JSON checkpoints.
        event_logger: JSONL event logger.
        resume: If True, skip steps with existing intermediate files.

    Returns:
        Dict mapping symptom_id to list of candidate dicts.
    """
    annotations_path = bdisen_dir / "bdi_majority_vote.jsonl"

    logger.info("Loading BDI-Sen annotations from %s", annotations_path)
    all_annotations = bdisen_load_annotations(annotations_path)

    if event_logger:
        symptom_counts: dict[str, dict[str, int]] = {}
        for ann in all_annotations:
            if ann.symptom not in symptom_counts:
                symptom_counts[ann.symptom] = {"total": 0, "label_1": 0}
            symptom_counts[ann.symptom]["total"] += 1
            if ann.label == 1:
                symptom_counts[ann.symptom]["label_1"] += 1

        event_logger.log(
            "bdisen_load",
            total_annotations=len(all_annotations),
            symptom_counts=symptom_counts,
        )

    if intermediate_dir:
        summary = {
            "total": len(all_annotations),
            "symptoms": {},
        }
        for ann in all_annotations:
            sym = ann.symptom
            if sym not in summary["symptoms"]:
                summary["symptoms"][sym] = {
                    "total": 0, "label_1": 0,
                    "severity_1": 0, "severity_2": 0, "severity_3": 0,
                }
            summary["symptoms"][sym]["total"] += 1
            if ann.label == 1:
                summary["symptoms"][sym]["label_1"] += 1
                if ann.severity == 1:
                    summary["symptoms"][sym]["severity_1"] += 1
                elif ann.severity == 2:
                    summary["symptoms"][sym]["severity_2"] += 1
                elif ann.severity == 3:
                    summary["symptoms"][sym]["severity_3"] += 1
        _save_intermediate(intermediate_dir / "bdisen_annotation_summary.json", summary)

    # Determine which symptoms to process
    if symptom_ids is None:
        symptom_ids = sorted(ASRS_TO_BDISEN.keys())
    else:
        symptom_ids = [s for s in symptom_ids if s in ASRS_TO_BDISEN]

    logger.info(
        "Extracting BDI-Sen confounders for %d symptoms: %s",
        len(symptom_ids), symptom_ids,
    )

    results: dict[int, list[dict]] = {}

    for symptom_id in symptom_ids:
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"bdisen_filtered_{symptom_id:02d}.json",
            )
            if cached is not None:
                results[symptom_id] = cached
                logger.info("Symptom %02d: loaded %d cached BDI-Sen candidates", symptom_id, len(cached))
                continue

        bdisen_symptoms = ASRS_TO_BDISEN[symptom_id]
        symptom_candidates: list[dict] = []

        for bdisen_sym in bdisen_symptoms:
            # Get all positive severities (1, 2, 3) — caller can filter later
            filtered = bdisen_filter_confounders(
                all_annotations, bdisen_sym,
                severity_levels=(1, 2, 3),
                require_first_person=True,
            )

            for ann in filtered:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "sentence_text": ann.sentence,
                    "bdisen_symptom": ann.symptom,
                    "severity": ann.severity,
                    "label": ann.label,
                    "source": "bdisen",
                })

        results[symptom_id] = symptom_candidates

        logger.info(
            "Symptom %02d: %d BDI-Sen confounder candidates (from %s)",
            symptom_id, len(symptom_candidates),
            ", ".join(bdisen_symptoms),
        )

        if event_logger:
            sev_dist = {}
            for c in symptom_candidates:
                s = c["severity"]
                sev_dist[s] = sev_dist.get(s, 0) + 1
            event_logger.log(
                "bdisen_symptom_candidates",
                symptom_id=symptom_id,
                bdisen_symptoms=bdisen_symptoms,
                candidate_count=len(symptom_candidates),
                severity_distribution=sev_dist,
                sample_texts=[c["sentence_text"][:100] for c in symptom_candidates[:3]],
            )

        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"bdisen_filtered_{symptom_id:02d}.json",
                symptom_candidates,
            )

    return results


# ---------------------------------------------------------------------------
# Step C: eRisk 2025 T1 boundary extraction (NEW in v3)
# ---------------------------------------------------------------------------


def extract_erisk2025_boundary(
    erisk2025_dir: Path,
    trec_dir: Path,
    symptoms: dict[int, SymptomDefinition],
    symptom_ids: list[int] | None = None,
    intermediate_dir: Path | None = None,
    event_logger: ExtractionEventLogger | None = None,
    resume: bool = False,
) -> dict[int, list[dict]]:
    """Step C: Extract eRisk 2025 T1 boundary candidates per ASRS symptom.

    For each symptom with a BDI-II mapping:
    1. Load majority and consensus qrels
    2. Find disagreement set (majority=True, consensus=False)
    3. Find agreement set (majority=True, consensus=True)
    4. Resolve sentence text from TREC files (full PRE/TEXT/POST)
    5. Apply first-person filter

    Args:
        erisk2025_dir: Path to eRisk 2025 T1 dataset directory.
        trec_dir: Path to directory containing s_*.trec files.
        symptoms: Dict of symptom_id -> SymptomDefinition.
        symptom_ids: Subset of symptoms to process (default: all with mapping).
        intermediate_dir: Directory for intermediate JSON checkpoints.
        event_logger: JSONL event logger.
        resume: If True, skip steps with existing intermediate files.

    Returns:
        Dict mapping symptom_id to list of candidate dicts.
    """
    majority_path = erisk2025_dir / "qrels_majority_merged.csv"
    consensus_path = erisk2025_dir / "qrels_consensus_merged.csv"

    logger.info("Loading eRisk 2025 T1 qrels...")
    majority_by_query = erisk2025_load_qrels_by_query(majority_path)
    consensus_by_query = erisk2025_load_qrels_by_query(consensus_path)

    if event_logger:
        maj_total = sum(len(v) for v in majority_by_query.values())
        con_total = sum(len(v) for v in consensus_by_query.values())
        maj_relevant = sum(
            sum(1 for q in v if q.relevant)
            for v in majority_by_query.values()
        )
        con_relevant = sum(
            sum(1 for q in v if q.relevant)
            for v in consensus_by_query.values()
        )
        event_logger.log(
            "erisk2025_qrels_load",
            majority_total=maj_total,
            majority_relevant=maj_relevant,
            consensus_total=con_total,
            consensus_relevant=con_relevant,
            query_count=len(majority_by_query),
        )

    if intermediate_dir:
        qrel_summary = {
            "majority": {
                "total": sum(len(v) for v in majority_by_query.values()),
                "relevant": sum(
                    sum(1 for q in v if q.relevant)
                    for v in majority_by_query.values()
                ),
                "queries": len(majority_by_query),
            },
            "consensus": {
                "total": sum(len(v) for v in consensus_by_query.values()),
                "relevant": sum(
                    sum(1 for q in v if q.relevant)
                    for v in consensus_by_query.values()
                ),
                "queries": len(consensus_by_query),
            },
        }
        _save_intermediate(intermediate_dir / "erisk2025_qrels_summary.json", qrel_summary)

    # Determine which BDI-II queries we need
    if symptom_ids is None:
        symptom_ids = sorted(ASRS_TO_BDIII.keys())
    else:
        symptom_ids = [s for s in symptom_ids if s in ASRS_TO_BDIII]

    needed_bdi_queries = set()
    for sid in symptom_ids:
        needed_bdi_queries.update(ASRS_TO_BDIII[sid])

    logger.info(
        "Processing %d BDI-II queries: %s (for %d ASRS symptoms)",
        len(needed_bdi_queries), sorted(needed_bdi_queries), len(symptom_ids),
    )

    # Extract boundary sets per BDI-II query
    boundary_sets: dict[int, dict] = {}

    for bdi_query in sorted(needed_bdi_queries):
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"erisk2025_boundary_{bdi_query:02d}.json",
            )
            if cached is not None:
                boundary_sets[bdi_query] = cached
                logger.info(
                    "BDI-II query %d (%s): loaded cached eRisk2025 boundary set",
                    bdi_query, BDIII_QUERY_NAMES.get(bdi_query, "?"),
                )
                continue

        disagreement, agreement = erisk2025_extract_boundary(
            majority_by_query, consensus_by_query, bdi_query,
        )

        boundary_sets[bdi_query] = {
            "bdi_query": bdi_query,
            "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "Unknown"),
            "disagreement_docids": disagreement,
            "agreement_docids": agreement,
        }

        if event_logger:
            event_logger.log(
                "erisk2025_boundary_set",
                bdi_query=bdi_query,
                bdi_name=BDIII_QUERY_NAMES.get(bdi_query, "?"),
                disagreement_count=len(disagreement),
                agreement_count=len(agreement),
            )

        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"erisk2025_boundary_{bdi_query:02d}.json",
                boundary_sets[bdi_query],
            )

    # Collect all docids we need to resolve
    all_docids: set[str] = set()
    for bs in boundary_sets.values():
        all_docids.update(bs["disagreement_docids"])
        all_docids.update(bs["agreement_docids"])

    logger.info("Resolving %d unique docids from eRisk 2025 TREC files...", len(all_docids))

    # Build user index and sentence lookup (only for needed users)
    user_index = erisk2025_build_user_index(trec_dir)
    sentence_lookup = erisk2025_build_lookup(trec_dir, docids=all_docids, user_index=user_index)

    # Map to ASRS symptoms
    results: dict[int, list[dict]] = {}

    for symptom_id in symptom_ids:
        if resume and intermediate_dir:
            cached = _load_intermediate(
                intermediate_dir / f"erisk2025_resolved_{symptom_id:02d}.json",
            )
            if cached is not None:
                results[symptom_id] = cached
                logger.info("Symptom %02d: loaded %d cached eRisk2025 candidates", symptom_id, len(cached))
                continue

        bdi_queries = ASRS_TO_BDIII[symptom_id]
        symptom_candidates: list[dict] = []

        for bdi_query in bdi_queries:
            bs = boundary_sets.get(bdi_query, {})

            # Disagreement candidates (score-1 material)
            disagree_docids = bs.get("disagreement_docids", [])
            disagree_sents = erisk2025_resolve_sentences(
                disagree_docids, sentence_lookup, require_first_person=True,
            )
            for sent in disagree_sents:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "docid": sent.docno,
                    "pre": sent.pre,
                    "text": sent.text,
                    "post": sent.post,
                    "bdi_query": bdi_query,
                    "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "?"),
                    "majority": True,
                    "consensus": False,
                    "type": "disagreement",
                    "source": "erisk2025",
                })

            # Agreement candidates (score-2 material)
            agree_docids = bs.get("agreement_docids", [])
            agree_sents = erisk2025_resolve_sentences(
                agree_docids, sentence_lookup, require_first_person=True,
            )
            for sent in agree_sents:
                symptom_candidates.append({
                    "symptom_id": symptom_id,
                    "docid": sent.docno,
                    "pre": sent.pre,
                    "text": sent.text,
                    "post": sent.post,
                    "bdi_query": bdi_query,
                    "bdi_name": BDIII_QUERY_NAMES.get(bdi_query, "?"),
                    "majority": True,
                    "consensus": True,
                    "type": "agreement",
                    "source": "erisk2025",
                })

        results[symptom_id] = symptom_candidates

        n_disagree = sum(1 for c in symptom_candidates if c["type"] == "disagreement")
        n_agree = sum(1 for c in symptom_candidates if c["type"] == "agreement")
        logger.info(
            "Symptom %02d: %d eRisk2025 candidates (%d disagreement, %d agreement) from BDI queries %s",
            symptom_id, len(symptom_candidates), n_disagree, n_agree, bdi_queries,
        )

        if event_logger:
            event_logger.log(
                "erisk2025_symptom_candidates",
                symptom_id=symptom_id,
                bdi_queries=bdi_queries,
                total=len(symptom_candidates),
                disagreement=n_disagree,
                agreement=n_agree,
                sample_texts=[c["text"][:100] for c in symptom_candidates[:3]],
            )

        if intermediate_dir:
            _save_intermediate(
                intermediate_dir / f"erisk2025_resolved_{symptom_id:02d}.json",
                symptom_candidates,
            )

    return results


# ---------------------------------------------------------------------------
# Pool TSV output
# ---------------------------------------------------------------------------


REDSM5_POOL_COLUMNS = [
    "symptom_id", "sentence_id", "sentence_text",
    "dsm5_symptom", "explanation", "source",
]

BDISEN_POOL_COLUMNS = [
    "symptom_id", "sentence_text", "bdisen_symptom",
    "severity", "label", "source",
]

ERISK2025_POOL_COLUMNS = [
    "symptom_id", "docid", "pre", "text", "post", "bdi_query", "bdi_name",
    "majority", "consensus", "type", "source",
]

ERISK2023_POOL_COLUMNS = [
    "symptom_id", "docid", "text", "bdi_query", "bdi_name",
    "majority", "consensus", "type", "source",
]


def write_redsm5_pool_tsv(
    filepath: Path,
    all_candidates: dict[int, list[dict]],
) -> int:
    """Write all RedSM5 confounder candidates to a single pool TSV.

    Returns:
        Total number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REDSM5_POOL_COLUMNS, delimiter="\t")
        writer.writeheader()

        for symptom_id in sorted(all_candidates.keys()):
            for cand in all_candidates[symptom_id]:
                writer.writerow({col: cand.get(col, "") for col in REDSM5_POOL_COLUMNS})
                total += 1

    logger.info("Wrote %d RedSM5 confounder candidates to %s", total, filepath)
    return total


def write_bdisen_pool_tsv(
    filepath: Path,
    all_candidates: dict[int, list[dict]],
) -> int:
    """Write all BDI-Sen confounder candidates to a single pool TSV.

    Returns:
        Total number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BDISEN_POOL_COLUMNS, delimiter="\t")
        writer.writeheader()

        for symptom_id in sorted(all_candidates.keys()):
            for cand in all_candidates[symptom_id]:
                writer.writerow({col: cand.get(col, "") for col in BDISEN_POOL_COLUMNS})
                total += 1

    logger.info("Wrote %d BDI-Sen confounder candidates to %s", total, filepath)
    return total


def write_erisk2025_pool_tsv(
    filepath: Path,
    all_candidates: dict[int, list[dict]],
) -> int:
    """Write all eRisk 2025 T1 boundary candidates to a single pool TSV.

    Returns:
        Total number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERISK2025_POOL_COLUMNS, delimiter="\t")
        writer.writeheader()

        for symptom_id in sorted(all_candidates.keys()):
            for cand in all_candidates[symptom_id]:
                writer.writerow({col: cand.get(col, "") for col in ERISK2025_POOL_COLUMNS})
                total += 1

    logger.info("Wrote %d eRisk2025 boundary candidates to %s", total, filepath)
    return total


def write_erisk2023_pool_tsv(
    filepath: Path,
    all_candidates: dict[int, list[dict]],
) -> int:
    """Write all eRisk 2023 boundary candidates to a single pool TSV.

    Returns:
        Total number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERISK2023_POOL_COLUMNS, delimiter="\t")
        writer.writeheader()

        for symptom_id in sorted(all_candidates.keys()):
            for cand in all_candidates[symptom_id]:
                writer.writerow({col: cand.get(col, "") for col in ERISK2023_POOL_COLUMNS})
                total += 1

    logger.info("Wrote %d eRisk2023 boundary candidates to %s", total, filepath)
    return total


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract confounder and boundary candidates from external datasets.",
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Path to pipeline config YAML.",
    )
    parser.add_argument(
        "--symptoms-config", default="config/symptoms.yaml",
        help="Path to symptoms config YAML.",
    )
    parser.add_argument(
        "--symptoms", type=str, default=None,
        help="Comma-separated symptom IDs (default: all with external data).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="candidates",
        help="Output directory for pool TSV files (default: candidates).",
    )
    parser.add_argument(
        "--redsm5-only", action="store_true",
        help="Only extract RedSM5 confounders.",
    )
    parser.add_argument(
        "--bdisen-only", action="store_true",
        help="Only extract BDI-Sen graded confounders.",
    )
    parser.add_argument(
        "--erisk2025-only", action="store_true",
        help="Only extract eRisk 2025 T1 boundaries.",
    )
    parser.add_argument(
        "--erisk2023-only", action="store_true",
        help="Only extract eRisk 2023 boundaries.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing intermediate files, skipping completed steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (not currently used, reserved).",
    )
    return parser.parse_args()


def run_extraction(
    config_path: str = "config/pipeline.yaml",
    symptoms_config: str = "config/symptoms.yaml",
    symptom_ids: list[int] | None = None,
    output_dir: str = "candidates",
    redsm5_only: bool = False,
    bdisen_only: bool = False,
    erisk2025_only: bool = False,
    erisk2023_only: bool = False,
    resume: bool = False,
) -> dict:
    """Run the full external candidate extraction pipeline.

    Returns:
        Summary dict with counts per source and symptom.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    intermediate_dir = out_path / "intermediate"
    log_dir = out_path / "logs"

    # Determine which steps to run
    only_flags = [redsm5_only, bdisen_only, erisk2025_only, erisk2023_only]
    run_all = not any(only_flags)

    # Set up event logger
    event_logger = ExtractionEventLogger(log_dir, run_id)
    event_logger.log(
        "extraction_start",
        config_path=config_path,
        symptoms_config=symptoms_config,
        symptom_ids=symptom_ids,
        redsm5_only=redsm5_only,
        bdisen_only=bdisen_only,
        erisk2025_only=erisk2025_only,
        erisk2023_only=erisk2023_only,
        resume=resume,
    )

    # Load config
    config = load_config(config_path, symptoms_config)
    symptoms = {s.item_number: s for s in config.symptoms}

    summary: dict = {
        "run_id": run_id,
        "redsm5": {},
        "bdisen": {},
        "erisk2025": {},
        "erisk2023": {},
    }

    # Step A: RedSM5 confounders
    if run_all or redsm5_only:
        redsm5_dir = config.redsm5_dir
        if redsm5_dir is None:
            redsm5_dir = config.project_root / "data" / "RedSM5"

        if redsm5_dir.exists():
            logger.info("=" * 60)
            logger.info("STEP A: RedSM5 Confounder Extraction")
            logger.info("=" * 60)

            redsm5_results = extract_redsm5_confounders(
                redsm5_dir=redsm5_dir,
                symptoms=symptoms,
                symptom_ids=symptom_ids,
                intermediate_dir=intermediate_dir,
                event_logger=event_logger,
                resume=resume,
            )

            pool_path = out_path / "redsm5_confounder_pool.tsv"
            total = write_redsm5_pool_tsv(pool_path, redsm5_results)

            summary["redsm5"] = {
                "total_candidates": total,
                "symptoms_covered": len(redsm5_results),
                "per_symptom": {
                    sid: len(cands) for sid, cands in sorted(redsm5_results.items())
                },
            }
        else:
            logger.warning("RedSM5 directory not found: %s", redsm5_dir)

    # Step B: BDI-Sen graded confounders
    if run_all or bdisen_only:
        bdisen_dir = config.bdisen_dir
        if bdisen_dir is None:
            bdisen_dir = config.project_root / "data" / "BDI-Sen" / "full_dataset"

        if bdisen_dir.exists():
            logger.info("=" * 60)
            logger.info("STEP B: BDI-Sen Graded Confounder Extraction")
            logger.info("=" * 60)

            bdisen_results = extract_bdisen_confounders(
                bdisen_dir=bdisen_dir,
                symptoms=symptoms,
                symptom_ids=symptom_ids,
                intermediate_dir=intermediate_dir,
                event_logger=event_logger,
                resume=resume,
            )

            pool_path = out_path / "bdisen_confounder_pool.tsv"
            total = write_bdisen_pool_tsv(pool_path, bdisen_results)

            summary["bdisen"] = {
                "total_candidates": total,
                "symptoms_covered": len(bdisen_results),
                "per_symptom": {
                    sid: len(cands) for sid, cands in sorted(bdisen_results.items())
                },
            }
        else:
            logger.warning("BDI-Sen directory not found: %s", bdisen_dir)

    # Step C: eRisk 2025 T1 boundaries
    if run_all or erisk2025_only:
        erisk2025_dir = config.erisk2025_dir
        if erisk2025_dir is None:
            erisk2025_dir = config.project_root / "data" / "eRisk-2025" / "eRisk25-datasets" / "t1-depression-symptom-ranking"

        trec_dir = config.erisk2025_trec_dir
        if trec_dir is None:
            trec_dir = erisk2025_dir / "erisk25-t1-dataset" / "erisk25-t1-dataset"

        if erisk2025_dir.exists():
            logger.info("=" * 60)
            logger.info("STEP C: eRisk 2025 T1 Boundary Extraction")
            logger.info("=" * 60)

            erisk2025_results = extract_erisk2025_boundary(
                erisk2025_dir=erisk2025_dir,
                trec_dir=trec_dir,
                symptoms=symptoms,
                symptom_ids=symptom_ids,
                intermediate_dir=intermediate_dir,
                event_logger=event_logger,
                resume=resume,
            )

            pool_path = out_path / "erisk2025_t1_boundary_pool.tsv"
            total = write_erisk2025_pool_tsv(pool_path, erisk2025_results)

            summary["erisk2025"] = {
                "total_candidates": total,
                "symptoms_covered": len(erisk2025_results),
                "per_symptom": {
                    sid: len(cands) for sid, cands in sorted(erisk2025_results.items())
                },
            }
        else:
            logger.warning("eRisk 2025 directory not found: %s", erisk2025_dir)

    # Step D: eRisk 2023 boundaries (legacy)
    if run_all or erisk2023_only:
        erisk2023_dir = config.erisk2023_dir
        if erisk2023_dir is None:
            erisk2023_dir = config.project_root / "data" / "eRisk2023_T1"

        trec_dir = config.erisk2023_trec_dir
        if trec_dir is None:
            trec_dir = erisk2023_dir / "new_data"

        if erisk2023_dir.exists():
            logger.info("=" * 60)
            logger.info("STEP D: eRisk 2023 Boundary Extraction")
            logger.info("=" * 60)

            erisk2023_results = extract_erisk2023_boundary(
                erisk2023_dir=erisk2023_dir,
                trec_dir=trec_dir,
                symptoms=symptoms,
                symptom_ids=symptom_ids,
                intermediate_dir=intermediate_dir,
                event_logger=event_logger,
                resume=resume,
            )

            pool_path = out_path / "erisk2023_boundary_pool.tsv"
            total = write_erisk2023_pool_tsv(pool_path, erisk2023_results)

            summary["erisk2023"] = {
                "total_candidates": total,
                "symptoms_covered": len(erisk2023_results),
                "per_symptom": {
                    sid: len(cands) for sid, cands in sorted(erisk2023_results.items())
                },
            }
        else:
            logger.warning("eRisk 2023 directory not found: %s", erisk2023_dir)

    # Final summary
    event_logger.log("extraction_complete", summary=summary)
    event_logger.close()

    _save_intermediate(out_path / "extraction_summary.json", summary)

    logger.info("=" * 60)
    logger.info("EXTERNAL CANDIDATE EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Run ID: %s", run_id)

    for source_name in ["redsm5", "bdisen", "erisk2025", "erisk2023"]:
        src = summary[source_name]
        if src:
            logger.info("  %s: %d candidates across %d symptoms",
                         source_name, src["total_candidates"], src["symptoms_covered"])
            for sid, count in sorted(src.get("per_symptom", {}).items()):
                logger.info("    Symptom %02d: %d candidates", int(sid), count)

    logger.info("  Intermediate files: %s/", intermediate_dir)
    logger.info("  Event log: %s", event_logger.filepath)

    return summary


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symptom_ids = None
    if args.symptoms:
        symptom_ids = [int(x.strip()) for x in args.symptoms.split(",")]

    run_extraction(
        config_path=args.config,
        symptoms_config=args.symptoms_config,
        symptom_ids=symptom_ids,
        output_dir=args.output_dir,
        redsm5_only=args.redsm5_only,
        bdisen_only=args.bdisen_only,
        erisk2025_only=args.erisk2025_only,
        erisk2023_only=args.erisk2023_only,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
