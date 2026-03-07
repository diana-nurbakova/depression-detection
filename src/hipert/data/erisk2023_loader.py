"""Load and process eRisk 2023 Task 1 BDI-II data for boundary extraction.

eRisk 2023 provides 21,580 relevance judgments across 21 BDI-II queries
in two granularities: majority vote and strict consensus. Sentences where
majority=relevant but consensus=not-relevant are natural borderline cases
(~2,076 sentences) useful as score-1 boundary candidates.

Data files:
    data/eRisk2023_T1/g_qrels_majority_2.csv — majority vote qrels
    data/eRisk2023_T1/g_rels_consenso.csv    — strict consensus qrels
    data/eRisk2023_T1/new_data/*.trec        — 3,107 user TREC files
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hipert.data.trec_parser import parse_trec_file_simple
from hipert.models import FIRST_PERSON_MARKERS, Sentence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ERisk2023Qrel:
    """One relevance judgment row from eRisk 2023 qrels."""

    query: int    # BDI-II query number (1-21)
    docid: str    # e.g. "s_405_1279_15"
    rel: int      # 0 or 1


def load_qrels(filepath: Path) -> list[ERisk2023Qrel]:
    """Load a qrel CSV file (majority or consensus).

    Expected columns: query, q0, docid, rel

    Args:
        filepath: Path to the qrel CSV.

    Returns:
        List of all qrel records.
    """
    qrels: list[ERisk2023Qrel] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qrels.append(ERisk2023Qrel(
                query=int(row["query"].strip()),
                docid=row["docid"].strip(),
                rel=int(row["rel"].strip()),
            ))

    logger.info("Loaded %d qrels from %s", len(qrels), filepath)
    return qrels


def load_qrels_by_query(filepath: Path) -> dict[int, list[ERisk2023Qrel]]:
    """Load qrels grouped by BDI-II query number.

    Args:
        filepath: Path to the qrel CSV.

    Returns:
        Dict mapping query number (1-21) to list of qrels.
    """
    all_qrels = load_qrels(filepath)
    grouped: dict[int, list[ERisk2023Qrel]] = defaultdict(list)

    for qrel in all_qrels:
        grouped[qrel.query].append(qrel)

    for query, qrels_list in sorted(grouped.items()):
        n_relevant = sum(1 for q in qrels_list if q.rel == 1)
        logger.debug(
            "  BDI-II query %d: %d total, %d relevant",
            query, len(qrels_list), n_relevant,
        )

    return dict(grouped)


def docid_to_user_file(docid: str) -> str:
    """Extract the user TREC filename from a docid.

    Docid format: ``s_{user}_{post}_{sentence}``
    TREC file: ``s_{user}.trec``

    For example, ``s_405_1279_15`` -> user ``405`` -> file ``s_405.trec``.
    The first element after ``s_`` is the user ID; subsequent parts are
    post and sentence indices.

    Args:
        docid: Document ID from qrels.

    Returns:
        TREC filename (e.g. "s_405.trec").
    """
    parts = docid.split("_")
    # Format is s_{user}_{post}_{sentence}, so parts[0]="s", parts[1]=user
    if len(parts) >= 4 and parts[0] == "s":
        user_id = parts[1]
        return f"s_{user_id}.trec"
    # Fallback: use everything up to the second-to-last underscore
    return parts[0] + ".trec"


def build_sentence_lookup(
    trec_dir: Path,
    docids: set[str] | None = None,
) -> dict[str, Sentence]:
    """Load eRisk 2023 TREC files into a docno -> Sentence lookup.

    Uses ``parse_trec_file_simple`` for the eRisk 2023 format (DOCNO+TEXT
    only, no PRE/POST context).

    Args:
        trec_dir: Directory containing .trec files (e.g. new_data/).
        docids: If provided, only load TREC files for users referenced
            by these docids. This avoids loading all 3,107 files.

    Returns:
        Dict mapping docno string to Sentence object.
    """
    if docids is not None:
        # Determine which user files we need
        needed_files = {docid_to_user_file(d) for d in docids}
        trec_files = [trec_dir / f for f in sorted(needed_files)]
        trec_files = [f for f in trec_files if f.exists()]
        logger.info(
            "Loading %d targeted TREC files (from %d unique docids)",
            len(trec_files), len(docids),
        )
    else:
        trec_files = sorted(trec_dir.glob("*.trec"))
        logger.info("Loading all %d TREC files from %s", len(trec_files), trec_dir)

    lookup: dict[str, Sentence] = {}
    total_sentences = 0

    for trec_path in trec_files:
        sentences = parse_trec_file_simple(trec_path)
        for sent in sentences:
            lookup[sent.docno] = sent
        total_sentences += len(sentences)

    logger.info(
        "Built sentence lookup: %d unique docnos from %d sentences in %d files",
        len(lookup), total_sentences, len(trec_files),
    )

    return lookup


def extract_boundary_candidates(
    majority_path: Path,
    consenso_path: Path,
    bdi_query: int,
) -> tuple[list[str], list[str]]:
    """Extract disagreement and agreement docid sets for one BDI-II query.

    - **Disagreement set:** majority=1 AND consensus=0
      Natural borderline cases where annotators disagreed.
    - **Agreement set:** majority=1 AND consensus=1
      Confirmed relevant sentences, potential score-2 candidates.

    Args:
        majority_path: Path to g_qrels_majority_2.csv.
        consenso_path: Path to g_rels_consenso.csv.
        bdi_query: BDI-II query number (1-21).

    Returns:
        Tuple of (disagreement_docids, agreement_docids).
    """
    majority_by_query = load_qrels_by_query(majority_path)
    consenso_by_query = load_qrels_by_query(consenso_path)

    majority_qrels = majority_by_query.get(bdi_query, [])
    consenso_qrels = consenso_by_query.get(bdi_query, [])

    # Build lookup: docid -> rel for each set
    majority_rel = {q.docid: q.rel for q in majority_qrels}
    consenso_rel = {q.docid: q.rel for q in consenso_qrels}

    # All docids that appear in either set
    all_docids = set(majority_rel.keys()) | set(consenso_rel.keys())

    disagreement: list[str] = []
    agreement: list[str] = []

    for docid in sorted(all_docids):
        maj = majority_rel.get(docid, 0)
        con = consenso_rel.get(docid, 0)

        if maj == 1 and con == 0:
            disagreement.append(docid)
        elif maj == 1 and con == 1:
            agreement.append(docid)

    logger.info(
        "BDI-II query %d: %d disagreement, %d agreement (from %d total docids)",
        bdi_query, len(disagreement), len(agreement), len(all_docids),
    )

    return disagreement, agreement


def extract_boundary_candidates_cached(
    majority_by_query: dict[int, list[ERisk2023Qrel]],
    consenso_by_query: dict[int, list[ERisk2023Qrel]],
    bdi_query: int,
) -> tuple[list[str], list[str]]:
    """Same as extract_boundary_candidates but with pre-loaded qrels.

    Avoids re-reading CSV files when processing multiple queries.

    Args:
        majority_by_query: Pre-loaded majority qrels grouped by query.
        consenso_by_query: Pre-loaded consensus qrels grouped by query.
        bdi_query: BDI-II query number (1-21).

    Returns:
        Tuple of (disagreement_docids, agreement_docids).
    """
    majority_qrels = majority_by_query.get(bdi_query, [])
    consenso_qrels = consenso_by_query.get(bdi_query, [])

    majority_rel = {q.docid: q.rel for q in majority_qrels}
    consenso_rel = {q.docid: q.rel for q in consenso_qrels}

    all_docids = set(majority_rel.keys()) | set(consenso_rel.keys())

    disagreement: list[str] = []
    agreement: list[str] = []

    for docid in sorted(all_docids):
        maj = majority_rel.get(docid, 0)
        con = consenso_rel.get(docid, 0)

        if maj == 1 and con == 0:
            disagreement.append(docid)
        elif maj == 1 and con == 1:
            agreement.append(docid)

    logger.info(
        "BDI-II query %d: %d disagreement, %d agreement (from %d total docids)",
        bdi_query, len(disagreement), len(agreement), len(all_docids),
    )

    return disagreement, agreement


def resolve_sentences(
    docids: list[str],
    lookup: dict[str, Sentence],
    require_first_person: bool = True,
) -> list[Sentence]:
    """Look up sentence texts from the TREC corpus by docid.

    Args:
        docids: List of document IDs to resolve.
        lookup: docno -> Sentence mapping from build_sentence_lookup().
        require_first_person: If True, only return sentences with
            first-person markers.

    Returns:
        List of resolved Sentence objects.
    """
    resolved: list[Sentence] = []
    missing = 0

    for docid in docids:
        sent = lookup.get(docid)
        if sent is None:
            missing += 1
            continue
        if require_first_person and not sent.has_first_person:
            continue
        resolved.append(sent)

    if missing > 0:
        logger.warning(
            "%d of %d docids not found in TREC corpus",
            missing, len(docids),
        )

    logger.debug(
        "Resolved %d sentences from %d docids (%d missing, first_person=%s)",
        len(resolved), len(docids), missing, require_first_person,
    )

    return resolved
