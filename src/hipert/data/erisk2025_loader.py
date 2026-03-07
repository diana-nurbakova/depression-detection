"""Load and process eRisk 2025 Task 1 BDI-II data for boundary extraction.

eRisk 2025 provides 11,042 relevance judgments across 21 BDI-II queries
in two granularities: majority vote and strict consensus. Sentences where
majority=relevant but consensus=not-relevant are natural borderline cases
(~2,707 sentences) useful as score-1 boundary candidates.

Key format differences from eRisk 2023:
  - CSV columns: query,doc_id,relevant (True/False strings)
  - TREC format: full PRE/TEXT/POST (same as eRisk 2026)
  - Docids: {userId}_{postIdx}_{sentIdx} (no "s_" prefix)
  - TREC files: numbered s_0.trec..s_6299.trec (not user-ID-based)

Data files:
    data/eRisk-2025/.../qrels_consensus_merged.csv
    data/eRisk-2025/.../qrels_majority_merged.csv
    data/eRisk-2025/.../erisk25-t1-dataset/erisk25-t1-dataset/*.trec
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hipert.data.trec_parser import parse_trec_file
from hipert.models import FIRST_PERSON_MARKERS, Sentence

logger = logging.getLogger(__name__)

# Regex to extract first DOCNO from a TREC file without full parsing.
_FIRST_DOCNO_RE = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>", re.DOTALL)


@dataclass(frozen=True)
class ERisk2025Qrel:
    """One relevance judgment row from eRisk 2025 qrels."""

    query: int       # BDI-II query number (1-21)
    doc_id: str      # e.g. "PgZVTC_0_0"
    relevant: bool   # True/False


def load_qrels(filepath: Path) -> list[ERisk2025Qrel]:
    """Load a qrel CSV file (majority or consensus).

    Expected columns: query,doc_id,relevant (with True/False strings).

    Args:
        filepath: Path to the qrel CSV.

    Returns:
        List of all qrel records.
    """
    import csv

    qrels: list[ERisk2025Qrel] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qrels.append(ERisk2025Qrel(
                query=int(row["query"].strip()),
                doc_id=row["doc_id"].strip(),
                relevant=row["relevant"].strip() == "True",
            ))

    logger.info("Loaded %d qrels from %s", len(qrels), filepath)
    return qrels


def load_qrels_by_query(filepath: Path) -> dict[int, list[ERisk2025Qrel]]:
    """Load qrels grouped by BDI-II query number.

    Args:
        filepath: Path to the qrel CSV.

    Returns:
        Dict mapping query number (1-21) to list of qrels.
    """
    all_qrels = load_qrels(filepath)
    grouped: dict[int, list[ERisk2025Qrel]] = defaultdict(list)

    for qrel in all_qrels:
        grouped[qrel.query].append(qrel)

    for query, qrels_list in sorted(grouped.items()):
        n_relevant = sum(1 for q in qrels_list if q.relevant)
        logger.debug(
            "  BDI-II query %d: %d total, %d relevant",
            query, len(qrels_list), n_relevant,
        )

    return dict(grouped)


def docid_to_user_id(docid: str) -> str:
    """Extract the user ID from a docid.

    Docid format: ``{userId}_{postIdx}_{sentIdx}``
    The userId may contain letters and digits but not underscores.

    For example, ``PgZVTC_0_0`` -> ``PgZVTC``.

    Since post and sentence indices are always numeric, we split from
    the right: the last two underscore-separated parts are sentIdx and
    postIdx, everything before is the userId.

    Args:
        docid: Document ID from qrels.

    Returns:
        User ID string.
    """
    parts = docid.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    # Fallback for unexpected format
    return docid.split("_")[0]


def build_user_index(trec_dir: Path) -> dict[str, Path]:
    """Build a mapping from user ID to TREC file path.

    Reads the first DOCNO from each .trec file to determine which user
    it contains. This is a one-time cost (~6,300 file peeks).

    Args:
        trec_dir: Directory containing s_*.trec files.

    Returns:
        Dict mapping user_id to the Path of its TREC file.
    """
    index: dict[str, Path] = {}
    trec_files = sorted(trec_dir.glob("s_*.trec"))

    for trec_path in trec_files:
        # Read just enough to find the first DOCNO
        try:
            with open(trec_path, "r", encoding="utf-8", errors="replace") as f:
                # First DOCNO is typically within the first 200 chars
                head = f.read(500)
        except OSError:
            continue

        match = _FIRST_DOCNO_RE.search(head)
        if match:
            first_docno = match.group(1).strip()
            user_id = docid_to_user_id(first_docno)
            index[user_id] = trec_path

    logger.info(
        "Built user index: %d users from %d TREC files in %s",
        len(index), len(trec_files), trec_dir,
    )
    return index


def build_sentence_lookup(
    trec_dir: Path,
    docids: set[str] | None = None,
    user_index: dict[str, Path] | None = None,
) -> dict[str, Sentence]:
    """Load eRisk 2025 TREC files into a docno -> Sentence lookup.

    Uses ``parse_trec_file`` for the eRisk 2025 format (full PRE/TEXT/POST).

    Args:
        trec_dir: Directory containing s_*.trec files.
        docids: If provided, only load TREC files for users referenced
            by these docids.
        user_index: Pre-built user->file index. If None, will be built.

    Returns:
        Dict mapping docno string to Sentence object.
    """
    if user_index is None:
        user_index = build_user_index(trec_dir)

    if docids is not None:
        needed_users = {docid_to_user_id(d) for d in docids}
        trec_files = []
        missing_users = 0
        for uid in sorted(needed_users):
            fpath = user_index.get(uid)
            if fpath is not None:
                trec_files.append(fpath)
            else:
                missing_users += 1
        if missing_users:
            logger.warning(
                "%d users not found in index (out of %d needed)",
                missing_users, len(needed_users),
            )
        logger.info(
            "Loading %d targeted TREC files (from %d unique docids, %d users)",
            len(trec_files), len(docids), len(needed_users),
        )
    else:
        trec_files = sorted(trec_dir.glob("s_*.trec"))
        logger.info("Loading all %d TREC files from %s", len(trec_files), trec_dir)

    lookup: dict[str, Sentence] = {}
    total_sentences = 0

    for trec_path in trec_files:
        sentences = parse_trec_file(trec_path)
        for sent in sentences:
            lookup[sent.docno] = sent
        total_sentences += len(sentences)

    logger.info(
        "Built sentence lookup: %d unique docnos from %d sentences in %d files",
        len(lookup), total_sentences, len(trec_files),
    )
    return lookup


def extract_boundary_candidates_cached(
    majority_by_query: dict[int, list[ERisk2025Qrel]],
    consensus_by_query: dict[int, list[ERisk2025Qrel]],
    bdi_query: int,
) -> tuple[list[str], list[str]]:
    """Extract disagreement and agreement doc_id sets for one BDI-II query.

    - **Disagreement set:** majority=True AND consensus=False
      Natural borderline cases where annotators disagreed.
    - **Agreement set:** majority=True AND consensus=True
      Confirmed relevant sentences, potential score-2 candidates.

    Args:
        majority_by_query: Pre-loaded majority qrels grouped by query.
        consensus_by_query: Pre-loaded consensus qrels grouped by query.
        bdi_query: BDI-II query number (1-21).

    Returns:
        Tuple of (disagreement_doc_ids, agreement_doc_ids).
    """
    majority_qrels = majority_by_query.get(bdi_query, [])
    consensus_qrels = consensus_by_query.get(bdi_query, [])

    majority_rel = {q.doc_id: q.relevant for q in majority_qrels}
    consensus_rel = {q.doc_id: q.relevant for q in consensus_qrels}

    all_doc_ids = set(majority_rel.keys()) | set(consensus_rel.keys())

    disagreement: list[str] = []
    agreement: list[str] = []

    for doc_id in sorted(all_doc_ids):
        maj = majority_rel.get(doc_id, False)
        con = consensus_rel.get(doc_id, False)

        if maj and not con:
            disagreement.append(doc_id)
        elif maj and con:
            agreement.append(doc_id)

    logger.info(
        "BDI-II query %d: %d disagreement, %d agreement (from %d total doc_ids)",
        bdi_query, len(disagreement), len(agreement), len(all_doc_ids),
    )

    return disagreement, agreement


def resolve_sentences(
    doc_ids: list[str],
    lookup: dict[str, Sentence],
    require_first_person: bool = True,
) -> list[Sentence]:
    """Look up sentence texts from the TREC corpus by doc_id.

    Args:
        doc_ids: List of document IDs to resolve.
        lookup: docno -> Sentence mapping from build_sentence_lookup().
        require_first_person: If True, only return sentences with
            first-person markers.

    Returns:
        List of resolved Sentence objects.
    """
    resolved: list[Sentence] = []
    missing = 0

    for doc_id in doc_ids:
        sent = lookup.get(doc_id)
        if sent is None:
            missing += 1
            continue
        if require_first_person and not sent.has_first_person:
            continue
        resolved.append(sent)

    if missing > 0:
        logger.warning(
            "%d of %d doc_ids not found in TREC corpus",
            missing, len(doc_ids),
        )

    logger.debug(
        "Resolved %d sentences from %d doc_ids (%d missing, first_person=%s)",
        len(resolved), len(doc_ids), missing, require_first_person,
    )

    return resolved
