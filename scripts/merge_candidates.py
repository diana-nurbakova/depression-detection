#!/usr/bin/env python3
"""Merge candidates from all sources into per-symptom files.

Implements Step F of annotation_protocol_spec_v3.md Section 3.7:
Combines eRisk2026 retrieval candidates (from retrieve_candidates.py),
RedSM5 confounders, BDI-Sen graded confounders, eRisk 2025 T1 boundary
candidates, and eRisk2023 boundary candidates into unified per-symptom
TSV files for human annotation.

Usage:
    uv run python scripts/merge_candidates.py
    uv run python scripts/merge_candidates.py --symptoms 7,8,9,10,11
    uv run python scripts/merge_candidates.py --candidates-dir candidates
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
from hipert.data.cross_dataset_mappings import ASRS_TO_BDIII, ASRS_TO_BDISEN, ASRS_TO_REDSM5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL event logger
# ---------------------------------------------------------------------------


class MergeEventLogger:
    """JSONL event logger for the merge pipeline."""

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self.filepath = log_dir / f"merge_{run_id}.jsonl"
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
# Merged TSV format
# ---------------------------------------------------------------------------

MERGED_COLUMNS = ["rank", "docno", "pre", "text", "post", "source", "source_detail"]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_retrieval_candidates(filepath: Path) -> list[dict]:
    """Load eRisk2026 retrieval candidates from an existing symptom TSV.

    Reads the TSV produced by retrieve_candidates.py (columns:
    rank, docno, pre, text, post, source).

    Args:
        filepath: Path to symptom_NN_candidates.tsv.

    Returns:
        List of candidate dicts with source_detail added.
    """
    candidates: list[dict] = []

    if not filepath.exists():
        logger.warning("Retrieval TSV not found: %s", filepath)
        return candidates

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source = row.get("source", "retrieval")
            # Existing TSVs have "retrieval" or "random" as source
            if source == "retrieval":
                source_detail = f"retrieval_rank_{row.get('rank', '?')}"
            elif source == "random":
                source_detail = "random_pool"
            else:
                source_detail = row.get("source_detail", source)

            candidates.append({
                "docno": row.get("docno", ""),
                "pre": row.get("pre", ""),
                "text": row.get("text", ""),
                "post": row.get("post", ""),
                "source": source,
                "source_detail": source_detail,
            })

    logger.debug("Loaded %d retrieval candidates from %s", len(candidates), filepath)
    return candidates


def load_redsm5_pool_for_symptom(
    pool_path: Path,
    symptom_id: int,
) -> list[dict]:
    """Load RedSM5 confounder candidates for one symptom from pool TSV.

    Args:
        pool_path: Path to redsm5_confounder_pool.tsv.
        symptom_id: ASRS item number to filter for.

    Returns:
        List of candidate dicts formatted for the merged TSV.
    """
    candidates: list[dict] = []

    if not pool_path.exists():
        logger.debug("RedSM5 pool TSV not found: %s", pool_path)
        return candidates

    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("symptom_id", -1)) != symptom_id:
                continue
            dsm5 = row.get("dsm5_symptom", "?")
            candidates.append({
                "docno": row.get("sentence_id", ""),
                "pre": "",
                "text": row.get("sentence_text", ""),
                "post": "",
                "source": "redsm5",
                "source_detail": f"{dsm5}, status=1",
            })

    logger.debug(
        "Loaded %d RedSM5 candidates for symptom %d", len(candidates), symptom_id,
    )
    return candidates


def load_bdisen_pool_for_symptom(
    pool_path: Path,
    symptom_id: int,
) -> list[dict]:
    """Load BDI-Sen graded confounder candidates for one symptom from pool TSV.

    Args:
        pool_path: Path to bdisen_confounder_pool.tsv.
        symptom_id: ASRS item number to filter for.

    Returns:
        List of candidate dicts formatted for the merged TSV.
    """
    candidates: list[dict] = []

    if not pool_path.exists():
        logger.debug("BDI-Sen pool TSV not found: %s", pool_path)
        return candidates

    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("symptom_id", -1)) != symptom_id:
                continue
            bdisen_sym = row.get("bdisen_symptom", "?")
            severity = row.get("severity", "?")
            candidates.append({
                "docno": f"bdisen_{bdisen_sym}_{severity}_{len(candidates)}",
                "pre": "",
                "text": row.get("sentence_text", ""),
                "post": "",
                "source": "bdisen",
                "source_detail": f"{bdisen_sym}, severity={severity}",
            })

    logger.debug(
        "Loaded %d BDI-Sen candidates for symptom %d", len(candidates), symptom_id,
    )
    return candidates


def load_erisk2025_pool_for_symptom(
    pool_path: Path,
    symptom_id: int,
) -> list[dict]:
    """Load eRisk 2025 T1 boundary candidates for one symptom from pool TSV.

    Args:
        pool_path: Path to erisk2025_t1_boundary_pool.tsv.
        symptom_id: ASRS item number to filter for.

    Returns:
        List of candidate dicts formatted for the merged TSV.
    """
    candidates: list[dict] = []

    if not pool_path.exists():
        logger.debug("eRisk2025 pool TSV not found: %s", pool_path)
        return candidates

    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("symptom_id", -1)) != symptom_id:
                continue
            bdi_q = row.get("bdi_query", "?")
            maj = row.get("majority", "?")
            con = row.get("consensus", "?")
            candidates.append({
                "docno": row.get("docid", ""),
                "pre": row.get("pre", ""),
                "text": row.get("text", ""),
                "post": row.get("post", ""),
                "source": "erisk2025",
                "source_detail": f"BDI-{bdi_q}, majority={maj}, consensus={con}",
            })

    logger.debug(
        "Loaded %d eRisk2025 candidates for symptom %d", len(candidates), symptom_id,
    )
    return candidates


def load_erisk2023_pool_for_symptom(
    pool_path: Path,
    symptom_id: int,
) -> list[dict]:
    """Load eRisk 2023 boundary candidates for one symptom from pool TSV.

    Args:
        pool_path: Path to erisk2023_boundary_pool.tsv.
        symptom_id: ASRS item number to filter for.

    Returns:
        List of candidate dicts formatted for the merged TSV.
    """
    candidates: list[dict] = []

    if not pool_path.exists():
        logger.debug("eRisk2023 pool TSV not found: %s", pool_path)
        return candidates

    with open(pool_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("symptom_id", -1)) != symptom_id:
                continue
            bdi_q = row.get("bdi_query", "?")
            maj = row.get("majority", "?")
            con = row.get("consensus", "?")
            candidates.append({
                "docno": row.get("docid", ""),
                "pre": "",
                "text": row.get("text", ""),
                "post": "",
                "source": "erisk2023",
                "source_detail": f"BDI-{bdi_q}, majority={maj}, consensus={con}",
            })

    logger.debug(
        "Loaded %d eRisk2023 candidates for symptom %d", len(candidates), symptom_id,
    )
    return candidates


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


def merge_symptom_candidates(
    retrieval: list[dict],
    redsm5: list[dict],
    bdisen: list[dict],
    erisk2025: list[dict],
    erisk2023: list[dict],
) -> list[dict]:
    """Merge candidates from all sources with sequential ranking.

    Order per spec v3 Section 2.5: retrieval (excluding score-0) first,
    then RedSM5 confounders, then BDI-Sen graded confounders, then
    eRisk 2025 T1 boundaries, then eRisk 2023 boundaries, then
    score-0 random candidates last.

    Deduplicates by docno (first occurrence wins).

    Args:
        retrieval: eRisk2026 retrieval + score-0 candidates.
        redsm5: RedSM5 confounder candidates.
        bdisen: BDI-Sen graded confounder candidates.
        erisk2025: eRisk 2025 T1 boundary candidates.
        erisk2023: eRisk2023 boundary candidates.

    Returns:
        Merged list with sequential rank numbers assigned.
    """
    # Split retrieval into actual retrieval and score-0
    retrieval_proper = [c for c in retrieval if c["source"] != "random"]
    score0 = [c for c in retrieval if c["source"] == "random"]

    # Merge in spec order: retrieval, RedSM5, BDI-Sen, eRisk2025, eRisk2023, score-0
    all_candidates = retrieval_proper + redsm5 + bdisen + erisk2025 + erisk2023 + score0

    # Deduplicate by docno
    seen: set[str] = set()
    merged: list[dict] = []
    for cand in all_candidates:
        docno = cand["docno"]
        if docno and docno in seen:
            continue
        seen.add(docno)
        merged.append(cand)

    # Assign sequential rank numbers
    for i, cand in enumerate(merged, start=1):
        cand["rank"] = i

    return merged


def write_merged_tsv(filepath: Path, candidates: list[dict]) -> int:
    """Write merged candidates to a TSV file.

    Args:
        filepath: Output path.
        candidates: List of candidate dicts with all MERGED_COLUMNS.

    Returns:
        Number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_COLUMNS, delimiter="\t")
        writer.writeheader()

        for cand in candidates:
            writer.writerow({col: cand.get(col, "") for col in MERGED_COLUMNS})

    return len(candidates)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_merge(
    candidates_dir: str = "candidates",
    config_path: str = "config/pipeline.yaml",
    symptoms_config: str = "config/symptoms.yaml",
    symptom_ids: list[int] | None = None,
) -> dict:
    """Run the full candidate merge pipeline.

    Returns:
        Summary dict with counts per symptom and source.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cand_path = Path(candidates_dir)
    log_dir = cand_path / "logs"
    intermediate_dir = cand_path / "intermediate"

    event_logger = MergeEventLogger(log_dir, run_id)
    event_logger.log(
        "merge_start",
        candidates_dir=candidates_dir,
        symptom_ids=symptom_ids,
    )

    # Load config for symptom list
    config = load_config(config_path, symptoms_config)
    all_symptom_ids = sorted(s.item_number for s in config.symptoms)

    if symptom_ids is None:
        symptom_ids = all_symptom_ids

    # Pool file paths
    redsm5_pool = cand_path / "redsm5_confounder_pool.tsv"
    bdisen_pool = cand_path / "bdisen_confounder_pool.tsv"
    erisk2025_pool = cand_path / "erisk2025_t1_boundary_pool.tsv"
    erisk2023_pool = cand_path / "erisk2023_boundary_pool.tsv"

    logger.info("=" * 60)
    logger.info("STEP F: Merge Candidates from All Sources")
    logger.info("=" * 60)
    logger.info("  Candidates dir: %s", cand_path)
    logger.info("  RedSM5 pool: %s (exists: %s)", redsm5_pool, redsm5_pool.exists())
    logger.info("  BDI-Sen pool: %s (exists: %s)", bdisen_pool, bdisen_pool.exists())
    logger.info("  eRisk2025 pool: %s (exists: %s)", erisk2025_pool, erisk2025_pool.exists())
    logger.info("  eRisk2023 pool: %s (exists: %s)", erisk2023_pool, erisk2023_pool.exists())

    summary: dict[int, dict] = {}

    for symptom_id in symptom_ids:
        # Load from all sources
        retrieval_tsv = cand_path / f"symptom_{symptom_id:02d}_candidates.tsv"
        retrieval = load_retrieval_candidates(retrieval_tsv)
        redsm5 = load_redsm5_pool_for_symptom(redsm5_pool, symptom_id)
        bdisen = load_bdisen_pool_for_symptom(bdisen_pool, symptom_id)
        erisk2025 = load_erisk2025_pool_for_symptom(erisk2025_pool, symptom_id)
        erisk2023 = load_erisk2023_pool_for_symptom(erisk2023_pool, symptom_id)

        # Merge
        merged = merge_symptom_candidates(retrieval, redsm5, bdisen, erisk2025, erisk2023)

        # Write merged TSV (overwrites the original retrieval-only file)
        output_path = cand_path / f"symptom_{symptom_id:02d}_candidates.tsv"
        total = write_merged_tsv(output_path, merged)

        # Count by source
        source_counts = {}
        for cand in merged:
            src = cand.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        summary[symptom_id] = {
            "total": total,
            "sources": source_counts,
        }

        logger.info(
            "Symptom %02d: %d merged candidates (%s)",
            symptom_id, total,
            ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())),
        )

        event_logger.log(
            "symptom_merge",
            symptom_id=symptom_id,
            total=total,
            source_counts=source_counts,
        )

        # Save intermediate
        intermediate_path = intermediate_dir / f"merged_symptom_{symptom_id:02d}.json"
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    # Final summary
    event_logger.log("merge_complete", summary={str(k): v for k, v in summary.items()})
    event_logger.close()

    # Save summary
    summary_path = cand_path / "merge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "symptoms": {str(k): v for k, v in summary.items()}},
            f, indent=2, ensure_ascii=False,
        )

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    for symptom_id in symptom_ids:
        s = summary[symptom_id]
        logger.info(
            "  Symptom %02d: %d total — %s",
            symptom_id, s["total"],
            ", ".join(f"{k}={v}" for k, v in sorted(s["sources"].items())),
        )
    logger.info("  Merged TSVs: %s/symptom_*_candidates.tsv", cand_path)
    logger.info("  Event log: %s", log_dir)

    return {"run_id": run_id, "symptoms": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge candidates from all sources into per-symptom files.",
    )
    parser.add_argument(
        "--candidates-dir", type=str, default="candidates",
        help="Directory containing candidate pool TSV files (default: candidates).",
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
        help="Comma-separated symptom IDs (default: all 18).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symptom_ids = None
    if args.symptoms:
        symptom_ids = [int(x.strip()) for x in args.symptoms.split(",")]

    run_merge(
        candidates_dir=args.candidates_dir,
        config_path=args.config,
        symptoms_config=args.symptoms_config,
        symptom_ids=symptom_ids,
    )


if __name__ == "__main__":
    main()
