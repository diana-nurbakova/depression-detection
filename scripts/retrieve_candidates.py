#!/usr/bin/env python3
"""Pre-filtering pipeline for annotation candidate retrieval.

Generates ~35 candidate sentences per ASRS symptom for human annotation
of few-shot examples. Implements Steps A-E of the annotation protocol
(specs/annotation_protocol_spec.md Section 3).

Usage:
    uv run python scripts/retrieve_candidates.py
    uv run python scripts/retrieve_candidates.py --symptoms 5,12
    uv run python scripts/retrieve_candidates.py --output-dir candidates
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hipert.config import load_config
from hipert.data.corpus import load_corpus
from hipert.models import Sentence, SymptomDefinition
from hipert.retrieval.annotation_queries import build_annotation_query
from hipert.retrieval.encoder import BiEncoderRetriever

logger = logging.getLogger(__name__)

TSV_COLUMNS = ["rank", "docno", "pre", "text", "post", "source"]


# ---------------------------------------------------------------------------
# Step A-C: Per-symptom retrieval
# ---------------------------------------------------------------------------


def retrieve_candidates_for_symptom(
    retriever: BiEncoderRetriever,
    symptom: SymptomDefinition,
    sentences: list[Sentence],
    top_k: int = 50,
    exclusion_k: int = 200,
) -> tuple[list[tuple[Sentence, int, float]], set[int]]:
    """Retrieve and filter candidates for one symptom.

    Steps A-C from the annotation protocol:
      A. Build expanded query from ASRS text + Layer 3 expressions
      B. Retrieve top candidates via bi-encoder cosine similarity
      C. Apply first-person filter

    Args:
        retriever: Initialized BiEncoderRetriever with encoded corpus.
        symptom: The symptom definition to query for.
        sentences: Full corpus sentence list (index-aligned with embeddings).
        top_k: Number of top candidates to keep before filtering.
        exclusion_k: Top-K for the exclusion set (score-0 must be outside).

    Returns:
        Tuple of:
        - Filtered candidates as list of (Sentence, rank, similarity_score)
        - Set of sentence indices in the top-exclusion_k
    """
    # Step A: Build annotation-specific query
    query = build_annotation_query(symptom)
    logger.info(
        "Symptom %d query: %.120s...",
        symptom.item_number, query,
    )

    # Step B: Retrieve top exclusion_k (superset of top_k)
    raw_results = retriever.retrieve(query, top_k=exclusion_k)

    # Track exclusion set (all indices in top-200)
    exclusion_indices = {idx for idx, _ in raw_results}

    # Take only top_k from results
    top_results = raw_results[:top_k]

    # Build (Sentence, rank, score) tuples
    candidates = [
        (sentences[idx], rank, score)
        for rank, (idx, score) in enumerate(top_results, start=1)
    ]

    # Step C: First-person filter
    filtered = [
        (sent, rank, score)
        for sent, rank, score in candidates
        if sent.has_first_person
    ]

    logger.info(
        "Symptom %d: %d/%d passed first-person filter",
        symptom.item_number, len(filtered), len(candidates),
    )

    return filtered, exclusion_indices


# ---------------------------------------------------------------------------
# Step D: Score-0 pool and selection
# ---------------------------------------------------------------------------


def build_score0_pool(
    sentences: list[Sentence],
    all_exclusion_indices: set[int],
    pool_size: int = 20,
    rng: random.Random | None = None,
) -> list[Sentence]:
    """Build the shared score-0 pool (Section 3.3).

    Selects random first-person sentences NOT in the top-200
    for ANY symptom. These serve as cross-symptom irrelevant candidates.

    Args:
        sentences: Full corpus sentence list.
        all_exclusion_indices: Union of top-200 indices across all symptoms.
        pool_size: Number of sentences to sample.
        rng: Random number generator for reproducibility.

    Returns:
        List of Sentence objects for the shared pool.
    """
    if rng is None:
        rng = random.Random(42)

    eligible_indices = [
        i for i, sent in enumerate(sentences)
        if sent.has_first_person and i not in all_exclusion_indices
    ]

    logger.info(
        "Score-0 pool: %d eligible sentences (first-person, not in any top-%d)",
        len(eligible_indices), len(all_exclusion_indices),
    )

    if len(eligible_indices) <= pool_size:
        selected_indices = eligible_indices
    else:
        selected_indices = rng.sample(eligible_indices, pool_size)

    return [sentences[i] for i in selected_indices]


def select_score0_for_symptom(
    sentences: list[Sentence],
    exclusion_indices: set[int],
    shared_pool: list[Sentence],
    count: int = 5,
    rng: random.Random | None = None,
) -> list[Sentence]:
    """Select score-0 candidates for a single symptom (Step D).

    Draws from the shared pool first, then supplements with
    symptom-specific random sentences if needed.

    Args:
        sentences: Full corpus.
        exclusion_indices: This symptom's top-200 indices.
        shared_pool: The cross-symptom shared pool.
        count: Number of score-0 candidates needed.
        rng: Random number generator.

    Returns:
        List of score-0 Sentence objects.
    """
    if rng is None:
        rng = random.Random(42)

    # Docnos in this symptom's exclusion set for fast lookup
    exclusion_docnos = {sentences[i].docno for i in exclusion_indices}

    # Filter shared pool to exclude sentences in this symptom's top-200
    eligible_from_pool = [
        s for s in shared_pool
        if s.docno not in exclusion_docnos
    ]

    if len(eligible_from_pool) >= count:
        return rng.sample(eligible_from_pool, count)

    # Supplement with additional random sentences
    selected = list(eligible_from_pool)
    needed = count - len(selected)
    selected_docnos = {s.docno for s in selected}
    pool_docnos = {s.docno for s in shared_pool}

    supplementary_indices = [
        i for i, sent in enumerate(sentences)
        if (sent.has_first_person
            and i not in exclusion_indices
            and sent.docno not in selected_docnos
            and sent.docno not in pool_docnos)
    ]

    if supplementary_indices:
        extra_indices = rng.sample(
            supplementary_indices,
            min(needed, len(supplementary_indices)),
        )
        selected.extend(sentences[i] for i in extra_indices)

    return selected


# ---------------------------------------------------------------------------
# Step E: Output writing
# ---------------------------------------------------------------------------


def write_symptom_candidates_tsv(
    filepath: Path,
    retrieval_candidates: list[tuple[Sentence, int, float]],
    score0_candidates: list[Sentence],
) -> int:
    """Write one symptom's candidates to TSV.

    Args:
        filepath: Output path for the TSV file.
        retrieval_candidates: List of (Sentence, rank, score) from retrieval.
        score0_candidates: List of Sentence objects for score-0 rows.

    Returns:
        Total number of rows written.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()

        for sent, rank, _score in retrieval_candidates:
            writer.writerow({
                "rank": rank,
                "docno": sent.docno,
                "pre": sent.pre,
                "text": sent.text,
                "post": sent.post,
                "source": "retrieval",
            })

        base_rank = len(retrieval_candidates) + 1
        for i, sent in enumerate(score0_candidates):
            writer.writerow({
                "rank": base_rank + i,
                "docno": sent.docno,
                "pre": sent.pre,
                "text": sent.text,
                "post": sent.post,
                "source": "random",
            })

    return len(retrieval_candidates) + len(score0_candidates)


def write_score0_pool_tsv(
    filepath: Path,
    pool: list[Sentence],
) -> None:
    """Write the shared score-0 pool to TSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["docno", "pre", "text", "post"],
            delimiter="\t",
        )
        writer.writeheader()
        for sent in pool:
            writer.writerow({
                "docno": sent.docno,
                "pre": sent.pre,
                "text": sent.text,
                "post": sent.post,
            })


# ---------------------------------------------------------------------------
# Annotation template generation
# ---------------------------------------------------------------------------


def create_annotation_templates(
    annotations_dir: Path,
    symptoms: dict[int, SymptomDefinition],
) -> None:
    """Create empty annotation template JSON files.

    These are skeleton files the annotator fills in during the
    manual annotation phase (Section 4 of the annotation protocol).
    """
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Per-symptom template files
    for item_num, symptom in sorted(symptoms.items()):
        template = {
            "symptom_id": item_num,
            "symptom_text": symptom.text,
            "symptom_factor": symptom.factor.value,
            "annotator": "",
            "annotation_date": "",
            "examples": [
                {
                    "score": score,
                    "docno": "",
                    "pre": "",
                    "text": "",
                    "post": "",
                    "source": "",
                    "synthetic": False,
                    "annotation": {
                        "symptom_match": "",
                        "self_reference": "",
                        "detail_level": "",
                        "confounders": "",
                        "score": score,
                        "confidence": 0,
                        "reasoning": "",
                    },
                }
                for score in [0, 1, 2, 3]
            ],
        }

        filepath = annotations_dir / f"symptom_{item_num:02d}_examples.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

    # Annotation summary template
    summary_path = annotations_dir / "annotation_summary.tsv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symptom_id", "score_0_docno", "score_1_docno",
                "score_2_docno", "score_3_docno",
                "synthetic_count", "notes",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item_num in sorted(symptoms.keys()):
            writer.writerow({
                "symptom_id": item_num,
                "score_0_docno": "",
                "score_1_docno": "",
                "score_2_docno": "",
                "score_3_docno": "",
                "synthetic_count": 0,
                "notes": "",
            })

    # Synthetic examples tracking file
    synthetic_path = annotations_dir / "synthetic_examples.json"
    with open(synthetic_path, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

    # Shared score-0 pool tracking file
    pool_path = annotations_dir / "score0_shared_pool.json"
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Shared score-0 sentences reusable within the same factor",
            "constraint": "Do NOT reuse score-0 across factors",
            "sentences": [],
        }, f, indent=2)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve annotation candidates per ADHD symptom.",
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
    parser.add_argument(
        "--output-dir", type=str, default="candidates",
        help="Output directory for candidate TSV files.",
    )
    parser.add_argument(
        "--annotations-dir", type=str, default="annotations",
        help="Output directory for annotation template JSON files.",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top candidates to retrieve per symptom (default: 50).",
    )
    parser.add_argument(
        "--exclusion-k", type=int, default=200,
        help="Top-K for exclusion set; score-0 must be outside (default: 200).",
    )
    parser.add_argument(
        "--score0-per-symptom", type=int, default=5,
        help="Random score-0 candidates per symptom (default: 5).",
    )
    parser.add_argument(
        "--score0-pool-size", type=int, default=20,
        help="Size of shared score-0 pool (default: 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_candidate_retrieval(
    config_path: str = "config/pipeline.yaml",
    symptoms_config: str = "config/symptoms.yaml",
    symptom_ids: list[int] | None = None,
    output_dir: str = "candidates",
    annotations_dir: str = "annotations",
    top_k: int = 50,
    exclusion_k: int = 200,
    score0_per_symptom: int = 5,
    score0_pool_size: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    """Run the full candidate retrieval pipeline.

    This is the core function that both the standalone script and the
    CLI subcommand call.

    Returns:
        Summary dict with counts per symptom and pool size.
    """
    rng = random.Random(seed)

    # Load config
    config = load_config(config_path, symptoms_config)
    symptoms = {s.item_number: s for s in config.symptoms}

    # Resolve symptom IDs
    if symptom_ids is None:
        symptom_ids = sorted(symptoms.keys())

    logger.info("Processing %d symptoms: %s", len(symptom_ids), symptom_ids)

    # Load corpus
    logger.info("Loading corpus from %s ...", config.corpus_dir)
    corpus = load_corpus(config.corpus_dir)
    sentences = list(corpus.values())
    logger.info("Loaded %d sentences", len(sentences))

    # Initialize retriever (reuses cached embeddings)
    model_name = config.retrieval_models[0]
    retriever = BiEncoderRetriever(
        model_name=model_name,
        cache_dir=config.output_dir / "embeddings",
    )
    retriever.encode_corpus(sentences)

    # Phase 1: Per-symptom retrieval (Steps A-C)
    out_path = Path(output_dir)
    all_exclusion_indices: set[int] = set()
    symptom_results: dict[int, list[tuple[Sentence, int, float]]] = {}
    symptom_exclusions: dict[int, set[int]] = {}

    for symptom_id in symptom_ids:
        symptom = symptoms[symptom_id]
        candidates, exclusion_set = retrieve_candidates_for_symptom(
            retriever=retriever,
            symptom=symptom,
            sentences=sentences,
            top_k=top_k,
            exclusion_k=exclusion_k,
        )
        symptom_results[symptom_id] = candidates
        symptom_exclusions[symptom_id] = exclusion_set
        all_exclusion_indices |= exclusion_set

    # Phase 2: Generate shared score-0 pool (Section 3.3)
    score0_pool = build_score0_pool(
        sentences=sentences,
        all_exclusion_indices=all_exclusion_indices,
        pool_size=score0_pool_size,
        rng=rng,
    )

    write_score0_pool_tsv(out_path / "score0_pool.tsv", score0_pool)
    logger.info("Wrote shared score-0 pool: %d sentences", len(score0_pool))

    # Phase 3: Per-symptom output with score-0 candidates (Steps D-E)
    summary: dict[int, dict[str, int]] = {}
    for symptom_id in symptom_ids:
        score0_for_symptom = select_score0_for_symptom(
            sentences=sentences,
            exclusion_indices=symptom_exclusions[symptom_id],
            shared_pool=score0_pool,
            count=score0_per_symptom,
            rng=rng,
        )

        filepath = out_path / f"symptom_{symptom_id:02d}_candidates.tsv"
        total = write_symptom_candidates_tsv(
            filepath=filepath,
            retrieval_candidates=symptom_results[symptom_id],
            score0_candidates=score0_for_symptom,
        )

        n_ret = len(symptom_results[symptom_id])
        n_s0 = len(score0_for_symptom)
        summary[symptom_id] = {
            "retrieval": n_ret,
            "score0": n_s0,
            "total": total,
        }
        logger.info(
            "Symptom %02d: wrote %d candidates (%d retrieval + %d score-0) to %s",
            symptom_id, total, n_ret, n_s0, filepath,
        )

    # Phase 4: Create annotation templates
    ann_path = Path(annotations_dir)
    create_annotation_templates(ann_path, symptoms)
    logger.info("Created annotation templates in %s", ann_path)

    # Summary
    logger.info("=" * 60)
    logger.info("CANDIDATE RETRIEVAL COMPLETE")
    logger.info("=" * 60)
    for symptom_id in symptom_ids:
        s = summary[symptom_id]
        logger.info(
            "  Symptom %02d: %d retrieval + %d score-0 = %d total",
            symptom_id, s["retrieval"], s["score0"], s["total"],
        )
    logger.info("  Shared score-0 pool: %d sentences", len(score0_pool))
    logger.info("  Candidates: %s/symptom_*_candidates.tsv", out_path)
    logger.info("  Templates:  %s/symptom_*_examples.json", ann_path)

    return {
        "symptoms": summary,
        "score0_pool_size": len(score0_pool),
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symptom_ids = None
    if args.symptoms:
        symptom_ids = [int(x.strip()) for x in args.symptoms.split(",")]

    run_candidate_retrieval(
        config_path=args.config,
        symptoms_config=args.symptoms_config,
        symptom_ids=symptom_ids,
        output_dir=args.output_dir,
        annotations_dir=args.annotations_dir,
        top_k=args.top_k,
        exclusion_k=args.exclusion_k,
        score0_per_symptom=args.score0_per_symptom,
        score0_pool_size=args.score0_pool_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
