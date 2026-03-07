"""Corpus-level operations: loading, statistics, and indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from hipert.data.trec_parser import iter_sentences, iter_trec_files, parse_trec_file
from hipert.models import Sentence

logger = logging.getLogger(__name__)


def load_corpus(corpus_dir: Path, show_progress: bool = True) -> dict[str, Sentence]:
    """Load the entire corpus into memory, keyed by DOCNO.

    Args:
        corpus_dir: Path to the directory containing .trec files.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Dictionary mapping DOCNO to Sentence objects.
    """
    corpus: dict[str, Sentence] = {}
    trec_files = list(iter_trec_files(corpus_dir))

    iterator: Iterator[Path] = trec_files
    if show_progress:
        iterator = tqdm(trec_files, desc="Loading corpus", unit="file")

    for trec_path in iterator:
        sentences = parse_trec_file(trec_path)
        for sent in sentences:
            corpus[sent.docno] = sent

    logger.info(
        "Corpus loaded: %d sentences from %d files",
        len(corpus), len(trec_files),
    )
    return corpus


def corpus_stats(corpus_dir: Path) -> dict:
    """Compute corpus statistics without loading everything into memory.

    Returns:
        Dictionary with corpus statistics.
    """
    trec_files = list(iter_trec_files(corpus_dir))
    total_sentences = 0
    sentences_per_file: list[int] = []
    first_person_count = 0
    sample_docnos: list[str] = []
    total_text_chars = 0

    for trec_path in tqdm(trec_files, desc="Scanning corpus", unit="file"):
        sentences = parse_trec_file(trec_path)
        count = len(sentences)
        sentences_per_file.append(count)
        total_sentences += count

        for sent in sentences:
            total_text_chars += len(sent.text)
            if sent.has_first_person:
                first_person_count += 1

        # Collect some sample DOCNOs
        if len(sample_docnos) < 10 and sentences:
            sample_docnos.append(sentences[0].docno)

    avg_per_file = (
        sum(sentences_per_file) / len(sentences_per_file)
        if sentences_per_file else 0
    )
    avg_text_len = total_text_chars / total_sentences if total_sentences else 0

    return {
        "total_files": len(trec_files),
        "total_sentences": total_sentences,
        "avg_sentences_per_file": round(avg_per_file, 1),
        "min_sentences_per_file": min(sentences_per_file) if sentences_per_file else 0,
        "max_sentences_per_file": max(sentences_per_file) if sentences_per_file else 0,
        "first_person_count": first_person_count,
        "first_person_pct": round(
            100 * first_person_count / total_sentences, 1,
        ) if total_sentences else 0,
        "avg_text_length_chars": round(avg_text_len, 1),
        "sample_docnos": sample_docnos,
    }
