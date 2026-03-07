"""Parse eRisk TREC-format files into Sentence objects.

Supports two TREC formats:

**eRisk 2026 Task 3** (full context triplet):

    <DOC>
        <DOCNO>userId_contextId_sentIdx</DOCNO>
        <PRE>previous sentence text</PRE>
        <TEXT>target sentence text</TEXT>
        <POST>following sentence text</POST>
    </DOC>

**eRisk 2023 Task 1** (DOCNO + TEXT only, no PRE/POST):

    <DOC>
        <DOCNO>s_userId_postId_sentIdx</DOCNO>
        <TEXT>sentence text</TEXT>
    </DOC>

Tags may be tab-indented. PRE and POST may be empty.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from hipert.models import Sentence

# Regex to extract one <DOC> block with full context (eRisk 2026).
# Uses DOTALL so . matches newlines within blocks.
_DOC_RE = re.compile(
    r"<DOC>\s*"
    r"<DOCNO>\s*(.*?)\s*</DOCNO>\s*"
    r"<PRE>(.*?)</PRE>\s*"
    r"<TEXT>(.*?)</TEXT>\s*"
    r"<POST>(.*?)</POST>\s*"
    r"</DOC>",
    re.DOTALL,
)

# Regex for simplified format with DOCNO + TEXT only (eRisk 2023).
_DOC_SIMPLE_RE = re.compile(
    r"<DOC>\s*"
    r"<DOCNO>\s*(.*?)\s*</DOCNO>\s*"
    r"<TEXT>(.*?)</TEXT>\s*"
    r"</DOC>",
    re.DOTALL,
)


def parse_trec_file(filepath: Path) -> list[Sentence]:
    """Parse a single .trec file into a list of Sentence objects.

    Args:
        filepath: Path to the .trec file.

    Returns:
        List of Sentence objects parsed from the file.
    """
    file_id = filepath.stem  # e.g. "s_0"

    text = filepath.read_text(encoding="utf-8", errors="replace")

    sentences: list[Sentence] = []
    for m in _DOC_RE.finditer(text):
        docno = m.group(1).strip()
        pre = m.group(2).strip()
        target = m.group(3).strip()
        post = m.group(4).strip()

        if not docno or not target:
            continue

        sentences.append(Sentence(
            docno=docno,
            pre=pre,
            text=target,
            post=post,
            file_id=file_id,
        ))

    return sentences


def parse_trec_file_simple(filepath: Path) -> list[Sentence]:
    """Parse a TREC file with DOCNO + TEXT only (eRisk 2023 format).

    Returns Sentence objects with empty ``pre`` and ``post`` fields since
    the eRisk 2023 format does not include context sentences.

    Args:
        filepath: Path to the .trec file.

    Returns:
        List of Sentence objects parsed from the file.
    """
    file_id = filepath.stem

    text = filepath.read_text(encoding="utf-8", errors="replace")

    sentences: list[Sentence] = []
    for m in _DOC_SIMPLE_RE.finditer(text):
        docno = m.group(1).strip()
        target = m.group(2).strip()

        if not docno or not target:
            continue

        sentences.append(Sentence(
            docno=docno,
            pre="",
            text=target,
            post="",
            file_id=file_id,
        ))

    return sentences


def iter_trec_files(corpus_dir: Path) -> Iterator[Path]:
    """Yield all .trec files in the corpus directory, sorted by name."""
    files = sorted(corpus_dir.glob("*.trec"))
    yield from files


def iter_sentences(corpus_dir: Path) -> Iterator[Sentence]:
    """Stream all sentences from all .trec files in the corpus directory."""
    for trec_path in iter_trec_files(corpus_dir):
        yield from parse_trec_file(trec_path)
