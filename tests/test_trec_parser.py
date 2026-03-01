"""Tests for TREC file parser."""

from pathlib import Path
from textwrap import dedent

import pytest

from hipert.data.trec_parser import parse_trec_file


@pytest.fixture
def trec_file(tmp_path: Path, sample_trec_content: str) -> Path:
    filepath = tmp_path / "s_test.trec"
    filepath.write_text(sample_trec_content, encoding="utf-8")
    return filepath


def test_parse_trec_file_basic(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    assert len(sentences) == 3


def test_parse_trec_file_docnos(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    docnos = [s.docno for s in sentences]
    assert docnos == ["testuser_0_0", "testuser_0_1", "testuser_1_0"]


def test_parse_trec_file_text(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    assert sentences[0].text == "I can never focus on anything for more than five minutes."
    assert sentences[1].text == "It's really frustrating."
    assert sentences[2].text == "The weather is nice today."


def test_parse_trec_file_context(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    # First sentence has empty PRE
    assert sentences[0].pre == ""
    assert sentences[0].post == "It's really frustrating."
    # Second sentence has PRE from first
    assert sentences[1].pre == "I can never focus on anything for more than five minutes."
    assert sentences[1].post == ""


def test_parse_trec_file_id(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    assert all(s.file_id == "s_test" for s in sentences)


def test_first_person_detection(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    assert sentences[0].has_first_person is True   # "I can never focus"
    assert sentences[1].has_first_person is False   # "It's really frustrating"
    assert sentences[2].has_first_person is False   # "The weather is nice"


def test_user_id_extraction(trec_file: Path) -> None:
    sentences = parse_trec_file(trec_file)
    assert sentences[0].user_id == "testuser"
    assert sentences[2].user_id == "testuser"


def test_parse_real_trec_file(corpus_dir: Path) -> None:
    """Test parsing of an actual TREC file from the corpus."""
    first_file = corpus_dir / "s_0.trec"
    if not first_file.exists():
        pytest.skip("Corpus data not available")

    sentences = parse_trec_file(first_file)
    assert len(sentences) > 0
    assert all(s.docno for s in sentences)
    assert all(s.text for s in sentences)
    assert all(s.file_id == "s_0" for s in sentences)
