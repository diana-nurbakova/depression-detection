"""Tests for eRisk 2023 dataset loader and simple TREC parser."""

from pathlib import Path

import pytest

from hipert.data.erisk2023_loader import (
    ERisk2023Qrel,
    build_sentence_lookup,
    docid_to_user_file,
    extract_boundary_candidates_cached,
    load_qrels,
    load_qrels_by_query,
    resolve_sentences,
)
from hipert.data.trec_parser import parse_trec_file_simple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_simple_trec(tmp_path: Path) -> Path:
    """Create a minimal eRisk 2023 format TREC file."""
    filepath = tmp_path / "s_0.trec"
    filepath.write_text(
        "<DOC>\n"
        "    <DOCNO>s_0_0_0</DOCNO>\n"
        "    <TEXT>I feel so tired all the time and I can't focus.</TEXT>\n"
        "</DOC>\n"
        "\n"
        "<DOC>\n"
        "    <DOCNO>s_0_1_0</DOCNO>\n"
        "    <TEXT>The restaurant had great pasta.</TEXT>\n"
        "</DOC>\n"
        "\n"
        "<DOC>\n"
        "    <DOCNO>s_0_2_0</DOCNO>\n"
        "    <TEXT>I keep forgetting my appointments and losing track of things.</TEXT>\n"
        "</DOC>\n",
        encoding="utf-8",
    )
    return filepath


@pytest.fixture
def sample_qrels_majority(tmp_path: Path) -> Path:
    """Create a minimal majority qrel CSV."""
    filepath = tmp_path / "g_qrels_majority_2.csv"
    filepath.write_text(
        "query,q0,docid,rel\n"
        "19,0,s_0_0_0,1\n"
        "19,0,s_0_1_0,0\n"
        "19,0,s_0_2_0,1\n"
        "11,0,s_1_0_0,1\n"
        "11,0,s_1_1_0,0\n",
        encoding="utf-8",
    )
    return filepath


@pytest.fixture
def sample_qrels_consenso(tmp_path: Path) -> Path:
    """Create a minimal consensus qrel CSV."""
    filepath = tmp_path / "g_rels_consenso.csv"
    filepath.write_text(
        "query,q0,docid,rel\n"
        "19,0,s_0_0_0,0\n"     # majority=1, consensus=0 -> disagreement
        "19,0,s_0_1_0,0\n"
        "19,0,s_0_2_0,1\n"     # majority=1, consensus=1 -> agreement
        "11,0,s_1_0_0,1\n"     # majority=1, consensus=1 -> agreement
        "11,0,s_1_1_0,0\n",
        encoding="utf-8",
    )
    return filepath


# ---------------------------------------------------------------------------
# parse_trec_file_simple tests
# ---------------------------------------------------------------------------


class TestParseSimpleTrecFile:

    def test_parses_all_docs(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        assert len(sentences) == 3

    def test_docnos_correct(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        docnos = [s.docno for s in sentences]
        assert docnos == ["s_0_0_0", "s_0_1_0", "s_0_2_0"]

    def test_text_content(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        assert "tired" in sentences[0].text
        assert "pasta" in sentences[1].text
        assert "forgetting" in sentences[2].text

    def test_empty_pre_post(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        for sent in sentences:
            assert sent.pre == ""
            assert sent.post == ""

    def test_file_id(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        assert all(s.file_id == "s_0" for s in sentences)

    def test_first_person_detection(self, sample_simple_trec: Path) -> None:
        sentences = parse_trec_file_simple(sample_simple_trec)
        assert sentences[0].has_first_person is True   # "I feel so tired..."
        assert sentences[1].has_first_person is False   # "The restaurant..."
        assert sentences[2].has_first_person is True   # "I keep forgetting..."


# ---------------------------------------------------------------------------
# Qrel loading tests
# ---------------------------------------------------------------------------


class TestLoadQrels:

    def test_loads_all_rows(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert len(qrels) == 5

    def test_parses_query_as_int(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert all(isinstance(q.query, int) for q in qrels)

    def test_parses_rel_as_int(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert all(q.rel in (0, 1) for q in qrels)

    def test_query_values(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        queries = {q.query for q in qrels}
        assert queries == {11, 19}


class TestLoadQrelsByQuery:

    def test_groups_correctly(self, sample_qrels_majority: Path) -> None:
        grouped = load_qrels_by_query(sample_qrels_majority)
        assert 19 in grouped
        assert 11 in grouped
        assert len(grouped[19]) == 3
        assert len(grouped[11]) == 2


# ---------------------------------------------------------------------------
# Docid to user file mapping
# ---------------------------------------------------------------------------


class TestDocidToUserFile:

    def test_standard_format(self) -> None:
        assert docid_to_user_file("s_405_1279_15") == "s_405.trec"

    def test_short_ids(self) -> None:
        assert docid_to_user_file("s_0_0_0") == "s_0.trec"

    def test_long_user_id(self) -> None:
        assert docid_to_user_file("s_1234_567_89") == "s_1234.trec"


# ---------------------------------------------------------------------------
# Boundary candidate extraction
# ---------------------------------------------------------------------------


class TestExtractBoundaryCandidates:

    def test_disagreement_set(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consenso: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consenso = load_qrels_by_query(sample_qrels_consenso)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consenso, bdi_query=19,
        )

        # s_0_0_0: majority=1, consensus=0 -> disagreement
        assert "s_0_0_0" in disagreement
        # s_0_2_0: majority=1, consensus=1 -> agreement
        assert "s_0_2_0" in agreement
        # s_0_1_0: majority=0 -> neither
        assert "s_0_1_0" not in disagreement
        assert "s_0_1_0" not in agreement

    def test_agreement_set(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consenso: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consenso = load_qrels_by_query(sample_qrels_consenso)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consenso, bdi_query=11,
        )
        # s_1_0_0: majority=1, consensus=1 -> agreement
        assert "s_1_0_0" in agreement
        assert len(disagreement) == 0

    def test_nonexistent_query(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consenso: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consenso = load_qrels_by_query(sample_qrels_consenso)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consenso, bdi_query=99,
        )
        assert len(disagreement) == 0
        assert len(agreement) == 0


# ---------------------------------------------------------------------------
# Sentence lookup and resolution
# ---------------------------------------------------------------------------


class TestBuildSentenceLookup:

    def test_builds_lookup(self, sample_simple_trec: Path) -> None:
        lookup = build_sentence_lookup(sample_simple_trec.parent)
        assert "s_0_0_0" in lookup
        assert "s_0_1_0" in lookup
        assert "s_0_2_0" in lookup

    def test_targeted_loading(self, sample_simple_trec: Path) -> None:
        lookup = build_sentence_lookup(
            sample_simple_trec.parent,
            docids={"s_0_0_0", "s_0_2_0"},
        )
        # Should load the file containing these docids
        assert "s_0_0_0" in lookup


class TestResolveSentences:

    def test_resolves_existing_docids(self, sample_simple_trec: Path) -> None:
        lookup = build_sentence_lookup(sample_simple_trec.parent)
        resolved = resolve_sentences(
            ["s_0_0_0", "s_0_2_0"], lookup, require_first_person=False,
        )
        assert len(resolved) == 2

    def test_skips_missing_docids(self, sample_simple_trec: Path) -> None:
        lookup = build_sentence_lookup(sample_simple_trec.parent)
        resolved = resolve_sentences(
            ["s_0_0_0", "s_nonexistent_99_0"], lookup, require_first_person=False,
        )
        assert len(resolved) == 1

    def test_first_person_filter(self, sample_simple_trec: Path) -> None:
        lookup = build_sentence_lookup(sample_simple_trec.parent)
        # s_0_1_0 is "The restaurant had great pasta" - no first person
        resolved = resolve_sentences(
            ["s_0_0_0", "s_0_1_0"], lookup, require_first_person=True,
        )
        assert len(resolved) == 1
        assert resolved[0].docno == "s_0_0_0"


# ---------------------------------------------------------------------------
# Real data tests
# ---------------------------------------------------------------------------


class TestRealData:
    """Tests against real eRisk 2023 data. Skipped if not available."""

    MAJORITY_PATH = Path("data/eRisk2023_T1/g_qrels_majority_2.csv")
    CONSENSO_PATH = Path("data/eRisk2023_T1/g_rels_consenso.csv")
    TREC_DIR = Path("data/eRisk2023_T1/new_data")

    def test_real_qrels_load(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2023 data not available")
        qrels = load_qrels(self.MAJORITY_PATH)
        assert len(qrels) > 1000

    def test_real_21_queries(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2023 data not available")
        grouped = load_qrels_by_query(self.MAJORITY_PATH)
        assert len(grouped) == 21

    def test_real_trec_files_parse(self) -> None:
        if not self.TREC_DIR.exists():
            pytest.skip("eRisk 2023 TREC files not available")
        first_file = self.TREC_DIR / "s_0.trec"
        if not first_file.exists():
            pytest.skip("s_0.trec not found")
        sentences = parse_trec_file_simple(first_file)
        assert len(sentences) > 0

    def test_real_boundary_extraction(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2023 data not available")
        majority = load_qrels_by_query(self.MAJORITY_PATH)
        consenso = load_qrels_by_query(self.CONSENSO_PATH)
        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consenso, bdi_query=19,
        )
        # BDI-19 (Concentration) should have substantial candidates
        assert len(disagreement) > 0 or len(agreement) > 0
