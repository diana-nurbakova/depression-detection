"""Tests for eRisk 2025 T1 dataset loader."""

from pathlib import Path

import pytest

from hipert.data.erisk2025_loader import (
    ERisk2025Qrel,
    build_sentence_lookup,
    build_user_index,
    docid_to_user_id,
    extract_boundary_candidates_cached,
    load_qrels,
    load_qrels_by_query,
    resolve_sentences,
)
from hipert.data.trec_parser import parse_trec_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trec_2025(tmp_path: Path) -> Path:
    """Create a minimal eRisk 2025 format TREC file (full PRE/TEXT/POST)."""
    filepath = tmp_path / "s_0.trec"
    filepath.write_text(
        "<DOC>\n"
        "\t<DOCNO>PgZVTC_0_0</DOCNO>\n"
        "\t<PRE></PRE>\n"
        "\t<TEXT>I feel so tired all the time and I can't focus.</TEXT>\n"
        "\t<POST></POST>\n"
        "</DOC>\n"
        "\n"
        "<DOC>\n"
        "\t<DOCNO>PgZVTC_1_0</DOCNO>\n"
        "\t<PRE></PRE>\n"
        "\t<TEXT>The restaurant had great pasta.</TEXT>\n"
        "\t<POST>We should go back sometime.</POST>\n"
        "</DOC>\n"
        "\n"
        "<DOC>\n"
        "\t<DOCNO>PgZVTC_2_0</DOCNO>\n"
        "\t<PRE>It was a tough day.</PRE>\n"
        "\t<TEXT>I keep forgetting my appointments and losing track of things.</TEXT>\n"
        "\t<POST></POST>\n"
        "</DOC>\n",
        encoding="utf-8",
    )

    filepath2 = tmp_path / "s_1.trec"
    filepath2.write_text(
        "<DOC>\n"
        "\t<DOCNO>abc123_0_0</DOCNO>\n"
        "\t<PRE></PRE>\n"
        "\t<TEXT>I can't sit still in meetings, always fidgeting.</TEXT>\n"
        "\t<POST></POST>\n"
        "</DOC>\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def sample_qrels_majority(tmp_path: Path) -> Path:
    """Create a minimal majority qrel CSV (eRisk 2025 format)."""
    filepath = tmp_path / "qrels_majority_merged.csv"
    filepath.write_text(
        "query,doc_id,relevant\n"
        "19,PgZVTC_0_0,True\n"
        "19,PgZVTC_1_0,False\n"
        "19,PgZVTC_2_0,True\n"
        "11,abc123_0_0,True\n",
        encoding="utf-8",
    )
    return filepath


@pytest.fixture
def sample_qrels_consensus(tmp_path: Path) -> Path:
    """Create a minimal consensus qrel CSV (eRisk 2025 format)."""
    filepath = tmp_path / "qrels_consensus_merged.csv"
    filepath.write_text(
        "query,doc_id,relevant\n"
        "19,PgZVTC_0_0,False\n"         # majority=True, consensus=False -> disagreement
        "19,PgZVTC_1_0,False\n"
        "19,PgZVTC_2_0,True\n"          # majority=True, consensus=True -> agreement
        "11,abc123_0_0,True\n",          # majority=True, consensus=True -> agreement
        encoding="utf-8",
    )
    return filepath


# ---------------------------------------------------------------------------
# TREC parsing tests (uses parse_trec_file for full-context format)
# ---------------------------------------------------------------------------


class TestParseTrecFile:

    def test_parses_all_docs(self, sample_trec_2025: Path) -> None:
        sentences = parse_trec_file(sample_trec_2025 / "s_0.trec")
        assert len(sentences) == 3

    def test_docnos_correct(self, sample_trec_2025: Path) -> None:
        sentences = parse_trec_file(sample_trec_2025 / "s_0.trec")
        docnos = [s.docno for s in sentences]
        assert docnos == ["PgZVTC_0_0", "PgZVTC_1_0", "PgZVTC_2_0"]

    def test_pre_post_content(self, sample_trec_2025: Path) -> None:
        sentences = parse_trec_file(sample_trec_2025 / "s_0.trec")
        # Third sentence has PRE context
        assert "tough day" in sentences[2].pre
        # Second sentence has POST context
        assert "go back" in sentences[1].post

    def test_first_person_detection(self, sample_trec_2025: Path) -> None:
        sentences = parse_trec_file(sample_trec_2025 / "s_0.trec")
        assert sentences[0].has_first_person is True   # "I feel so tired..."
        assert sentences[1].has_first_person is False   # "The restaurant..."
        assert sentences[2].has_first_person is True   # "I keep forgetting..."


# ---------------------------------------------------------------------------
# Qrel loading tests
# ---------------------------------------------------------------------------


class TestLoadQrels:

    def test_loads_all_rows(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert len(qrels) == 4

    def test_parses_query_as_int(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert all(isinstance(q.query, int) for q in qrels)

    def test_parses_relevant_as_bool(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        assert all(isinstance(q.relevant, bool) for q in qrels)

    def test_true_false_parsing(self, sample_qrels_majority: Path) -> None:
        qrels = load_qrels(sample_qrels_majority)
        relevant_count = sum(1 for q in qrels if q.relevant)
        assert relevant_count == 3  # PgZVTC_0_0, PgZVTC_2_0, abc123_0_0

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
        assert len(grouped[11]) == 1


# ---------------------------------------------------------------------------
# Docid to user ID mapping
# ---------------------------------------------------------------------------


class TestDocidToUserId:

    def test_standard_format(self) -> None:
        assert docid_to_user_id("PgZVTC_0_0") == "PgZVTC"

    def test_longer_indices(self) -> None:
        assert docid_to_user_id("abc123_1093_4") == "abc123"

    def test_single_char_user(self) -> None:
        assert docid_to_user_id("X_0_0") == "X"


# ---------------------------------------------------------------------------
# User index and sentence lookup
# ---------------------------------------------------------------------------


class TestBuildUserIndex:

    def test_builds_index(self, sample_trec_2025: Path) -> None:
        index = build_user_index(sample_trec_2025)
        assert "PgZVTC" in index
        assert "abc123" in index
        assert len(index) == 2

    def test_paths_are_valid(self, sample_trec_2025: Path) -> None:
        index = build_user_index(sample_trec_2025)
        for uid, fpath in index.items():
            assert fpath.exists()


class TestBuildSentenceLookup:

    def test_builds_lookup(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(sample_trec_2025)
        assert "PgZVTC_0_0" in lookup
        assert "PgZVTC_1_0" in lookup
        assert "abc123_0_0" in lookup

    def test_targeted_loading(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(
            sample_trec_2025,
            docids={"PgZVTC_0_0", "PgZVTC_2_0"},
        )
        # Should load the file containing PgZVTC, which has 3 sentences
        assert "PgZVTC_0_0" in lookup
        assert "PgZVTC_2_0" in lookup
        # abc123 file should NOT be loaded
        assert "abc123_0_0" not in lookup

    def test_preserves_context(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(sample_trec_2025)
        sent = lookup["PgZVTC_2_0"]
        assert "tough day" in sent.pre
        assert "forgetting" in sent.text


# ---------------------------------------------------------------------------
# Boundary candidate extraction
# ---------------------------------------------------------------------------


class TestExtractBoundaryCandidates:

    def test_disagreement_set(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consensus: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consensus = load_qrels_by_query(sample_qrels_consensus)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consensus, bdi_query=19,
        )

        # PgZVTC_0_0: majority=True, consensus=False -> disagreement
        assert "PgZVTC_0_0" in disagreement
        # PgZVTC_2_0: majority=True, consensus=True -> agreement
        assert "PgZVTC_2_0" in agreement
        # PgZVTC_1_0: majority=False -> neither
        assert "PgZVTC_1_0" not in disagreement
        assert "PgZVTC_1_0" not in agreement

    def test_agreement_set(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consensus: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consensus = load_qrels_by_query(sample_qrels_consensus)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consensus, bdi_query=11,
        )
        assert "abc123_0_0" in agreement
        assert len(disagreement) == 0

    def test_nonexistent_query(
        self,
        sample_qrels_majority: Path,
        sample_qrels_consensus: Path,
    ) -> None:
        majority = load_qrels_by_query(sample_qrels_majority)
        consensus = load_qrels_by_query(sample_qrels_consensus)

        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consensus, bdi_query=99,
        )
        assert len(disagreement) == 0
        assert len(agreement) == 0


# ---------------------------------------------------------------------------
# Sentence resolution
# ---------------------------------------------------------------------------


class TestResolveSentences:

    def test_resolves_existing_docids(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(sample_trec_2025)
        resolved = resolve_sentences(
            ["PgZVTC_0_0", "PgZVTC_2_0"], lookup, require_first_person=False,
        )
        assert len(resolved) == 2

    def test_skips_missing_docids(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(sample_trec_2025)
        resolved = resolve_sentences(
            ["PgZVTC_0_0", "nonexistent_99_0"], lookup, require_first_person=False,
        )
        assert len(resolved) == 1

    def test_first_person_filter(self, sample_trec_2025: Path) -> None:
        lookup = build_sentence_lookup(sample_trec_2025)
        # PgZVTC_1_0 is "The restaurant had great pasta" - no first person
        resolved = resolve_sentences(
            ["PgZVTC_0_0", "PgZVTC_1_0"], lookup, require_first_person=True,
        )
        assert len(resolved) == 1
        assert resolved[0].docno == "PgZVTC_0_0"


# ---------------------------------------------------------------------------
# Real data tests
# ---------------------------------------------------------------------------


class TestRealData:
    """Tests against real eRisk 2025 data. Skipped if not available."""

    MAJORITY_PATH = Path("data/eRisk-2025/eRisk25-datasets/t1-depression-symptom-ranking/qrels_majority_merged.csv")
    CONSENSUS_PATH = Path("data/eRisk-2025/eRisk25-datasets/t1-depression-symptom-ranking/qrels_consensus_merged.csv")
    TREC_DIR = Path("data/eRisk-2025/eRisk25-datasets/t1-depression-symptom-ranking/erisk25-t1-dataset/erisk25-t1-dataset")

    def test_real_qrels_load(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2025 data not available")
        qrels = load_qrels(self.MAJORITY_PATH)
        assert len(qrels) > 10000

    def test_real_21_queries(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2025 data not available")
        grouped = load_qrels_by_query(self.MAJORITY_PATH)
        assert len(grouped) == 21

    def test_real_trec_files_parse(self) -> None:
        if not self.TREC_DIR.exists():
            pytest.skip("eRisk 2025 TREC files not available")
        first_file = self.TREC_DIR / "s_0.trec"
        if not first_file.exists():
            pytest.skip("s_0.trec not found")
        sentences = parse_trec_file(first_file)
        assert len(sentences) > 0
        # Should have PRE/TEXT/POST (not simple format)
        assert any(s.pre or s.post for s in sentences)

    def test_real_user_index(self) -> None:
        if not self.TREC_DIR.exists():
            pytest.skip("eRisk 2025 TREC files not available")
        index = build_user_index(self.TREC_DIR)
        assert len(index) > 6000

    def test_real_boundary_extraction(self) -> None:
        if not self.MAJORITY_PATH.exists():
            pytest.skip("eRisk 2025 data not available")
        majority = load_qrels_by_query(self.MAJORITY_PATH)
        consensus = load_qrels_by_query(self.CONSENSUS_PATH)
        disagreement, agreement = extract_boundary_candidates_cached(
            majority, consensus, bdi_query=19,
        )
        assert len(disagreement) > 0 or len(agreement) > 0
