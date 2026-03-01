"""Shared test fixtures."""

from pathlib import Path

import pytest

from hipert.models import LLMOutput, Sentence


@pytest.fixture
def sample_sentence() -> Sentence:
    return Sentence(
        docno="user1_0_0",
        pre="I was in a meeting today.",
        text="I kept zoning out and couldn't focus on what my boss was saying.",
        post="She had to repeat herself twice.",
        file_id="s_0",
    )


@pytest.fixture
def sample_llm_output() -> LLMOutput:
    return LLMOutput(
        symptom_match="YES",
        self_reference="DIRECT",
        detail_level="MEDIUM",
        confounders="NONE",
        score=2,
        confidence=4,
        reasoning="Writer describes personal inattention during conversation.",
        raw_text="SYMPTOM_MATCH: YES\nSELF_REFERENCE: DIRECT\n...",
    )


@pytest.fixture
def corpus_dir() -> Path:
    """Path to the actual TREC test data."""
    return Path(
        "data/eRisk-2026/"
        "task3-adhd-symptom-ranking-20260204T094934Z-3-001/"
        "task3-adhd-symptom-ranking/output_trec_files/output_trec_files"
    )


@pytest.fixture
def sample_trec_content() -> str:
    return """\
<DOC>
    <DOCNO>testuser_0_0</DOCNO>
    <PRE></PRE>
    <TEXT>I can never focus on anything for more than five minutes.</TEXT>
    <POST>It's really frustrating.</POST>
</DOC>

<DOC>
    <DOCNO>testuser_0_1</DOCNO>
    <PRE>I can never focus on anything for more than five minutes.</PRE>
    <TEXT>It's really frustrating.</TEXT>
    <POST></POST>
</DOC>

<DOC>
    <DOCNO>testuser_1_0</DOCNO>
    <PRE></PRE>
    <TEXT>The weather is nice today.</TEXT>
    <POST></POST>
</DOC>
"""
