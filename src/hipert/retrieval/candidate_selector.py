"""Candidate selection pipeline.

Orchestrates: query expansion → bi-encoder retrieval → first-person
filter → keyword boost → top-K selection per symptom.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from hipert.models import CandidateScore, Sentence, SymptomDefinition
from hipert.retrieval.encoder import BiEncoderRetriever
from hipert.retrieval.filters import apply_first_person_filter, apply_keyword_boost
from hipert.retrieval.query_expansion import build_expanded_query

logger = logging.getLogger(__name__)


class CandidateSelector:
    """Orchestrates the full candidate selection pipeline."""

    def __init__(
        self,
        retriever: BiEncoderRetriever,
        symptoms: dict[int, SymptomDefinition],
        keyword_clusters: dict[str, list[str]],
        top_k: int = 5000,
        keyword_boost: float = 0.05,
        first_person_filter: bool = True,
        output_dir: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.symptoms = symptoms
        self.keyword_clusters = keyword_clusters
        self.top_k = top_k
        self.keyword_boost = keyword_boost
        self.first_person_filter = first_person_filter
        self.output_dir = output_dir

    def select_candidates(
        self,
        sentences: list[Sentence],
        symptom_ids: list[int] | None = None,
    ) -> dict[int, list[CandidateScore]]:
        """Run candidate selection for specified symptoms.

        Args:
            sentences: All corpus sentences.
            symptom_ids: Which symptoms to select for.
                If None, all 18 symptoms.

        Returns:
            Dictionary mapping symptom_id to list of CandidateScore.
        """
        if symptom_ids is None:
            symptom_ids = sorted(self.symptoms.keys())

        # Ensure corpus is encoded
        if not self.retriever.is_loaded:
            self.retriever.encode_corpus(sentences)

        results: dict[int, list[CandidateScore]] = {}

        for symptom_id in tqdm(
            symptom_ids, desc="Selecting candidates", unit="symptom",
        ):
            symptom = self.symptoms[symptom_id]
            candidates = self._select_for_symptom(
                symptom, sentences,
            )
            results[symptom_id] = candidates

            logger.info(
                "Symptom %d: %d candidates selected (top_k=%d)",
                symptom_id, len(candidates), self.top_k,
            )

            # Save to disk if output_dir is set
            if self.output_dir is not None:
                self._save_candidates(symptom_id, candidates)

        return results

    def _select_for_symptom(
        self,
        symptom: SymptomDefinition,
        sentences: list[Sentence],
    ) -> list[CandidateScore]:
        """Select candidates for a single symptom."""
        # Build expanded query
        query = build_expanded_query(symptom)

        # Retrieve top-K * 2 (buffer for filtering)
        retrieve_k = min(self.top_k * 2, len(sentences))
        raw_results = self.retriever.retrieve(query, top_k=retrieve_k)

        # Build CandidateScore objects
        candidates = [
            CandidateScore(
                sentence=sentences[idx],
                symptom_id=symptom.item_number,
                retrieval_score=score,
            )
            for idx, score in raw_results
        ]

        # Apply first-person filter
        if self.first_person_filter:
            before_count = len(candidates)
            candidates = apply_first_person_filter(candidates)
            logger.debug(
                "Symptom %d first-person filter: %d -> %d (%.1f%% retained)",
                symptom.item_number, before_count, len(candidates),
                100 * len(candidates) / before_count if before_count else 0,
            )

        # Apply keyword boost
        keywords = self._get_keywords_for_symptom(symptom)
        if keywords:
            candidates = apply_keyword_boost(
                candidates, keywords, self.keyword_boost,
            )

        # Sort by combined score and take top-K
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        candidates = candidates[:self.top_k]

        return candidates

    def _get_keywords_for_symptom(
        self,
        symptom: SymptomDefinition,
    ) -> list[str]:
        """Get combined keywords: symptom-specific + cluster-level."""
        keywords = list(symptom.keywords)

        # Add general ADHD keywords
        keywords.extend(self.keyword_clusters.get("general_adhd", []))

        return keywords

    def _save_candidates(
        self,
        symptom_id: int,
        candidates: list[CandidateScore],
    ) -> None:
        """Save candidates to a JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"symptom_{symptom_id}.json"

        data = [
            {
                "docno": c.sentence.docno,
                "file_id": c.sentence.file_id,
                "text": c.sentence.text,
                "pre": c.sentence.pre,
                "post": c.sentence.post,
                "retrieval_score": round(c.retrieval_score, 6),
                "keyword_boost": round(c.keyword_boost, 6),
                "combined_score": round(c.combined_score, 6),
            }
            for c in candidates
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(
            "Saved %d candidates for symptom %d to %s",
            len(candidates), symptom_id, filepath,
        )


def load_candidates(
    filepath: Path,
    symptom_id: int,
) -> list[CandidateScore]:
    """Load candidates from a previously saved JSON file.

    Args:
        filepath: Path to the candidates JSON file.
        symptom_id: The symptom ID these candidates belong to.

    Returns:
        List of CandidateScore objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        CandidateScore(
            sentence=Sentence(
                docno=item["docno"],
                pre=item.get("pre", ""),
                text=item["text"],
                post=item.get("post", ""),
                file_id=item.get("file_id", ""),
            ),
            symptom_id=symptom_id,
            retrieval_score=item.get("retrieval_score", 0.0),
            keyword_boost=item.get("keyword_boost", 0.0),
        )
        for item in data
    ]
