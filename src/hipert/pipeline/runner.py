"""End-to-end pipeline orchestration.

Ties together: corpus loading → retrieval → scoring → output.
Supports resumption from checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from hipert.config import PipelineConfig
from hipert.data.corpus import corpus_stats, load_corpus
from hipert.data.trec_writer import write_trec_rankings
from hipert.models import CandidateScore, Sentence, SymptomDefinition
from hipert.pipeline.checkpoint import CheckpointManager, ScoringProgress
from hipert.retrieval.candidate_selector import CandidateSelector, load_candidates
from hipert.retrieval.encoder import BiEncoderRetriever
from hipert.scoring.llm_client import LLMClient, make_llm_client
from hipert.scoring.prompt_builder import PromptBuilder
from hipert.scoring.scorer import ScoringCascade
from hipert.utils.logging import LLMCallLogger, PipelineEventLogger, setup_logging

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full HiPerT-ADHD pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Setup logging
        self.pipeline_logger, self.llm_logger = setup_logging(
            config.log_dir, config.log_level,
        )

        # Setup checkpoint manager
        self.checkpoint = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            silver_labels_dir=config.output_dir / "silver_labels",
        )

        # Symptom lookup
        self.symptoms: dict[int, SymptomDefinition] = {
            s.item_number: s for s in config.symptoms
        }

        self.pipeline_logger.log(
            "pipeline_init",
            symptoms_count=len(self.symptoms),
            corpus_dir=str(config.corpus_dir),
        )

    def run_parse(self, stats_only: bool = False) -> dict:
        """Parse corpus and display statistics."""
        self.pipeline_logger.log("parse_start", stats_only=stats_only)

        stats = corpus_stats(self.config.corpus_dir)

        self.pipeline_logger.log("parse_complete", **stats)
        return stats

    def run_retrieve(
        self,
        symptom_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> dict[int, list[CandidateScore]]:
        """Run bi-encoder retrieval and candidate selection."""
        self.pipeline_logger.log(
            "retrieve_start",
            symptom_ids=symptom_ids,
            top_k=top_k,
        )

        # Load corpus
        logger.info("Loading corpus...")
        corpus = load_corpus(self.config.corpus_dir)
        sentences = list(corpus.values())

        # Initialize retriever
        model_name = self.config.retrieval_models[0]
        retriever = BiEncoderRetriever(
            model_name=model_name,
            cache_dir=self.config.output_dir / "embeddings",
        )

        # Initialize candidate selector
        selector = CandidateSelector(
            retriever=retriever,
            symptoms=self.symptoms,
            keyword_clusters=self.config.keyword_clusters,
            top_k=top_k or self.config.retrieval_top_k,
            keyword_boost=self.config.keyword_boost,
            first_person_filter=self.config.first_person_filter,
            output_dir=self.config.output_dir / "candidates",
        )

        # Run selection
        results = selector.select_candidates(sentences, symptom_ids)

        self.pipeline_logger.log(
            "retrieve_complete",
            symptoms_processed=list(results.keys()),
            candidates_per_symptom={
                k: len(v) for k, v in results.items()
            },
        )

        return results

    def run_score(
        self,
        symptom_ids: list[int] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        resume: bool = True,
    ) -> None:
        """Run LLM scoring cascade on candidates.

        Args:
            symptom_ids: Which symptoms to score (default: all).
            limit: Max sentences to score per symptom (for testing).
            dry_run: Build prompts but don't call LLM.
            resume: Resume from checkpoint.
        """
        if symptom_ids is None:
            symptom_ids = sorted(self.symptoms.keys())

        self.pipeline_logger.log(
            "score_start",
            symptom_ids=symptom_ids,
            limit=limit,
            dry_run=dry_run,
            resume=resume,
        )

        # Create LLM clients
        llama_client = make_llm_client(self.config.primary_provider)
        llama_client.max_retries = self.config.llm_max_retries
        llama_client.rate_limit_delay = self.config.llm_rate_limit_delay
        llama_client.timeout = self.config.llm_read_timeout

        gpt_client: Optional[LLMClient] = None
        if self.config.escalation_provider.base_url:
            gpt_client = make_llm_client(self.config.escalation_provider)
            gpt_client.max_retries = self.config.llm_max_retries

        # Create prompt builder
        prompt_builder = PromptBuilder(
            symptoms=self.symptoms,
            examples={},  # TODO: load few-shot examples
        )

        # Create scoring cascade
        cascade = ScoringCascade(
            llama_client=llama_client,
            gpt_client=gpt_client,
            prompt_builder=prompt_builder,
            llm_logger=self.llm_logger,
            escalation_max_rate=self.config.escalation_max_rate,
        )

        # Score each symptom
        for symptom_id in symptom_ids:
            self._score_symptom(
                cascade=cascade,
                symptom_id=symptom_id,
                limit=limit,
                dry_run=dry_run,
                resume=resume,
            )

        self.pipeline_logger.log(
            "score_complete",
            cascade_stats=cascade.stats,
        )

        logger.info("Scoring complete. Stats: %s", cascade.stats)

    def _score_symptom(
        self,
        cascade: ScoringCascade,
        symptom_id: int,
        limit: int | None,
        dry_run: bool,
        resume: bool,
    ) -> None:
        """Score all candidates for a single symptom."""
        # Load candidates
        candidates_path = (
            self.config.output_dir / "candidates" / f"symptom_{symptom_id}.json"
        )
        if not candidates_path.exists():
            logger.warning(
                "No candidates file for symptom %d at %s. "
                "Run 'retrieve' first.",
                symptom_id, candidates_path,
            )
            return

        candidates = load_candidates(candidates_path, symptom_id)

        # Apply limit
        if limit is not None:
            candidates = candidates[:limit]

        # Check existing progress for resume
        already_scored: set[str] = set()
        if resume:
            already_scored = self.checkpoint.get_scored_ids(symptom_id)
            if already_scored:
                logger.info(
                    "Symptom %d: resuming, %d already scored",
                    symptom_id, len(already_scored),
                )

        # Filter out already-scored candidates
        to_score = [
            c for c in candidates
            if c.sentence.docno not in already_scored
        ]

        if not to_score:
            logger.info("Symptom %d: all candidates already scored", symptom_id)
            return

        logger.info(
            "Symptom %d: scoring %d candidates (%d total, %d already done)",
            symptom_id, len(to_score), len(candidates), len(already_scored),
        )

        self.pipeline_logger.log(
            "symptom_score_start",
            symptom_id=symptom_id,
            total_candidates=len(candidates),
            to_score=len(to_score),
            already_scored=len(already_scored),
        )

        if dry_run:
            # Just build and display prompts
            prompt_builder = cascade.prompt_builder if hasattr(cascade, 'prompt_builder') else PromptBuilder(self.symptoms)
            for c in to_score[:3]:
                prompt = prompt_builder.build_user_prompt(
                    symptom_id, c.sentence,
                )
                logger.info(
                    "DRY RUN — Symptom %d, Sentence %s:\n%s",
                    symptom_id, c.sentence.docno, prompt[:500],
                )
            logger.info("Dry run: showed %d sample prompts", min(3, len(to_score)))
            return

        # Score with progress bar
        progress = tqdm(
            to_score,
            desc=f"Scoring symptom {symptom_id}",
            unit="sent",
        )

        for candidate in progress:
            try:
                result = cascade.score_single(
                    candidate.sentence, symptom_id,
                )
                # Save incrementally
                self.checkpoint.append_result(symptom_id, result)

                progress.set_postfix(
                    score=result.final_label,
                    esc=f"{cascade.escalation_rate:.0%}",
                )

            except Exception as e:
                logger.error(
                    "Failed to score sentence %s for symptom %d: %s",
                    candidate.sentence.docno, symptom_id, e,
                )
                self.pipeline_logger.log(
                    "score_error",
                    symptom_id=symptom_id,
                    sentence_id=candidate.sentence.docno,
                    error=str(e),
                )

        self.pipeline_logger.log(
            "symptom_score_complete",
            symptom_id=symptom_id,
            cascade_stats=cascade.stats,
        )

    def run_output(self, top_n: int = 1000) -> None:
        """Generate TREC-format rankings from scored results."""
        self.pipeline_logger.log("output_start", top_n=top_n)

        rankings_dir = self.config.output_dir / "rankings"
        rankings_dir.mkdir(parents=True, exist_ok=True)

        write_trec_rankings(
            silver_labels_dir=self.config.output_dir / "silver_labels",
            output_dir=rankings_dir,
            top_n=top_n,
        )

        self.pipeline_logger.log("output_complete")

    def close(self) -> None:
        """Clean up resources."""
        self.pipeline_logger.close()
        self.llm_logger.close()
