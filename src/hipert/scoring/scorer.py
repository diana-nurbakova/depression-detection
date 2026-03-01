"""Scoring cascade orchestrator.

Manages the full Llama → (optional GPT escalation) → resolution flow
with comprehensive logging of every LLM call.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from hipert.models import CandidateScore, LLMOutput, ScoringResult, Sentence
from hipert.scoring.escalation import check_escalation
from hipert.scoring.llm_client import LLMClient
from hipert.scoring.prompt_builder import PromptBuilder
from hipert.scoring.resolution import resolve_label
from hipert.scoring.response_parser import parse_llm_response
from hipert.utils.logging import LLMCallLogger

logger = logging.getLogger(__name__)


class ScoringCascade:
    """Orchestrates the Llama-first scoring cascade with GPT escalation."""

    def __init__(
        self,
        llama_client: LLMClient,
        gpt_client: Optional[LLMClient],
        prompt_builder: PromptBuilder,
        llm_logger: LLMCallLogger,
        escalation_max_rate: float = 0.40,
    ) -> None:
        self.llama_client = llama_client
        self.gpt_client = gpt_client
        self.prompt_builder = prompt_builder
        self.llm_logger = llm_logger
        self.escalation_max_rate = escalation_max_rate

        # Register system prompt hash
        self._system_prompt = PromptBuilder.get_system_prompt()
        self._system_prompt_hash = self.llm_logger.log_system_prompt(
            self._system_prompt,
        )

        # Track escalation rate
        self._total_scored = 0
        self._total_escalated = 0

    def score_single(
        self,
        sentence: Sentence,
        symptom_id: int,
    ) -> ScoringResult:
        """Score a single sentence for one symptom.

        Flow:
        1. Build prompt and call Llama
        2. Parse structured response
        3. Check escalation triggers
        4. If escalated and GPT available, call GPT
        5. Resolve final label and confidence
        6. Log everything
        """
        user_prompt = self.prompt_builder.build_user_prompt(
            symptom_id, sentence,
        )

        # --- Step 1: Call Llama ---
        llama_output, llama_raw, llama_warnings, llama_meta = self._call_llm(
            client=self.llama_client,
            user_prompt=user_prompt,
            symptom_id=symptom_id,
            sentence=sentence,
            is_escalation=False,
        )

        # --- Step 2: Check escalation ---
        should_escalate, triggers = check_escalation(llama_output, symptom_id)

        # Safety: cap escalation rate
        self._total_scored += 1
        current_rate = (
            self._total_escalated / self._total_scored
            if self._total_scored > 0 else 0
        )
        if should_escalate and current_rate >= self.escalation_max_rate:
            logger.debug(
                "Escalation suppressed (rate %.1f%% >= cap %.1f%%)",
                current_rate * 100, self.escalation_max_rate * 100,
            )
            should_escalate = False
            triggers = [f"SUPPRESSED (rate cap {self.escalation_max_rate:.0%})"]

        # --- Step 3: GPT escalation ---
        gpt_output: Optional[LLMOutput] = None
        gpt_call_id: Optional[str] = None

        if should_escalate and self.gpt_client is not None:
            self._total_escalated += 1
            escalation_prompt = self.prompt_builder.build_escalation_prompt(
                symptom_id, sentence, llama_output, triggers,
            )

            gpt_output, gpt_raw, gpt_warnings, gpt_meta = self._call_llm(
                client=self.gpt_client,
                user_prompt=escalation_prompt,
                symptom_id=symptom_id,
                sentence=sentence,
                is_escalation=True,
                llama_output=llama_output,
            )

        # --- Step 4: Resolve label ---
        final_label, confidence_weight = resolve_label(
            llama_output=llama_output,
            gpt_output=gpt_output,
            escalated=should_escalate and gpt_output is not None,
            symptom_id=symptom_id,
        )

        # --- Step 5: Log the full Llama call (with resolution info) ---
        llama_call_id = self.llm_logger.log_call(
            provider=self.llama_client.provider,
            model=self.llama_client.model,
            symptom_id=symptom_id,
            sentence_id=sentence.docno,
            sentence_text=sentence.text,
            is_escalation=False,
            system_prompt_hash=self._system_prompt_hash,
            user_prompt=user_prompt,
            temperature=self.llama_client.temperature,
            max_tokens=self.llama_client.max_tokens,
            raw_response=llama_raw,
            parsed={
                "symptom_match": llama_output.symptom_match,
                "self_reference": llama_output.self_reference,
                "detail_level": llama_output.detail_level,
                "confounders": llama_output.confounders,
                "score": llama_output.score,
                "confidence": llama_output.confidence,
                "reasoning": llama_output.reasoning,
            },
            parse_success=len(llama_warnings) == 0,
            parse_warnings=llama_warnings,
            escalation_triggered=should_escalate,
            escalation_triggers=triggers,
            gpt_call_id=gpt_call_id,
            final_label=final_label,
            confidence_weight=confidence_weight,
            resolution_source=(
                "llama_only" if gpt_output is None
                else "llama_gpt_cascade"
            ),
            latency_ms=llama_meta.get("latency_ms", 0),
            prompt_tokens=llama_meta.get("prompt_tokens", 0),
            completion_tokens=llama_meta.get("completion_tokens", 0),
            total_tokens=llama_meta.get("total_tokens", 0),
            time_to_first_chunk_ms=llama_meta.get("time_to_first_chunk_ms"),
            stream_chunks=llama_meta.get("stream_chunks"),
            success=True,
        )

        # --- Step 6: Log GPT call if escalated ---
        if gpt_output is not None:
            gpt_call_id = self.llm_logger.log_call(
                provider=self.gpt_client.provider,
                model=self.gpt_client.model,
                symptom_id=symptom_id,
                sentence_id=sentence.docno,
                sentence_text=sentence.text,
                is_escalation=True,
                system_prompt_hash=self._system_prompt_hash,
                user_prompt=escalation_prompt,
                temperature=self.gpt_client.temperature,
                max_tokens=self.gpt_client.max_tokens,
                raw_response=gpt_raw,
                parsed={
                    "symptom_match": gpt_output.symptom_match,
                    "self_reference": gpt_output.self_reference,
                    "detail_level": gpt_output.detail_level,
                    "confounders": gpt_output.confounders,
                    "score": gpt_output.score,
                    "confidence": gpt_output.confidence,
                    "reasoning": gpt_output.reasoning,
                },
                parse_success=len(gpt_warnings) == 0,
                parse_warnings=gpt_warnings,
                llama_call_id=llama_call_id,
                llama_output_for_escalation={
                    "symptom_match": llama_output.symptom_match,
                    "self_reference": llama_output.self_reference,
                    "detail_level": llama_output.detail_level,
                    "confounders": llama_output.confounders,
                    "score": llama_output.score,
                    "confidence": llama_output.confidence,
                    "reasoning": llama_output.reasoning,
                },
                final_label=final_label,
                confidence_weight=confidence_weight,
                resolution_source="llama_gpt_cascade",
                latency_ms=gpt_meta.get("latency_ms", 0),
                prompt_tokens=gpt_meta.get("prompt_tokens", 0),
                completion_tokens=gpt_meta.get("completion_tokens", 0),
                total_tokens=gpt_meta.get("total_tokens", 0),
                time_to_first_chunk_ms=gpt_meta.get("time_to_first_chunk_ms"),
                stream_chunks=gpt_meta.get("stream_chunks"),
                success=True,
            )

        return ScoringResult(
            sentence_id=sentence.docno,
            symptom_id=symptom_id,
            llama_output=llama_output,
            gpt_output=gpt_output,
            escalated=should_escalate and gpt_output is not None,
            escalation_triggers=triggers,
            final_label=final_label,
            confidence_weight=confidence_weight,
        )

    def _call_llm(
        self,
        client: LLMClient,
        user_prompt: str,
        symptom_id: int,
        sentence: Sentence,
        is_escalation: bool,
        llama_output: Optional[LLMOutput] = None,
    ) -> tuple[LLMOutput, str, list[str], dict]:
        """Call an LLM and parse the response.

        Returns:
            Tuple of (parsed_output, raw_text, warnings, metadata).
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            t0 = time.monotonic()
            response = client.chat_completion(messages)
            elapsed_ms = (time.monotonic() - t0) * 1000

            raw_text = client.get_content(response)
            parsed_output, warnings = parse_llm_response(raw_text)

            usage = response.get("usage", {})
            meta_from_response = response.get("_meta", {})

            meta = {
                "latency_ms": meta_from_response.get(
                    "latency_ms", round(elapsed_ms, 1),
                ),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "time_to_first_chunk_ms": meta_from_response.get(
                    "time_to_first_chunk_ms",
                ),
                "stream_chunks": meta_from_response.get("stream_chunks"),
            }

            return parsed_output, raw_text, warnings, meta

        except Exception as e:
            logger.error(
                "LLM call failed for sentence %s, symptom %d: %s",
                sentence.docno, symptom_id, e,
            )
            # Log the failed call
            self.llm_logger.log_call(
                provider=client.provider,
                model=client.model,
                symptom_id=symptom_id,
                sentence_id=sentence.docno,
                sentence_text=sentence.text,
                is_escalation=is_escalation,
                system_prompt_hash=self._system_prompt_hash,
                user_prompt=user_prompt,
                temperature=client.temperature,
                max_tokens=client.max_tokens,
                raw_response="",
                success=False,
                error=str(e),
            )
            raise

    @property
    def escalation_rate(self) -> float:
        if self._total_scored == 0:
            return 0.0
        return self._total_escalated / self._total_scored

    @property
    def stats(self) -> dict:
        return {
            "total_scored": self._total_scored,
            "total_escalated": self._total_escalated,
            "escalation_rate": round(self.escalation_rate, 4),
            "llama_stats": self.llama_client.stats,
            "gpt_stats": (
                self.gpt_client.stats if self.gpt_client else None
            ),
        }
