"""Orchestrator — programmatic module + LLM reasoning module."""

from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Optional

from .assessors import run_all_assessors
from .linguistic import (
    compute_cumulative_features,
    detect_persona_profile,
    estimate_engagement_band,
    extract_features,
)
from .llm_client import LLMClient, parse_json_response
from .models import (
    ASSESSOR_ITEMS,
    BDI_ITEMS,
    AssessorOutput,
    ConversationTurn,
    ItemScore,
    ItemState,
    LinguisticFeatures,
    OrchestratorGuidance,
    SeverityBand,
    TopicArea,
    score_to_band,
)
from .prompts import INTERVIEWER_SYSTEM_PROMPT, ORCHESTRATOR_REASONING_PROMPT
from .scoring import collect_item_scores, pass1_score, compute_preliminary_consensus
from .tom import TomPerceptionTracker

logger = logging.getLogger(__name__)

# Default topic order (funnel: low-stigma → high-stigma)
DEFAULT_TOPIC_ORDER = [
    TopicArea.EMOTIONAL_STATE,
    TopicArea.ACTIVITIES_INTERESTS,
    TopicArea.DAILY_ROUTINE,
    TopicArea.SELF_PERCEPTION,
    TopicArea.FUTURE_OUTLOOK,
    TopicArea.DECISION_MAKING,
]

# BDI domain categories for coverage checking
BDI_DOMAIN_ITEMS = {
    "COGNITIVE": {2, 3, 5, 6, 7, 8, 9, 14},  # self-worth, guilt, pessimism, etc.
    "AFFECTIVE": {1, 4, 10, 12, 13, 17},      # sadness, pleasure, interest, crying, etc.
    "SOMATIC": {15, 16, 18, 20, 21},           # energy, sleep, appetite, fatigue, sex
    "FUNCTIONAL": {11, 13, 19},                # agitation, concentration, activity
}


class Orchestrator:
    """Manages the conversation loop between interviewer and persona."""

    def __init__(
        self,
        interviewer_client: LLMClient,
        assessor_client: LLMClient,
        orchestrator_client: LLMClient,
        max_turns: int = 10,
        min_turns: int = 5,
        assess_every_n: int = 1,
        parallel_assessors: bool = True,
        termination_confidence: float = 0.5,
        symptom_scorer=None,
        tom_tracker: Optional[TomPerceptionTracker] = None,
    ):
        self.interviewer_client = interviewer_client
        self.assessor_client = assessor_client
        self.orchestrator_client = orchestrator_client
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.assess_every_n = assess_every_n
        self.parallel_assessors = parallel_assessors
        self.termination_confidence = termination_confidence
        self.symptom_scorer = symptom_scorer  # Optional SymptomScorer
        self.tom_tracker = tom_tracker  # Optional TomPerceptionTracker

        # State
        self.conversation: list[ConversationTurn] = []
        self.features_history: list[LinguisticFeatures] = []
        self.assessor_outputs: dict[str, AssessorOutput] = {}
        self.topics_covered: list[TopicArea] = []
        self.topics_remaining: list[TopicArea] = list(DEFAULT_TOPIC_ORDER)
        self.item_scores: dict[int, ItemScore] = {}
        self.last_band: Optional[SeverityBand] = None
        self.current_band: Optional[SeverityBand] = None

    def get_transcript(self) -> str:
        """Format conversation as a readable transcript."""
        lines = []
        for turn in self.conversation:
            role = "Interviewer" if turn.role == "user" else "Person"
            lines.append(f"{role}: {turn.message}")
        return "\n".join(lines)

    def get_linguistic_summary(self) -> str:
        """Format cumulative linguistic features as a summary string."""
        cum = compute_cumulative_features(self.features_history)
        profile = detect_persona_profile(self.features_history)
        summary = (
            f"Absolutist density: {cum['absolutist_density']:.4f} ({cum['absolutist_band'].value})\n"
            f"Total words: {cum['total_words']}, Avg response length: {cum['avg_response_length']:.0f}\n"
            f"Total hedging: {cum['total_hedging']}, Total coping: {cum['total_coping']}\n"
            f"Negative emotion: {cum['total_negative_emotion']}, "
            f"Positive emotion: {cum['total_positive_emotion']}\n"
            f"Persona profile: {profile}"
        )

        # Append max-pooled symptom relevance across all turns (if available)
        relevance_vectors = [
            f.symptom_relevance for f in self.features_history
            if f.symptom_relevance is not None
        ]
        if relevance_vectors:
            import numpy as np
            max_rel = np.max(relevance_vectors, axis=0)
            top_items = sorted(
                range(len(max_rel)), key=lambda i: max_rel[i], reverse=True
            )[:5]
            top_str = ", ".join(
                f"{BDI_ITEMS[i + 1]}: {max_rel[i]:.2f}" for i in top_items
            )
            summary += f"\nSentence transformer top-5 signals: {top_str}"

        return summary

    def generate_interviewer_message(
        self, guidance: OrchestratorGuidance, turn_number: int
    ) -> str:
        """Generate the next interviewer message using the LLM."""
        # Build guidance JSON for the interviewer
        guidance_json = json.dumps({
            "turn_number": turn_number,
            "max_turns": self.max_turns,
            "next_topic": guidance.next_topic.value if guidance.next_topic else "EMOTIONAL_STATE",
            "exploration_gaps": guidance.exploration_gaps,
            "suggested_angle": guidance.suggested_angle,
            "topics_covered": [t.value for t in self.topics_covered],
            "interviewer_adaptation": guidance.interviewer_adaptation,
        }, indent=2)

        # Build conversation history for the LLM
        conv_messages = []
        for turn in self.conversation:
            conv_messages.append({"role": turn.role, "content": turn.message})

        user_msg = f"Orchestrator guidance:\n{guidance_json}"

        messages = [
            {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
        ]
        messages.extend(conv_messages)
        messages.append({"role": "user", "content": user_msg})

        response = self.interviewer_client.complete(messages)
        interviewer_msg = response.strip()

        # ToM: record which BDI domains the interviewer question targets
        if self.tom_tracker is not None:
            self.tom_tracker.update_interviewer(turn_number, interviewer_msg)

        return interviewer_msg

    def run_assessors(self) -> dict[str, AssessorOutput]:
        """Run all 4 assessors on the current conversation."""
        transcript = self.get_transcript()
        ling_summary = self.get_linguistic_summary()
        return run_all_assessors(
            self.assessor_client, transcript, ling_summary, self.parallel_assessors
        )

    def check_domain_coverage(self) -> dict[str, bool]:
        """Check which BDI domains have received evidence so far.

        Returns a dict mapping domain name to whether it has any coverage.
        """
        covered_items = {
            item_id for item_id, score in self.item_scores.items()
            if score.state != ItemState.NO_EVIDENCE
        }

        coverage = {}
        for domain, items in BDI_DOMAIN_ITEMS.items():
            coverage[domain] = bool(covered_items & items)
        return coverage

    def get_uncovered_domains(self) -> list[str]:
        """Return list of domain names with zero coverage."""
        return [d for d, covered in self.check_domain_coverage().items() if not covered]

    def run_orchestrator_reasoning(self, turn_number: int) -> OrchestratorGuidance:
        """Run the LLM reasoning module to decide next action."""
        # Build item state map
        item_state_map = {}
        for item_id in range(1, 22):
            if item_id in self.item_scores:
                s = self.item_scores[item_id]
                item_state_map[f"{item_id}_{BDI_ITEMS[item_id].lower().replace(' ', '_')}"] = {
                    "state": s.state.value,
                    "score": s.score,
                    "conf": s.confidence,
                }
            else:
                item_state_map[f"{item_id}_{BDI_ITEMS[item_id].lower().replace(' ', '_')}"] = {
                    "state": "NO_EVIDENCE",
                    "score": None,
                    "conf": 0,
                }

        # Severity estimates
        p1_total = pass1_score(self.item_scores)
        consensus, a_band, abs_band, eng_band = compute_preliminary_consensus(
            p1_total, self.features_history
        )

        cum = compute_cumulative_features(self.features_history)

        # Domain coverage check — identify gaps
        uncovered_domains = self.get_uncovered_domains()

        orchestrator_input: dict = {
            "turn_number": turn_number,
            "remaining_turns": self.max_turns - turn_number,
            "conversation_summary": self.get_transcript()[-1000:],  # Last 1000 chars
            "item_state_map": item_state_map,
            "severity_estimates": {
                "assessor_total": p1_total,
                "assessor_band": a_band.value,
                "absolutist_density": cum["absolutist_density"],
                "absolutist_band": abs_band.value,
                "engagement_band": eng_band.value,
                "bands_aligned": a_band == abs_band == eng_band,
            },
            "topics_covered": [t.value for t in self.topics_covered],
            "topics_remaining": [t.value for t in self.topics_remaining],
            "uncovered_bdi_domains": uncovered_domains,
        }

        # ToM: inject coverage gaps and alignment gap for the orchestrator to act on
        if self.tom_tracker is not None and self.tom_tracker.guide_interviewer:
            orchestrator_input["tom_perception_context"] = (
                self.tom_tracker.get_orchestrator_context()
            )

        input_data = json.dumps(orchestrator_input, indent=2)

        messages = [
            {"role": "system", "content": ORCHESTRATOR_REASONING_PROMPT},
            {"role": "user", "content": input_data},
        ]
        response_text = self.orchestrator_client.complete(messages)
        parsed = parse_json_response(response_text)

        if parsed is None:
            # Fallback: use next topic in order
            next_topic = self.topics_remaining[0] if self.topics_remaining else TopicArea.ADAPTIVE_FOLLOWUP
            return OrchestratorGuidance(
                decision="CONTINUE",
                next_topic=next_topic,
                suggested_angle="Explore the next topic area naturally.",
            )

        # Parse guidance
        decision = parsed.get("decision", "CONTINUE")
        next_topic_str = parsed.get("next_topic", "")
        try:
            next_topic = TopicArea(next_topic_str)
        except ValueError:
            next_topic = self.topics_remaining[0] if self.topics_remaining else TopicArea.ADAPTIVE_FOLLOWUP

        suggested_angle = parsed.get("suggested_angle", "")
        exploration_gaps = parsed.get("exploration_gaps", [])

        # Domain coverage override: if somatic has zero coverage after turn 3,
        # force a daily routine question targeting sleep/energy/appetite
        if turn_number >= 3 and "SOMATIC" in uncovered_domains and decision == "CONTINUE":
            logger.info(
                "Domain coverage override: SOMATIC has zero coverage at turn %d, "
                "forcing DAILY_ROUTINE probe", turn_number
            )
            next_topic = TopicArea.DAILY_ROUTINE
            suggested_angle = (
                "Ask about a typical day lately — specifically how sleep, "
                "energy, and meals have been. Use: 'Could you walk me through "
                "a typical day lately — how have your sleep, energy, and meals been?'"
            )
            if "SOMATIC (sleep, appetite, energy, fatigue)" not in exploration_gaps:
                exploration_gaps.insert(0, "SOMATIC (sleep, appetite, energy, fatigue)")

        return OrchestratorGuidance(
            decision=decision,
            next_topic=next_topic,
            suggested_angle=suggested_angle,
            exploration_gaps=exploration_gaps,
            priority_reasoning=parsed.get("priority_reasoning", ""),
            conflict_notes=parsed.get("conflict_notes", ""),
            interviewer_adaptation=parsed.get("interviewer_adaptation", ""),
        )

    def should_terminate(self, turn_number: int) -> tuple[bool, str]:
        """Check termination conditions."""
        if turn_number >= self.max_turns:
            return True, "Maximum turns reached"

        if turn_number < self.min_turns:
            return False, ""

        # Check confidence
        confidences = [s.confidence for s in self.item_scores.values()]
        if confidences:
            min_conf = min(confidences)
            mean_conf = mean(confidences)

            if min_conf >= 0.6:
                return True, "All items assessed with sufficient confidence"

            if self.last_band and self.current_band and self.last_band == self.current_band:
                if mean_conf >= self.termination_confidence:
                    return True, "Severity band stable with adequate confidence"

        return False, ""

    def process_persona_response(self, response: str, turn_number: int):
        """Process a persona response: extract features and update state."""
        features = extract_features(response, symptom_scorer=self.symptom_scorer)
        self.features_history.append(features)

        self.conversation.append(ConversationTurn(
            role="assistant",
            message=response,
            turn_number=turn_number,
            linguistic_features=features,
        ))

    def process_turn_assessment(self, turn_number: int):
        """Run assessors and update state after a turn."""
        self.assessor_outputs = self.run_assessors()
        self.item_scores = collect_item_scores(self.assessor_outputs)
        p1_total = pass1_score(self.item_scores)
        self.last_band = self.current_band
        self.current_band = score_to_band(p1_total)

        # ToM: update expressed profile and compute Wasserstein distances
        if self.tom_tracker is not None:
            self.tom_tracker.update_expressed(turn_number, self.item_scores)
            self.tom_tracker.compute_wasserstein_metrics(turn_number)
