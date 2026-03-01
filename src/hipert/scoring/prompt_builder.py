"""Build LLM prompts for ADHD symptom relevance scoring.

Constructs:
- System prompt (shared across all symptoms)
- Per-symptom user prompt with clinical elaboration + few-shot examples
- GPT escalation prompt including Llama's structured output
"""

from __future__ import annotations

from hipert.models import FewShotExample, LLMOutput, Sentence, SymptomDefinition

# System prompt from specs/llm_prompt_template_with_examples.md
SYSTEM_PROMPT = """\
You are a clinical psychologist specializing in adult ADHD assessment.
You have deep expertise in interpreting self-reported experiences
against DSM-5-TR ADHD criteria and the ASRS-v1.1 screening instrument.

Your task is to assess how relevant a social media sentence is to a
specific ADHD symptom. You will see the TARGET sentence with one
sentence of context before and after it.

## SCORING CRITERIA

SCORE 0 — IRRELEVANT:
  The sentence does not address this symptom, OR mentions it only
  in abstract/general terms without the writer's own experience.

SCORE 1 — MARGINALLY RELEVANT:
  The sentence touches on the symptom area but is vague, indirect,
  or ambiguous. A possible connection exists but is not explicit.

SCORE 2 — MODERATELY RELEVANT:
  The sentence clearly addresses this symptom AND conveys the
  writer's own experience, but lacks specificity or rich detail.

SCORE 3 — HIGHLY RELEVANT:
  The sentence explicitly describes the writer's own experience of
  this exact symptom with concrete detail, specific situations,
  or clear behavioral examples.

## IMPORTANT RULES

- Both positive and negative statements count as relevant. ("I can
  finally focus since starting Vyvanse" IS relevant to attention.)
- Medication effects that describe symptom changes ARE relevant.
- Merely naming ADHD or the symptom is NOT enough for score >= 2.
  The sentence must describe actual experience.
- When uncertain between two adjacent scores, choose the LOWER one.
- A sentence relevant to ADHD in general but not THIS specific
  symptom should score 0 or 1, not higher.

## RESPONSE FORMAT

You MUST respond using EXACTLY this template:

SYMPTOM_MATCH: [YES|PARTIAL|NO]
SELF_REFERENCE: [DIRECT|INDIRECT|NONE]
DETAIL_LEVEL: [HIGH|MEDIUM|LOW|NONE]
CONFOUNDERS: [list alternatives or "NONE"]
SCORE: [0|1|2|3]
CONFIDENCE: [1|2|3|4|5]
REASONING: [1-2 sentences]

CONFIDENCE scale:
  1 = Very uncertain, this could easily be a different score
  2 = Somewhat uncertain, I see arguments for adjacent scores
  3 = Moderately confident, the score seems right but edge cases exist
  4 = Confident, the evidence clearly supports this score
  5 = Very confident, this is an unambiguous case"""

# Factor display names
_FACTOR_NAMES = {
    "Inattention": "Inattention",
    "Motor_HI": "Motor Hyperactivity-Impulsivity",
    "Verbal_HI": "Verbal Hyperactivity-Impulsivity",
}


class PromptBuilder:
    """Builds prompts for LLM scoring of ADHD symptom relevance."""

    def __init__(
        self,
        symptoms: dict[int, SymptomDefinition],
        examples: dict[int, list[FewShotExample]] | None = None,
    ) -> None:
        self.symptoms = symptoms
        self.examples = examples or {}

    @staticmethod
    def get_system_prompt() -> str:
        """Return the shared system prompt."""
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        symptom_id: int,
        sentence: Sentence,
    ) -> str:
        """Build the per-symptom user prompt.

        Includes clinical elaboration (respecting token budget) and
        few-shot examples if available.
        """
        symptom = self.symptoms[symptom_id]
        factor_name = _FACTOR_NAMES.get(symptom.factor.value, symptom.factor.value)

        # Build clinical elaboration based on token budget
        elaboration = self._build_elaboration(symptom)

        # Build examples block
        examples_block = self._build_examples_block(symptom_id)

        # Assemble prompt
        parts = [
            f'ASRS SYMPTOM #{symptom.item_number}: "{symptom.text}"',
            f"SYMPTOM CATEGORY: {factor_name}",
            "",
            "CLINICAL ELABORATION:",
            elaboration,
        ]

        if examples_block:
            parts.extend([
                "",
                "--- SCORING EXAMPLES FOR THIS SYMPTOM ---",
                "",
                examples_block,
            ])

        parts.extend([
            "",
            "--- NOW SCORE THIS SENTENCE ---",
            "",
            f"BEFORE: {sentence.pre}" if sentence.pre else "BEFORE:",
            f">>> TARGET: {sentence.text} <<<",
            f"AFTER: {sentence.post}" if sentence.post else "AFTER:",
        ])

        return "\n".join(parts)

    def build_escalation_prompt(
        self,
        symptom_id: int,
        sentence: Sentence,
        llama_output: LLMOutput,
        trigger_descriptions: list[str],
    ) -> str:
        """Build the GPT escalation prompt including Llama's output.

        GPT receives both the original scoring context and Llama's
        full structured output as preliminary analysis.
        """
        # Include the same scoring context as the original prompt
        user_prompt = self.build_user_prompt(symptom_id, sentence)

        trigger_text = "; ".join(trigger_descriptions)

        escalation_context = f"""\
A preliminary automated assessment produced the following:

SYMPTOM_MATCH: {llama_output.symptom_match}
SELF_REFERENCE: {llama_output.self_reference}
DETAIL_LEVEL: {llama_output.detail_level}
CONFOUNDERS: {llama_output.confounders}
SCORE: {llama_output.score}
CONFIDENCE: {llama_output.confidence}
REASONING: {llama_output.reasoning}

Escalation reason: {trigger_text}

Please provide your INDEPENDENT assessment using the same format.
You may agree or disagree with the preliminary analysis. Focus
especially on the escalation reason when forming your judgment."""

        return f"{user_prompt}\n\n{escalation_context}"

    def _build_elaboration(self, symptom: SymptomDefinition) -> str:
        """Build clinical elaboration respecting token budget strategy."""
        parts: list[str] = []

        if symptom.token_budget == "full_4":
            # Items 7-11: all 4 layers
            if symptom.clinical_definition:
                parts.append(f"Clinical Definition: {symptom.clinical_definition}")
            if symptom.adult_manifestation:
                parts.append(
                    f"Adult Manifestation: {symptom.adult_manifestation}",
                )
            if symptom.discussion_topics:
                parts.append(
                    f"Online Discussion Patterns: {symptom.discussion_topics}",
                )
            if symptom.differential_markers:
                parts.append(
                    f"Differential Markers: {symptom.differential_markers}",
                )

        elif symptom.token_budget == "compressed_3":
            # Items 1-4, 6, 13, 14: L1 + L3 + L4
            if symptom.clinical_definition:
                parts.append(f"Clinical Definition: {symptom.clinical_definition}")
            if symptom.discussion_topics:
                parts.append(
                    f"Online Discussion Patterns: {symptom.discussion_topics}",
                )
            if symptom.differential_markers:
                parts.append(
                    f"Differential Markers: {symptom.differential_markers}",
                )

        else:
            # minimal_2: Items 5, 12, 15-18: L1 + L3
            if symptom.clinical_definition:
                parts.append(f"Clinical Definition: {symptom.clinical_definition}")
            if symptom.discussion_topics:
                parts.append(
                    f"Online Discussion Patterns: {symptom.discussion_topics}",
                )

        return "\n\n".join(parts) if parts else "No elaboration available."

    def _build_examples_block(self, symptom_id: int) -> str:
        """Build the few-shot examples block for a symptom."""
        examples = self.examples.get(symptom_id, [])
        if not examples:
            return ""

        blocks: list[str] = []
        for i, ex in enumerate(sorted(examples, key=lambda e: e.score), 1):
            block = f"""\
EXAMPLE {i} (Score {ex.score}):
BEFORE: {ex.pre}
>>> TARGET: {ex.text} <<<
AFTER: {ex.post}

SYMPTOM_MATCH: {ex.symptom_match}
SELF_REFERENCE: {ex.self_reference}
DETAIL_LEVEL: {ex.detail_level}
CONFOUNDERS: {ex.confounders}
SCORE: {ex.score}
CONFIDENCE: {ex.confidence}
REASONING: {ex.reasoning}"""
            blocks.append(block)

        return "\n\n".join(blocks)
