"""Ablation study framework for the Task 1 pipeline.

Defines ablation configurations A0-A7 and a runner that executes
each configuration against TalkDep conversations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .assessors import run_all_assessors, run_single_assessor
from .config import PipelineConfig, load_config
from .evaluation import (
    AblationResult,
    TalkDepConversation,
    evaluate_persona,
    load_talkdep_conversations,
)
from .justificator import run_justificator
from .linguistic import compute_cumulative_features, extract_features
from .llm_client import LLMClient, make_clients, parse_json_response
from .models import (
    ASSESSOR_ITEMS,
    BDI_ITEMS,
    AssessorOutput,
    ItemScore,
    ItemState,
    SeverityBand,
    score_to_band,
)
from .post_hoc_correction import CorrectionStrategy, apply_correction
from .prompts import ASSESSOR_SHARED_PREAMBLE, get_assessor_prompt
from .scoring import (
    collect_item_scores,
    compute_final_total,
    compute_preliminary_consensus,
    pass1_score,
    pass2_bayesian_prior,
    run_scoring_pipeline,
    select_top4_mechanical,
)

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    description: str

    # Component toggles
    use_specialized_assessors: bool = True  # False = single general assessor
    use_linguistic_features: bool = True
    use_bayesian_prior: bool = True
    use_justificator: bool = True

    # Assessor configuration
    assessor_temperature: float = 0.1
    assessor_model: str = ""  # Empty = use default from PipelineConfig

    # Scoring configuration
    prior_confidence: float = 0.3

    # Post-hoc correction
    correction_strategy: str = "none"  # "none", "flat_minus_2", "proportional_085", "minus_5", etc.

    # Feature thresholds (absolutist density)
    absolutist_thresholds: tuple[float, float, float] = (0.005, 0.012, 0.025)


# Predefined ablation configurations from the spec (Section 6)
ABLATION_CONFIGS: dict[str, AblationConfig] = {
    "A0": AblationConfig(
        name="A0_baseline",
        description="Baseline: single general assessor, no linguistic features, no prior, no justificator",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
    ),
    "A1": AblationConfig(
        name="A1_specialized",
        description="+ 4 specialized assessors (no linguistic, no prior, no justificator)",
        use_specialized_assessors=True,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
    ),
    "A2": AblationConfig(
        name="A2_linguistic",
        description="+ Linguistic features (specialized assessors, no prior, no justificator)",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=False,
        use_justificator=False,
    ),
    "A3": AblationConfig(
        name="A3_prior",
        description="+ Bayesian prior (specialized assessors, linguistic, no justificator)",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=True,
        use_justificator=False,
    ),
    "A4": AblationConfig(
        name="A4_justificator",
        description="Full pipeline: specialized assessors + linguistic + prior + justificator",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=True,
        use_justificator=True,
    ),
    "A5": AblationConfig(
        name="A5_temp_sweep_low",
        description="Full pipeline with lower assessor temperature (0.05)",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=True,
        use_justificator=True,
        assessor_temperature=0.05,
    ),
    "A6": AblationConfig(
        name="A6_temp_sweep_high",
        description="Full pipeline with higher assessor temperature (0.3)",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=True,
        use_justificator=True,
        assessor_temperature=0.3,
    ),
    "A7": AblationConfig(
        name="A7_no_prior",
        description="Full pipeline without Bayesian prior (tests prior contribution)",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=False,
        use_justificator=True,
    ),
    # --- Post-hoc correction variants (matching v2 3-run strategy) ---
    "A0_band_aware": AblationConfig(
        name="A0_band_aware",
        description="POST_A0 + band_aware correction (Run 1 — safety run, ADODL=0.950)",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
        correction_strategy="band_aware",
    ),
    "A0_flat_minus_2": AblationConfig(
        name="A0_flat_minus_2",
        description="POST_A0 + flat -2 correction (Run 2 — calibrated risk)",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
        correction_strategy="flat_minus_2",
    ),
    "A0_flat_minus_3": AblationConfig(
        name="A0_flat_minus_3",
        description="POST_A0 + flat -3 correction (Run 3 — balanced hedge)",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
        correction_strategy="flat_minus_3",
    ),
    # --- Other correction variants (for comparison) ---
    "A0_none": AblationConfig(
        name="A0_none",
        description="POST_A0 + no correction (raw assessor output)",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
        correction_strategy="none",
    ),
    "A0_minus5": AblationConfig(
        name="A0_minus5",
        description="POST_A0 + minus_5 correction",
        use_specialized_assessors=False,
        use_linguistic_features=False,
        use_bayesian_prior=False,
        use_justificator=False,
        correction_strategy="minus_5",
    ),
    "A7_proportional_085": AblationConfig(
        name="A7_proportional_085",
        description="PRE_A7 + proportional ×0.85 correction",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=False,
        use_justificator=True,
        correction_strategy="proportional_085",
    ),
    "A7_progressive": AblationConfig(
        name="A7_progressive",
        description="PRE_A7 + progressive correction",
        use_specialized_assessors=True,
        use_linguistic_features=True,
        use_bayesian_prior=False,
        use_justificator=True,
        correction_strategy="progressive",
    ),
}


# General (non-specialized) assessor prompt for A0 baseline
GENERAL_ASSESSOR_PROMPT = """You are a depression symptom assessor. Analyse the conversation transcript
and score ALL 21 BDI-II items.

For each item, provide:
- score: 0-3 (null if no evidence)
- confidence: 0.0-1.0
- state: "SCORED", "NO_EVIDENCE", or "EVIDENCE_OF_ABSENCE"
- evidence: brief reasoning

BDI-II Items:
1-Sadness, 2-Pessimism, 3-Past failure, 4-Loss of pleasure, 5-Guilty feelings,
6-Punishment feelings, 7-Self-dislike, 8-Self-criticalness, 9-Suicidal thoughts,
10-Crying, 11-Agitation, 12-Loss of interest, 13-Indecisiveness, 14-Worthlessness,
15-Loss of energy, 16-Sleep changes, 17-Irritability, 18-Appetite changes,
19-Concentration difficulty, 20-Tiredness/fatigue, 21-Loss of interest in sex.

Score 0 means normal/absent. Score 3 means most severe.
Be conservative: only score items with clear conversational evidence.
For Item 9 (Suicidal thoughts), require strong evidence before scoring above 0.
For Item 21 (Sex), only score if explicitly discussed.

ITEM-SPECIFIC SCORING GUIDELINES:

Item 1 (Sadness): Measures FEELING SAD or UNHAPPY. DO NOT score emotional numbness,
flatness, or "walking through a fog" as sadness — numbness is the ABSENCE of emotion
(maps to Item 4: Loss of Pleasure). Only score with explicit sadness statements
("I feel sad", "I'm unhappy", crying, feeling blue/low/miserable).

Item 19 (Concentration Difficulty): Measures ability to FOCUS ATTENTION on cognitive
tasks. DO NOT score sleep disruption (Item 16), feeling foggy/numb (Item 4), "going
through the motions" (anhedonia), or workload overwhelm as concentration difficulty.
Require explicit mention of difficulty focusing, reading, or following conversations.

Item 20 (Tiredness/Fatigue): Score 3 requires the person is UNABLE to do most daily
activities. If they still work, maintain basic routines, or go out, cap at score 2.
Reserve 3 for housebound/bedbound due to fatigue.

Respond ONLY with valid JSON:
{"items": [{"id": N, "name": "...", "score": 0-3 or null, "confidence": 0.0-1.0,
"state": "SCORED|NO_EVIDENCE|EVIDENCE_OF_ABSENCE", "evidence": "..."}]}"""


def _run_general_assessor(
    client: LLMClient,
    transcript: str,
    linguistic_summary: str = "",
) -> dict[str, AssessorOutput]:
    """Run a single general assessor covering all 21 items (A0 baseline)."""
    user_message = (
        f"Analyse this conversation transcript and score ALL 21 BDI-II items.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    if linguistic_summary:
        user_message += f"\n\nLINGUISTIC FEATURES:\n{linguistic_summary}"

    messages = [
        {"role": "system", "content": GENERAL_ASSESSOR_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response_text = client.complete(messages)
    raw_json = parse_json_response(response_text)

    # Parse into 4 AssessorOutput objects to match the pipeline interface
    all_items: dict[int, dict] = {}
    if raw_json and "items" in raw_json:
        for item_data in raw_json["items"]:
            item_id = item_data.get("id")
            if item_id is not None:
                all_items[item_id] = item_data

    outputs = {}
    for assessor_name, item_ids in ASSESSOR_ITEMS.items():
        items = []
        for item_id in item_ids:
            if item_id in all_items:
                d = all_items[item_id]
                score = d.get("score")
                confidence = float(d.get("confidence", 0.0))
                state_str = d.get("state", "")
                evidence = d.get("evidence", "")

                if state_str in ("SCORED", "EVIDENCE_OF_ABSENCE", "NO_EVIDENCE"):
                    state = ItemState(state_str)
                elif score is None:
                    state = ItemState.NO_EVIDENCE
                elif score == 0 and confidence >= 0.5:
                    state = ItemState.EVIDENCE_OF_ABSENCE
                elif score == 0 and confidence < 0.5:
                    state = ItemState.NO_EVIDENCE
                else:
                    state = ItemState.SCORED

                if score is not None:
                    score = int(score)

                items.append(
                    ItemScore(
                        item_id=item_id,
                        item_name=BDI_ITEMS.get(item_id, f"Item {item_id}"),
                        score=score,
                        confidence=confidence,
                        state=state,
                        evidence=evidence,
                    )
                )
            else:
                items.append(
                    ItemScore(
                        item_id=item_id,
                        item_name=BDI_ITEMS[item_id],
                        score=None,
                        confidence=0.0,
                        state=ItemState.NO_EVIDENCE,
                        evidence="Not included in general assessor response",
                    )
                )
        outputs[assessor_name] = AssessorOutput(
            assessor_name=assessor_name, items=items
        )

    return outputs


def run_ablation_single(
    ablation_cfg: AblationConfig,
    conversation: TalkDepConversation,
    clients: dict[str, LLMClient],
    pipeline_cfg: PipelineConfig,
) -> dict:
    """Run a single ablation configuration on a single persona.

    Returns a dict with predicted_total, predicted_band, top4, item_scores, timing.
    """
    t0 = time.monotonic()
    transcript = conversation.transcript

    # Extract linguistic features from persona responses
    features_history = []
    if ablation_cfg.use_linguistic_features:
        for line in transcript.split("\n"):
            if line.startswith("Person:"):
                text = line[len("Person:") :].strip()
                features_history.append(extract_features(text))

    linguistic_summary = ""
    if features_history and ablation_cfg.use_linguistic_features:
        cum = compute_cumulative_features(features_history)
        linguistic_summary = (
            f"Absolutist density: {cum['absolutist_density']:.4f} "
            f"(band: {cum['absolutist_band'].value})\n"
            f"Avg response length: {cum['avg_response_length']:.1f} words\n"
            f"Total hedging: {cum['total_hedging']}, "
            f"Total coping: {cum['total_coping']}\n"
            f"Negative emotion: {cum['total_negative_emotion']}, "
            f"Positive emotion: {cum['total_positive_emotion']}"
        )

    # Run assessors
    assessor_client = clients["assessor"]
    if ablation_cfg.assessor_temperature != pipeline_cfg.assessor.temperature:
        # Override temperature for this run
        assessor_client = LLMClient(
            provider=assessor_client.provider,
            base_url=assessor_client.base_url,
            api_key=assessor_client.api_key,
            model=assessor_client.model,
            temperature=ablation_cfg.assessor_temperature,
            max_tokens=assessor_client.max_tokens,
            max_retries=assessor_client.max_retries,
            timeout=assessor_client.timeout,
        )

    if ablation_cfg.use_specialized_assessors:
        assessor_outputs = run_all_assessors(
            assessor_client,
            transcript,
            linguistic_summary=linguistic_summary,
            parallel=pipeline_cfg.execution.parallel_assessors,
        )
    else:
        assessor_outputs = _run_general_assessor(
            assessor_client, transcript, linguistic_summary
        )

    # Scoring pipeline
    item_scores = collect_item_scores(assessor_outputs)
    p1_total = pass1_score(item_scores)

    # Pass 2: Bayesian prior (conditional)
    if ablation_cfg.use_bayesian_prior and features_history:
        consensus, assessor_band, abs_band, eng_band = compute_preliminary_consensus(
            p1_total, features_history
        )
        item_scores = pass2_bayesian_prior(
            item_scores, p1_total, consensus, assessor_band
        )
    elif ablation_cfg.use_linguistic_features and features_history:
        # Still compute consensus for reporting even without applying prior
        consensus, assessor_band, abs_band, eng_band = compute_preliminary_consensus(
            p1_total, features_history
        )
    else:
        assessor_band = score_to_band(p1_total)
        consensus = assessor_band

    p2_total = compute_final_total(item_scores)
    p2_band = score_to_band(p2_total)

    # Justificator
    final_total = p2_total
    final_band = p2_band
    top4_names = [
        BDI_ITEMS.get(item.item_id, f"Item {item.item_id}")
        for item in select_top4_mechanical(item_scores)
    ]

    if ablation_cfg.use_justificator:
        justificator_client = clients.get("justificator")
        if justificator_client:
            try:
                j_output = run_justificator(
                    justificator_client,
                    conversation.name,
                    transcript,
                    assessor_outputs,
                    item_scores,
                    p2_total,
                    p2_band,
                    features_history,
                )
                final_total = j_output.final_total
                final_band = j_output.final_band
                if j_output.top_4_symptoms:
                    top4_names = [
                        s.get("name", s) if isinstance(s, dict) else s
                        for s in j_output.top_4_symptoms[:4]
                    ]
            except Exception as e:
                logger.warning(
                    "Justificator failed for %s: %s — using Pass 2 scores",
                    conversation.name,
                    e,
                )

    # Post-hoc correction
    correction_result = None
    if ablation_cfg.correction_strategy != "none":
        strategy = CorrectionStrategy(ablation_cfg.correction_strategy)
        correction_result = apply_correction(final_total, strategy)
        final_total = correction_result["corrected_total"]
        final_band = score_to_band(final_total)

    elapsed = time.monotonic() - t0

    return {
        "predicted_total": final_total,
        "predicted_band": final_band,
        "top4": top4_names,
        "item_scores": item_scores,
        "pass1_total": p1_total,
        "pass2_total": p2_total,
        "correction": correction_result,
        "timing_s": round(elapsed, 1),
    }


def run_ablation(
    ablation_cfg: AblationConfig,
    conversations: list[TalkDepConversation],
    pipeline_cfg: PipelineConfig,
    output_dir: Optional[Path] = None,
    personas: Optional[list[str]] = None,
) -> AblationResult:
    """Run an ablation configuration against all TalkDep personas.

    Args:
        ablation_cfg: The ablation variant to test.
        conversations: List of TalkDep conversations to evaluate.
        pipeline_cfg: Base pipeline configuration.
        output_dir: Optional directory to save per-persona results.
        personas: Optional list of persona names to evaluate (default: all).

    Returns:
        AblationResult with aggregated metrics.
    """
    logger.info(
        "Running ablation %s: %s", ablation_cfg.name, ablation_cfg.description
    )

    clients = make_clients(pipeline_cfg)
    persona_evaluations = []

    # Filter personas if specified
    target_convs = conversations
    if personas:
        target_convs = [c for c in conversations if c.name in personas]

    for conv in target_convs:
        logger.info(
            "Evaluating %s (golden=%d, band=%s)...",
            conv.name,
            conv.golden_total,
            conv.golden_band.value,
        )

        try:
            result = run_ablation_single(ablation_cfg, conv, clients, pipeline_cfg)

            eval_result = evaluate_persona(
                name=conv.name,
                predicted_total=result["predicted_total"],
                predicted_top4=result["top4"],
                item_scores=result.get("item_scores"),
            )
            persona_evaluations.append(eval_result)

            mark = "+" if eval_result.band_correct else "X"
            logger.info(
                "  %s: predicted=%d (%s) vs golden=%d (%s) [%s] — %.1fs",
                conv.name,
                result["predicted_total"],
                result["predicted_band"].value,
                conv.golden_total,
                conv.golden_band.value,
                mark,
                result["timing_s"],
            )

            # Save per-persona result
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                persona_file = output_dir / f"{ablation_cfg.name}_{conv.name}.json"
                _save = {
                    "persona": conv.name,
                    "config": ablation_cfg.name,
                    "golden_total": conv.golden_total,
                    "golden_band": conv.golden_band.value,
                    "predicted_total": result["predicted_total"],
                    "predicted_band": result["predicted_band"].value,
                    "pass1_total": result["pass1_total"],
                    "pass2_total": result["pass2_total"],
                    "correction": result.get("correction"),
                    "band_correct": eval_result.band_correct,
                    "deviation": eval_result.absolute_deviation,
                    "top4": result["top4"],
                    "timing_s": result["timing_s"],
                    "item_scores": {
                        str(k): {
                            "score": v.score,
                            "confidence": v.confidence,
                            "state": v.state.value,
                        }
                        for k, v in result["item_scores"].items()
                    },
                }
                persona_file.write_text(json.dumps(_save, indent=2))

        except Exception as e:
            logger.error("Failed to evaluate %s: %s", conv.name, e, exc_info=True)

    ablation_result = AblationResult(
        config_name=ablation_cfg.name,
        persona_results=persona_evaluations,
    )

    logger.info(
        "Ablation %s complete: DCHR=%.1f%%, MAD=%.1f, ASHR=%.1f%%",
        ablation_cfg.name,
        ablation_result.dchr * 100,
        ablation_result.mad,
        ablation_result.ashr_proxy * 100,
    )

    return ablation_result


def run_full_ablation_study(
    pipeline_cfg: PipelineConfig,
    talkdep_dir: str | Path = "data/TalkDep",
    configs: Optional[list[str]] = None,
    personas: Optional[list[str]] = None,
    output_dir: str | Path = "runs/ablation",
) -> list[AblationResult]:
    """Run the complete ablation study.

    Args:
        pipeline_cfg: Base pipeline configuration.
        talkdep_dir: Path to TalkDep repo.
        configs: List of config names to run (default: all A0-A7).
        personas: Optional list of persona names (default: all 12).
        output_dir: Directory for results.

    Returns:
        List of AblationResult, one per configuration.
    """
    output_path = Path(output_dir)
    conversations = load_talkdep_conversations(talkdep_dir)

    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    results = []
    for config_name in configs:
        if config_name not in ABLATION_CONFIGS:
            logger.warning("Unknown ablation config: %s", config_name)
            continue

        ablation_cfg = ABLATION_CONFIGS[config_name]
        config_output = output_path / config_name
        result = run_ablation(
            ablation_cfg, conversations, pipeline_cfg, config_output, personas
        )
        results.append(result)

    # Save summary
    if results:
        summary_file = output_path / "ablation_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_data = [r.summary() for r in results]
        summary_file.write_text(json.dumps(summary_data, indent=2))
        logger.info("Ablation summary saved to %s", summary_file)

    return results
