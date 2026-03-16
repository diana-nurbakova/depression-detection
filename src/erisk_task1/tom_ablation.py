"""ToM ablation study — replays TalkDep conversations through the Orchestrator.

Unlike the standard ablation (which runs assessors once on the full transcript),
this module replays pre-recorded TalkDep conversations turn-by-turn through the
Orchestrator, running assessors incrementally.  Two conditions are compared:

  1. **tom_on**:  TomPerceptionTracker is active, providing coverage-gap guidance
     to the orchestrator and computing Wasserstein metrics.
  2. **tom_off**: No ToM tracker — standard incremental assessment baseline.

The pre-recorded interviewer questions are used as-is (no re-generation), so the
ToM tracker's main observable effects are:
  - Coverage gap detection and Wasserstein distance metrics
  - Orchestrator reasoning receives ToM context (for logging / analysis)
  - Ground-truth W_accuracy when golden BDI-II vectors are available

Output: per-persona JSON results + summary table comparing both conditions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .evaluation import (
    GOLDEN_KEY_SYMPTOMS,
    GOLDEN_SCORES,
    AblationResult,
    PersonaEvaluation,
    evaluate_persona,
    format_comparison_table,
)
from .linguistic import extract_features
from .llm_client import LLMClient, make_clients
from .models import (
    BDI_ITEMS,
    ConversationTurn,
    ItemScore,
    SeverityBand,
    score_to_band,
)
from .scoring import (
    collect_item_scores,
    compute_final_total,
    pass1_score,
    run_scoring_pipeline,
    select_top4_mechanical,
)
from .tom import TomPerceptionTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Replay a single TalkDep conversation through the Orchestrator
# ---------------------------------------------------------------------------


def replay_talkdep_conversation(
    persona_name: str,
    turns: list[dict],
    clients: dict[str, LLMClient],
    pipeline_cfg: PipelineConfig,
    tom_enabled: bool = True,
    ground_truth: Optional[list[int]] = None,
    assess_every_n: int = 0,
) -> dict:
    """Replay a pre-recorded TalkDep conversation through the assessment pipeline.

    Args:
        persona_name: Name of the TalkDep persona (e.g. "Maria").
        turns: List of {"speaker": "interviewer"|"persona", "text": "..."}.
        clients: LLM clients dict (at minimum "assessor").
        pipeline_cfg: Pipeline config (used for assessor settings).
        tom_enabled: Whether to enable the ToM perception tracker.
        ground_truth: Optional 21-dim BDI-II vector for W_accuracy.
        assess_every_n: Run assessors every N persona turns. 0 = use config default.

    Returns:
        Dict with predicted_total, predicted_band, item_scores, tom_summary, timing.
    """
    from .assessors import run_all_assessors
    from .orchestrator import Orchestrator

    t0 = time.monotonic()

    assess_interval = assess_every_n or pipeline_cfg.execution.assess_every_n_turns

    # Build ToM tracker
    tom_tracker = None
    if tom_enabled:
        gt_array = np.array(ground_truth, dtype=np.float32) if ground_truth else None
        tom_tracker = TomPerceptionTracker(
            guide_interviewer=True,
            cost_metric=pipeline_cfg.tom.cost_metric,
            ground_truth=gt_array,
        )
        logger.info(
            "  ToM tracker enabled (ground_truth=%s, cost_metric=%s)",
            "yes" if ground_truth else "no",
            pipeline_cfg.tom.cost_metric,
        )

    # Lightweight orchestrator — we only use it for state management,
    # not for generating interviewer messages.
    orch = Orchestrator(
        interviewer_client=clients["assessor"],  # placeholder, not used for generation
        assessor_client=clients["assessor"],
        orchestrator_client=clients.get("orchestrator", clients["assessor"]),
        max_turns=999,  # no early termination — replay all turns
        min_turns=999,
        assess_every_n=assess_interval,
        parallel_assessors=pipeline_cfg.execution.parallel_assessors,
        tom_tracker=tom_tracker,
    )

    # Replay turns
    persona_turn_count = 0
    turn_number = 0

    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"]

        if speaker == "interviewer":
            turn_number += 1
            orch.conversation.append(
                ConversationTurn(role="user", message=text, turn_number=turn_number)
            )
            # ToM: record interviewer attention
            if tom_tracker is not None:
                tom_tracker.update_interviewer(turn_number, text)

        elif speaker == "persona":
            # Extract linguistic features (same as pipeline.py)
            orch.process_persona_response(text, turn_number)
            persona_turn_count += 1

            # Run assessors at interval
            if persona_turn_count % assess_interval == 0:
                orch.process_turn_assessment(turn_number)

    # Final assessment if not done on last persona turn
    if persona_turn_count % assess_interval != 0:
        orch.process_turn_assessment(turn_number)

    # Scoring pipeline
    scoring_result = run_scoring_pipeline(
        orch.assessor_outputs, orch.features_history
    )
    final_total = scoring_result["pass2_total"]
    final_band = score_to_band(final_total)

    # Top-4 symptoms
    top4_items = select_top4_mechanical(scoring_result["item_scores"])
    top4_names = [
        BDI_ITEMS.get(item.item_id, item.item_name)
        for item in top4_items
        if item.item_id in BDI_ITEMS or item.item_name
    ]

    # ToM summary
    tom_summary: dict = {}
    if tom_tracker is not None:
        tom_summary = tom_tracker.to_summary_dict()
        logger.info(
            "  ToM: %d turns tracked, %d gaps, POT=%s",
            len(tom_summary.get("turns_tracked", [])),
            len(tom_summary.get("coverage_gaps", {}).get("gaps", [])),
            tom_summary.get("pot_available"),
        )

    elapsed = time.monotonic() - t0

    return {
        "persona": persona_name,
        "tom_enabled": tom_enabled,
        "predicted_total": final_total,
        "predicted_band": final_band,
        "pass1_total": scoring_result["pass1_total"],
        "pass2_total": scoring_result["pass2_total"],
        "top4": top4_names,
        "item_scores": scoring_result["item_scores"],
        "tom_summary": tom_summary,
        "turns_replayed": turn_number,
        "persona_turns": persona_turn_count,
        "timing_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# 2. Load TalkDep saved conversations (from save-talkdep output)
# ---------------------------------------------------------------------------


def load_saved_talkdep(
    data_dir: str | Path,
    personas: Optional[list[str]] = None,
) -> list[dict]:
    """Load saved TalkDep conversations + ground truth.

    Args:
        data_dir: Directory produced by `save-talkdep` (e.g. data/talkdep_conversations).
        personas: Optional filter — list of persona names.

    Returns:
        List of dicts with keys: name, golden_total, golden_band, turns, ground_truth_vector.
    """
    data_dir = Path(data_dir)

    # Load golden scores
    golden_path = data_dir / "golden_scores.json"
    if not golden_path.exists():
        raise FileNotFoundError(
            f"golden_scores.json not found in {data_dir}. "
            f"Run 'save-talkdep' first to export TalkDep conversations."
        )
    with open(golden_path) as f:
        golden_scores = json.load(f)

    # Load ground truth vectors
    gt_path = data_dir / "ground_truth.json"
    ground_truth: dict[str, list[int]] = {}
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)

    # Load per-persona conversations
    result = []
    for name in sorted(golden_scores.keys()):
        if personas and name not in personas:
            continue

        # Load all_sessions.json (combined turns across all 5 sessions)
        all_sessions_path = data_dir / name / "all_sessions.json"
        if not all_sessions_path.exists():
            logger.warning("No all_sessions.json for %s, skipping", name)
            continue

        with open(all_sessions_path) as f:
            session_data = json.load(f)

        gs = golden_scores[name]
        result.append({
            "name": name,
            "golden_total": gs["total"],
            "golden_band": gs["band"],
            "turns": session_data["turns"],
            "ground_truth_vector": ground_truth.get(name),
        })

    return result


# ---------------------------------------------------------------------------
# 3. Run the full ToM ablation study
# ---------------------------------------------------------------------------


def run_tom_ablation(
    pipeline_cfg: PipelineConfig,
    data_dir: str | Path = "data/talkdep_conversations",
    personas: Optional[list[str]] = None,
    output_dir: str | Path = "runs/tom_ablation",
    assess_every_n: int = 0,
    sessions: Optional[list[int]] = None,
) -> tuple[AblationResult, AblationResult]:
    """Run the ToM ablation study: tom_on vs tom_off on TalkDep.

    Args:
        pipeline_cfg: Pipeline config (assessor model, etc.).
        data_dir: Directory with saved TalkDep conversations (from save-talkdep).
        personas: Optional list of persona names to test.
        output_dir: Output directory for results.
        assess_every_n: Assessor interval override (0 = config default).
        sessions: Optional list of session numbers to use (default: all 5 combined).

    Returns:
        Tuple of (tom_off_result, tom_on_result) as AblationResult objects.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    conversations = load_saved_talkdep(data_dir, personas)
    if not conversations:
        raise ValueError(f"No conversations loaded from {data_dir}")

    logger.info(
        "ToM ablation: %d personas, assess_every=%s",
        len(conversations),
        assess_every_n or "config default",
    )

    # If sessions specified, load individual session files instead
    if sessions:
        data_path = Path(data_dir)
        for conv in conversations:
            combined_turns = []
            for sn in sessions:
                session_path = data_path / conv["name"] / f"session_{sn}.json"
                if session_path.exists():
                    with open(session_path) as f:
                        sd = json.load(f)
                    combined_turns.extend(sd["turns"])
                else:
                    logger.warning(
                        "Session %d not found for %s", sn, conv["name"]
                    )
            conv["turns"] = combined_turns

    clients = make_clients(pipeline_cfg)

    # Run both conditions
    conditions = [
        ("tom_off", False),
        ("tom_on", True),
    ]
    all_results: dict[str, AblationResult] = {}

    for condition_name, tom_enabled in conditions:
        logger.info("=" * 60)
        logger.info("Condition: %s", condition_name)
        logger.info("=" * 60)

        condition_dir = output_path / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        persona_evals: list[PersonaEvaluation] = []

        for conv in conversations:
            name = conv["name"]
            golden_total = conv["golden_total"]
            logger.info(
                "  %s (golden=%d, band=%s) ...",
                name, golden_total, conv["golden_band"],
            )

            try:
                result = replay_talkdep_conversation(
                    persona_name=name,
                    turns=conv["turns"],
                    clients=clients,
                    pipeline_cfg=pipeline_cfg,
                    tom_enabled=tom_enabled,
                    ground_truth=conv.get("ground_truth_vector"),
                    assess_every_n=assess_every_n,
                )

                # Evaluate against golden
                eval_result = evaluate_persona(
                    name=name,
                    predicted_total=result["predicted_total"],
                    predicted_top4=result["top4"],
                    item_scores=result.get("item_scores"),
                )
                persona_evals.append(eval_result)

                mark = "+" if eval_result.band_correct else "X"
                logger.info(
                    "    predicted=%d (%s) vs golden=%d (%s) [%s] -- %.1fs",
                    result["predicted_total"],
                    result["predicted_band"].value,
                    golden_total,
                    conv["golden_band"],
                    mark,
                    result["timing_s"],
                )

                # Save per-persona result
                _save_persona_result(result, condition_dir, condition_name)

            except Exception as e:
                logger.error(
                    "    FAILED: %s: %s", name, e, exc_info=True
                )

        ablation_result = AblationResult(
            config_name=condition_name,
            persona_results=persona_evals,
        )

        logger.info(
            "Condition %s: DCHR=%.1f%%, MAD=%.1f, ADODL=%.3f, ASHR=%.1f%%",
            condition_name,
            ablation_result.dchr * 100,
            ablation_result.mad,
            ablation_result.adodl,
            ablation_result.ashr_proxy * 100,
        )

        all_results[condition_name] = ablation_result

    # Save summary
    summary = {
        condition: r.summary() for condition, r in all_results.items()
    }
    summary_path = output_path / "tom_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    # Build ToM analysis summary (W_accuracy across personas for tom_on)
    _save_tom_analysis(output_path, conversations)

    return all_results["tom_off"], all_results["tom_on"]


# ---------------------------------------------------------------------------
# 4. Output helpers
# ---------------------------------------------------------------------------


def _save_persona_result(result: dict, output_dir: Path, condition: str) -> None:
    """Save a per-persona replay result to JSON."""
    name = result["persona"]
    save_data = {
        "persona": name,
        "condition": condition,
        "tom_enabled": result["tom_enabled"],
        "golden_total": GOLDEN_SCORES.get(name, -1),
        "golden_band": score_to_band(GOLDEN_SCORES.get(name, 0)).value,
        "predicted_total": result["predicted_total"],
        "predicted_band": result["predicted_band"].value,
        "pass1_total": result["pass1_total"],
        "pass2_total": result["pass2_total"],
        "top4": result["top4"],
        "turns_replayed": result["turns_replayed"],
        "persona_turns": result["persona_turns"],
        "timing_s": result["timing_s"],
        "item_scores": {
            str(k): {
                "score": v.score,
                "confidence": v.confidence,
                "state": v.state.value,
                "evidence": v.evidence,
            }
            for k, v in result["item_scores"].items()
        },
    }

    # Include ToM data for tom_on condition
    if result.get("tom_summary"):
        save_data["tom_summary"] = result["tom_summary"]

    fpath = output_dir / f"{condition}_{name}.json"
    fpath.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_tom_analysis(output_dir: Path, conversations: list[dict]) -> None:
    """Aggregate ToM metrics from tom_on results for analysis."""
    tom_on_dir = output_dir / "tom_on"
    if not tom_on_dir.exists():
        return

    analysis = {}
    for conv in conversations:
        name = conv["name"]
        fpath = tom_on_dir / f"tom_on_{name}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)

        tom = data.get("tom_summary", {})
        if not tom:
            continue

        # Extract key metrics
        w_accuracy = tom.get("W_accuracy", {})
        coverage = tom.get("coverage_gaps", {})
        gaps = coverage.get("gaps", [])

        analysis[name] = {
            "golden_total": conv["golden_total"],
            "golden_band": conv["golden_band"],
            "predicted_total": data["predicted_total"],
            "predicted_band": data["predicted_band"],
            "pot_available": tom.get("pot_available", False),
            "turns_tracked": tom.get("turns_tracked", []),
            "n_coverage_gaps": len(gaps),
            "coverage_gap_categories": [g.get("category") for g in gaps],
            "final_W_accuracy": (
                w_accuracy[str(max(int(k) for k in w_accuracy))]
                if w_accuracy else None
            ),
            "W_accuracy_trajectory": w_accuracy,
            "category_mass": coverage.get("category_mass", {}),
        }

    analysis_path = output_dir / "tom_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("ToM analysis saved to %s", analysis_path)


def format_tom_comparison(
    tom_off: AblationResult,
    tom_on: AblationResult,
) -> str:
    """Format a comparison table: tom_off vs tom_on."""
    return format_comparison_table([tom_off, tom_on])


def format_tom_analysis_table(analysis_path: str | Path) -> str:
    """Format the ToM analysis JSON into a readable table."""
    path = Path(analysis_path)
    if not path.exists():
        return "No ToM analysis file found."

    with open(path) as f:
        analysis = json.load(f)

    if not analysis:
        return "No ToM analysis data."

    lines = [
        "ToM Analysis: Wasserstein Distance & Coverage Gaps",
        "=" * 65,
        "",
        f"{'Persona':<10} {'Golden':>6} {'Pred':>6} {'W_acc':>8} {'Gaps':>5} "
        f"{'Gap Categories':<30}",
        "-" * 65,
    ]

    for name, data in sorted(analysis.items(), key=lambda x: x[1]["golden_total"]):
        w_acc = data.get("final_W_accuracy")
        w_acc_str = f"{w_acc:.3f}" if w_acc is not None else "n/a"
        gap_cats = ", ".join(data.get("coverage_gap_categories", [])) or "none"
        lines.append(
            f"{name:<10} {data['golden_total']:>6} {data['predicted_total']:>6} "
            f"{w_acc_str:>8} {data['n_coverage_gaps']:>5} {gap_cats:<30}"
        )

    # Summary stats
    w_accs = [
        d["final_W_accuracy"] for d in analysis.values()
        if d.get("final_W_accuracy") is not None
    ]
    if w_accs:
        lines.append("")
        lines.append(
            f"Mean W_accuracy: {sum(w_accs)/len(w_accs):.3f}  "
            f"(lower = better alignment with ground truth)"
        )

    return "\n".join(lines)
