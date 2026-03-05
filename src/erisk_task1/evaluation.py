"""TalkDep evaluation harness for the Task 1 pipeline.

Loads TalkDep conversations with golden BDI-II scores and computes
evaluation metrics per Appendix A of the pipeline specification:
  - DCHR (Depression Category Hit Rate)
  - MAD (Mean Absolute Deviation)
  - ADODL (Average Difference between Overall Depression Levels)
  - ASHR-proxy (Average Symptom Hit Rate)
  - Boundary accuracy
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional

from .models import BDI_ITEMS, ItemScore, ItemState, SeverityBand, score_to_band

logger = logging.getLogger(__name__)

# Golden-truth BDI-II total scores from TalkDep
GOLDEN_SCORES: dict[str, int] = {
    "Maria": 40,
    "Marco": 38,
    "Elena": 35,
    "Linda": 28,
    "Laura": 23,
    "James": 22,
    "Alex": 15,
    "Gabriel": 13,
    "Ethan": 12,
    "Priya": 7,
    "Maya": 6,
    "Noah": 5,
}

# Mapping from TalkDep profile symptom names to canonical BDI-II names
# (per Section A.4 of the spec)
_SYMPTOM_CANONICAL: dict[str, str] = {
    "Sadness": "Sadness",
    "Hopelessness": "Pessimism",
    "Feelings of Hopelessness": "Pessimism",
    "Self-criticism": "Self-criticalness",
    "Self-Criticism": "Self-criticalness",
    "Minor Self-Criticism": "Self-criticalness",
    "Transient Self-Criticism": "Self-criticalness",
    "Decreased Sense of Accomplishment": "Self-criticalness",
    "Loss of Interest": "Loss of interest",
    "Loss of interest": "Loss of interest",
    "Tiredness": "Tiredness or fatigue",
    "Fatigue": "Tiredness or fatigue",
    "Extreme Fatigue": "Tiredness or fatigue",
    "Transient Fatigue": "Tiredness or fatigue",
    "Guilt": "Guilty feelings",
    "Guilty Feelings": "Guilty feelings",
    "Feelings of Guilt": "Guilty feelings",
    "Indecisiveness": "Indecisiveness",
    "Difficulty Making Decisions": "Indecisiveness",
    "Irritability": "Irritability",
    "Mild Restlessness": "Agitation",
    "Agitation": "Agitation",
    "Sleep Disturbances": "Changes in sleeping pattern",
    "Changes in Sleeping Pattern": "Changes in sleeping pattern",
    "Mild Difficulty Sleeping": "Changes in sleeping pattern",
    "Minor Sleep Changes": "Changes in sleeping pattern",
    "Appetite Changes": "Changes in appetite",
    "Reduced Appetite": "Changes in appetite",
    "Difficulty Concentrating": "Concentration difficulty",
    "Concentration Difficulty": "Concentration difficulty",
    "Loss of Energy": "Loss of energy",
    "Low Motivation": "Loss of energy",
    "Self-Doubt": "Self-dislike",
    "Mild Self-Doubt": "Self-dislike",
    "Low Confidence in Social Situations": "Self-dislike",
    "Feelings of Worthlessness": "Worthlessness",
    "Social Withdrawal": "Loss of interest",
    "Feeling of Isolation": "Loss of interest",
    "Decreased Enthusiasm": "Loss of pleasure",
    "Loss of Pleasure": "Loss of pleasure",
    "Loss of Pleasure (Anhedonia)": "Loss of pleasure",
    "Crying": "Crying",
    "Occasional Worry": "Agitation",
    "Occasional Anxiety": "Agitation",
    "Past Failure": "Past failure",
    "Pessimism": "Pessimism",
}

# Key symptoms per persona (from TalkDep profiles, mapped to canonical BDI-II names)
GOLDEN_KEY_SYMPTOMS: dict[str, list[str]] = {
    # Severe
    "Maria": ["Sadness", "Self-criticalness", "Loss of interest", "Tiredness or fatigue"],
    "Marco": ["Past failure", "Agitation", "Loss of interest", "Concentration difficulty"],
    "Elena": ["Pessimism", "Crying", "Tiredness or fatigue", "Loss of interest"],
    # Moderate
    "Linda": ["Guilty feelings", "Pessimism", "Indecisiveness", "Tiredness or fatigue"],
    "Laura": ["Sadness", "Worthlessness", "Tiredness or fatigue", "Concentration difficulty"],
    "James": ["Loss of energy", "Worthlessness", "Loss of interest", "Indecisiveness"],
    # Mild
    "Alex": ["Concentration difficulty", "Irritability", "Changes in sleeping pattern", "Changes in appetite"],
    "Gabriel": ["Irritability", "Self-criticalness", "Changes in appetite", "Self-dislike"],
    "Ethan": ["Loss of pleasure", "Loss of interest", "Changes in sleeping pattern", "Indecisiveness"],
    # Minimal
    "Priya": ["Agitation", "Changes in sleeping pattern", "Self-criticalness", "Loss of pleasure"],
    "Maya": ["Agitation", "Self-criticalness", "Tiredness or fatigue"],
    "Noah": ["Self-dislike", "Loss of energy", "Irritability", "Changes in sleeping pattern"],
}

# Boundary personas: near band edges, hardest to classify correctly
BOUNDARY_PERSONAS = {"Ethan", "Gabriel", "Alex", "Linda"}


def canonicalize_symptom(name: str) -> str:
    """Map a symptom name to canonical BDI-II name."""
    return _SYMPTOM_CANONICAL.get(name, name)


@dataclass
class TalkDepConversation:
    """A parsed TalkDep conversation with golden scores."""

    name: str
    transcript: str
    golden_total: int
    golden_band: SeverityBand
    golden_key_symptoms: list[str]


@dataclass
class PersonaEvaluation:
    """Evaluation result for a single persona."""

    name: str
    golden_total: int
    golden_band: SeverityBand
    predicted_total: int
    predicted_band: SeverityBand
    band_correct: bool
    absolute_deviation: int
    cr_score: float  # Closeness ratio for ADODL
    predicted_top4: list[str]
    symptom_hit_rate: float  # ASHR-proxy per-persona
    item_scores: dict[int, Optional[int]] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Aggregated evaluation result for an ablation configuration."""

    config_name: str
    persona_results: list[PersonaEvaluation]

    @property
    def dchr(self) -> float:
        """Depression Category Hit Rate: fraction of correct bands."""
        if not self.persona_results:
            return 0.0
        return sum(1 for r in self.persona_results if r.band_correct) / len(
            self.persona_results
        )

    @property
    def mad(self) -> float:
        """Mean Absolute Deviation from golden scores."""
        if not self.persona_results:
            return 0.0
        return mean(r.absolute_deviation for r in self.persona_results)

    @property
    def adodl(self) -> float:
        """Average Difference between Overall Depression Levels.

        CR = (MAD_max - |actual - predicted|) / MAD_max, MAD_max = 63
        ADODL = mean(CR) across all personas. Range: 0-1, higher is better.
        """
        if not self.persona_results:
            return 0.0
        return mean(r.cr_score for r in self.persona_results)

    @property
    def ashr_proxy(self) -> float:
        """Average Symptom Hit Rate proxy across all 12 personas.

        Per spec: for K=0 golden symptoms and predicted=[], hit_rate=1.0
        """
        if not self.persona_results:
            return 0.0
        return mean(r.symptom_hit_rate for r in self.persona_results)

    @property
    def band_accuracy_by_severity(self) -> dict[str, float]:
        """Band accuracy broken down by golden severity."""
        by_band: dict[str, list[bool]] = {}
        for r in self.persona_results:
            band = r.golden_band.value
            by_band.setdefault(band, []).append(r.band_correct)
        return {
            band: sum(hits) / len(hits) for band, hits in by_band.items()
        }

    @property
    def boundary_accuracy(self) -> float:
        """Accuracy on boundary personas (Ethan=12, Gabriel=13, Alex=15, Linda=28)."""
        boundary = [r for r in self.persona_results if r.name in BOUNDARY_PERSONAS]
        if not boundary:
            return 0.0
        return sum(1 for r in boundary if r.band_correct) / len(boundary)

    def summary(self) -> dict:
        return {
            "config": self.config_name,
            "n_personas": len(self.persona_results),
            "dchr": round(self.dchr, 3),
            "mad": round(self.mad, 2),
            "adodl": round(self.adodl, 3),
            "ashr_proxy": round(self.ashr_proxy, 3),
            "boundary_accuracy": round(self.boundary_accuracy, 3),
            "band_accuracy_by_severity": self.band_accuracy_by_severity,
            "per_persona": [
                {
                    "name": r.name,
                    "golden": r.golden_total,
                    "predicted": r.predicted_total,
                    "golden_band": r.golden_band.value,
                    "predicted_band": r.predicted_band.value,
                    "band_ok": r.band_correct,
                    "deviation": r.absolute_deviation,
                    "cr": round(r.cr_score, 3),
                    "symptom_hit_rate": round(r.symptom_hit_rate, 2),
                }
                for r in self.persona_results
            ],
        }


def load_talkdep_conversations(
    talkdep_dir: str | Path,
) -> list[TalkDepConversation]:
    """Load all TalkDep final conversations with golden scores.

    Args:
        talkdep_dir: Path to the TalkDep repo root
            (e.g., "data/TalkDep")

    Returns:
        List of parsed conversations sorted by golden BDI-II score (ascending).
    """
    talkdep_dir = Path(talkdep_dir)
    conv_dir = (
        talkdep_dir
        / "persona-development"
        / "conversation_generation"
        / "final_conversations"
    )

    if not conv_dir.exists():
        raise FileNotFoundError(f"TalkDep conversation dir not found: {conv_dir}")

    conversations = []
    for name, golden_total in sorted(GOLDEN_SCORES.items(), key=lambda x: x[1]):
        fname = f"{name.lower()}-final-conversation.txt"
        fpath = conv_dir / fname
        if not fpath.exists():
            logger.warning("Conversation not found for %s: %s", name, fpath)
            continue

        raw_text = fpath.read_text(encoding="utf-8")
        transcript = _parse_talkdep_transcript(raw_text)

        conversations.append(
            TalkDepConversation(
                name=name,
                transcript=transcript,
                golden_total=golden_total,
                golden_band=score_to_band(golden_total),
                golden_key_symptoms=GOLDEN_KEY_SYMPTOMS.get(name, []),
            )
        )

    logger.info("Loaded %d TalkDep conversations", len(conversations))
    return conversations


def _parse_talkdep_transcript(raw_text: str) -> str:
    """Parse a TalkDep conversation file into a clean transcript."""
    turns = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?:\d+\.\s*)?\*\*(\w+):\*\*\s*(.*)", line)
        if m:
            role_name = m.group(1)
            utterance = m.group(2).strip()
            if not utterance:
                continue
            role = "Interviewer" if role_name == "Therapist" else "Person"
            turns.append(f"{role}: {utterance}")
    return "\n".join(turns)


def _compute_symptom_hit_rate(
    golden_symptoms: list[str],
    predicted_symptoms: list[str],
) -> float:
    """Compute ASHR-proxy hit rate for a single persona.

    Per spec Section A.4:
      If K=0 and predicted=[]: 1.0
      If K=0 and predicted!=[]: 0.0
      If K>0: |predicted ∩ golden| / K
    """
    golden_set = set(canonicalize_symptom(s) for s in golden_symptoms)
    predicted_set = set(canonicalize_symptom(s) for s in predicted_symptoms)

    k = len(golden_set)
    if k == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0
    overlap = len(golden_set & predicted_set)
    return overlap / k


def evaluate_persona(
    name: str,
    predicted_total: int,
    predicted_top4: list[str],
    item_scores: Optional[dict[int, ItemScore]] = None,
) -> PersonaEvaluation:
    """Evaluate predictions for a single persona against golden truth."""
    golden_total = GOLDEN_SCORES[name]
    golden_band = score_to_band(golden_total)
    predicted_band = score_to_band(predicted_total)
    deviation = abs(predicted_total - golden_total)

    # ADODL closeness ratio: CR = (63 - deviation) / 63
    cr_score = (63 - deviation) / 63

    golden_symptoms = GOLDEN_KEY_SYMPTOMS.get(name, [])
    symptom_hit_rate = _compute_symptom_hit_rate(golden_symptoms, predicted_top4)

    item_dict = {}
    if item_scores:
        item_dict = {k: v.score for k, v in item_scores.items()}

    return PersonaEvaluation(
        name=name,
        golden_total=golden_total,
        golden_band=golden_band,
        predicted_total=predicted_total,
        predicted_band=predicted_band,
        band_correct=golden_band == predicted_band,
        absolute_deviation=deviation,
        cr_score=cr_score,
        predicted_top4=predicted_top4,
        symptom_hit_rate=symptom_hit_rate,
        item_scores=item_dict,
    )


def format_comparison_table(results: list[AblationResult]) -> str:
    """Format a comparison table across ablation configurations (Appendix A.6)."""
    if not results:
        return "No results to display."

    lines = []
    header = (
        f"{'Config':<20} {'DCHR':>6} {'MAD':>6} {'ADODL':>6} "
        f"{'ASHR':>6} {'Boundary':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r.config_name:<20} {r.dchr:>6.1%} {r.mad:>6.1f} "
            f"{r.adodl:>6.3f} {r.ashr_proxy:>6.1%} {r.boundary_accuracy:>8.1%}"
        )

    lines.append("")
    lines.append("Per-persona breakdown:")
    lines.append("")

    # Per-persona comparison (Appendix A.7)
    persona_names = [r.name for r in results[0].persona_results]
    header2 = f"{'Persona':<10} {'Golden':>6}"
    for r in results:
        short = r.config_name[:8]
        header2 += f" {short:>10}"
    lines.append(header2)
    lines.append("-" * len(header2))

    for i, name in enumerate(persona_names):
        golden = GOLDEN_SCORES[name]
        boundary = "*" if name in BOUNDARY_PERSONAS else " "
        line = f"{name + boundary:<10} {golden:>6}"
        for r in results:
            pr = r.persona_results[i]
            mark = "+" if pr.band_correct else "X"
            line += f" {pr.predicted_total:>6}{mark:>4}"
        lines.append(line)

    return "\n".join(lines)


def format_error_analysis(result: AblationResult) -> str:
    """Generate error analysis for a single ablation result (Section A.8)."""
    lines = [f"Error Analysis: {result.config_name}", "=" * 50]

    misclassified = [r for r in result.persona_results if not r.band_correct]
    if not misclassified:
        lines.append("All personas correctly classified.")
        return "\n".join(lines)

    lines.append(f"Misclassified: {len(misclassified)}/{len(result.persona_results)}")
    lines.append("")

    for r in misclassified:
        boundary = " (BOUNDARY)" if r.name in BOUNDARY_PERSONAS else ""
        direction = "over" if r.predicted_total > r.golden_total else "under"
        lines.append(
            f"  {r.name}{boundary}: golden={r.golden_total} ({r.golden_band.value}) "
            f"-> predicted={r.predicted_total} ({r.predicted_band.value}) "
            f"[{direction}-scored by {r.absolute_deviation}]"
        )

    return "\n".join(lines)


def compute_component_contribution(
    baseline: AblationResult,
    enhanced: AblationResult,
) -> dict:
    """Compute the contribution of adding a component (A(n-1) -> A(n)).

    Per Section A.8: report effect sizes, consistency, and boundary impact.
    """
    n = len(baseline.persona_results)
    improved = 0
    worsened = 0
    unchanged = 0

    for b, e in zip(baseline.persona_results, enhanced.persona_results):
        if e.absolute_deviation < b.absolute_deviation:
            improved += 1
        elif e.absolute_deviation > b.absolute_deviation:
            worsened += 1
        else:
            unchanged += 1

    # Boundary impact
    boundary_fixes = []
    boundary_breaks = []
    for b, e in zip(baseline.persona_results, enhanced.persona_results):
        if b.name in BOUNDARY_PERSONAS:
            if not b.band_correct and e.band_correct:
                boundary_fixes.append(b.name)
            elif b.band_correct and not e.band_correct:
                boundary_breaks.append(b.name)

    return {
        "baseline": baseline.config_name,
        "enhanced": enhanced.config_name,
        "dchr_delta": round(enhanced.dchr - baseline.dchr, 3),
        "mad_delta": round(enhanced.mad - baseline.mad, 2),
        "adodl_delta": round(enhanced.adodl - baseline.adodl, 3),
        "ashr_delta": round(enhanced.ashr_proxy - baseline.ashr_proxy, 3),
        "boundary_delta": round(enhanced.boundary_accuracy - baseline.boundary_accuracy, 3),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "boundary_fixes": boundary_fixes,
        "boundary_breaks": boundary_breaks,
    }
