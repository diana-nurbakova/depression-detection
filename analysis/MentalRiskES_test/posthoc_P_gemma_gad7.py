"""Post-Hoc Analysis P_gemma — Gemma GAD-7 re-scoring via OpenRouter.

Replaces the Llama-3.3-70B GAD-7 assessor with a Gemma model and a redesigned
prompt (specs/MentalRiskES/gemma_gad7_prompt_spec.md) that targets the
diagnosed over-prediction mechanisms:
  - removes the severity anchor (no "if anxiety primary, total >= 10" floor)
  - adds anti-ceiling guidance for item 2 (uncontrollable worry)
  - introduces per-item confidence estimation as implicit calibration
  - uses concrete frequency markers ("2-6 days out of 14") instead of labels

Runs against the FULL 82-round test set, fixing the round-30 truncation bug
that affected the live submission.

Output:
  output/mentalriskes_gemma_gad7/<model_short>/round{N}.json    server-format,
                                                                 keyed by session
  output/mentalriskes_gemma_gad7/<model_short>/raw.jsonl         full LLM
                                                                 responses with
                                                                 reasoning +
                                                                 confidence

Usage:
  python analysis/MentalRiskES_test/posthoc_P_gemma_gad7.py \
      --model google/gemma-3-27b-it \
      --max-sessions 0          # 0 = all; e.g. 1 for smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

logger = logging.getLogger("gemma_gad7")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts (verbatim from gemma_gad7_prompt_spec.md §2.1 and §2.2)
# v1 = original; v2 adds severe example, fixes item 5/6 indirect evidence,
#      severity calibration, refined confidence framing
# ─────────────────────────────────────────────────────────────────────────────
GAD7_SYSTEM_PROMPT_V1 = """You are a clinical psychologist completing the GAD-7 questionnaire based on a therapeutic conversation in Spanish. You must estimate how the patient would respond to each item based on the evidence in the conversation.

## THE GAD-7 QUESTIONNAIRE

The GAD-7 (Generalized Anxiety Disorder – 7) asks the patient:
"During the LAST TWO WEEKS, how often have you been bothered by the following problems?"

### Items

1. Sentirse nervioso/a, intranquilo/a o con los nervios de punta
   (Feeling nervous, anxious, or on edge)

2. No poder dejar de preocuparse o no poder controlar la preocupación
   (Not being able to stop or control worrying)

3. Preocuparse demasiado por diferentes cosas
   (Worrying too much about different things)

4. Dificultad para relajarse
   (Trouble relaxing)

5. Estar tan inquieto/a que es difícil permanecer sentado/a tranquilamente
   (Being so restless that it is hard to sit still)

6. Molestarse o ponerse irritable fácilmente
   (Becoming easily annoyed or irritable)

7. Sentir miedo como si algo terrible pudiera pasar
   (Feeling afraid, as if something awful might happen)

### Response Scale

Each item is scored 0 to 3 based on FREQUENCY over the past two weeks:

| Score | Label (Spanish) | Label (English) | What this means concretely |
|-------|-----------------|-----------------|---------------------------|
| 0 | Nunca | Not at all | The symptom is absent or occurred at most 1 day in two weeks |
| 1 | Varios días | Several days | The symptom occurred on roughly 2–6 days out of 14. It is present but intermittent — not a daily pattern |
| 2 | Más de la mitad de los días | More than half the days | The symptom occurred on roughly 7–11 days out of 14. It is a frequent pattern but with some symptom-free days |
| 3 | Casi todos los días | Nearly every day | The symptom occurred on 12–14 out of 14 days. It is pervasive and near-constant |

### Severity Bands (for reference, NOT for scoring)

| Total Score | Severity |
|-------------|----------|
| 0–4 | Minimal anxiety |
| 5–9 | Mild anxiety |
| 10–14 | Moderate anxiety |
| 15–21 | Severe anxiety |

## SCORING GUIDELINES

### General Principles

- Score each item INDEPENDENTLY based on the evidence for THAT specific symptom. Do not let your overall impression of anxiety severity determine individual item scores.
- The conversation is a WINDOW into the patient's past two weeks, not a complete record. Score based on what the conversation reveals about the patient's daily life and symptom patterns.
- Distinguish between in-session emotional expression (e.g., crying while discussing a difficult topic) and reported daily symptom patterns (e.g., "no puedo dormir por las preocupaciones").
- When the patient describes a symptom but does not specify frequency, default to score 1 (several days) rather than 2 or 3. Reserve score 3 for symptoms described with explicit high-frequency markers.

### Frequency Markers in Spanish (to help calibrate scores)

**Score 3 indicators** (near-daily — require STRONG evidence):
  "todo el tiempo", "siempre", "cada día", "constantemente", "no para nunca", "24/7", "día y noche"

**Score 2 indicators** (more than half the days):
  "muchas veces", "la mayoría de los días", "casi siempre", "muy a menudo", "frecuentemente"

**Score 1 indicators** (several days — the DEFAULT when a symptom is present):
  "a veces", "de vez en cuando", "algunos días", "hay días que", "no siempre pero..."

**Score 0 indicators** (absent):
  "no me pasa", "no tengo ese problema", "eso no", "para nada"
  Also score 0 when there is NO evidence at all for the item in the conversation.

### Item-Specific Guidance

**Item 2 (Uncontrollable worry):** This is the most commonly over-scored item. A patient describing worries or rumination does NOT automatically score 3. Score 3 ONLY if the patient explicitly describes inability to stop worrying as a near-daily, pervasive pattern. Describing worry loops ("un bucle", "ideas dando vueltas") during a therapy session is evidence for the item being present (score >= 1), but does not by itself indicate near-daily frequency. Distinguish between:
  - "A veces me preocupo y no puedo parar" -> Score 1
  - "Muchos días no puedo controlar la preocupación" -> Score 2
  - "Todos los días, desde que me levanto hasta que me acuesto, no paro de preocuparme" -> Score 3

**Item 5 (Psychomotor restlessness):** This is the most commonly ABSENT symptom. It requires physical restlessness (inability to sit still, pacing, fidgeting), not just mental agitation. Most anxious patients score 0–1 on this item. Score 2–3 only with clear physical evidence.

**Item 7 (Dread/fear):** Distinguish between specific situational fears (which count less toward GAD) and pervasive, non-specific dread ("como si algo malo fuera a pasar"). Score the latter.

### Confidence Assessment

For each item, also rate your confidence in the score:
- **HIGH:** Direct, explicit evidence in the conversation for both the symptom AND its frequency
- **MEDIUM:** The symptom is clearly present but frequency must be inferred from context
- **LOW:** Scoring is based on indirect evidence, absence of counter-evidence, or general clinical impression rather than specific statements

## EXAMPLE ASSESSMENTS

### Example 1: Moderate Anxiety (GAD-7 Total = 11)

**Context:** Female patient, 35 years old, in ACT therapy. Primary complaint is work-related stress and worry about her children's wellbeing. She describes difficulty sleeping due to worry, tension in her shoulders, and being snappy with her partner.

| Item | Score | Confidence | Reasoning |
|------|-------|------------|-----------|
| 1. Nervousness | 2 | HIGH | Patient says "la mayoría de los días estoy tensa, noto la presión en los hombros" — explicit frequency marker for more than half the days |
| 2. Uncontrollable worry | 2 | MEDIUM | Describes worry about children and work as frequent but not constant: "muchas veces me pongo a pensar y me cuesta parar." Does not describe it as every day -> score 2, not 3 |
| 3. Excessive worry | 2 | HIGH | Multiple worry domains (work, children, partner relationship) with clear distress |
| 4. Difficulty relaxing | 2 | MEDIUM | Reports shoulder tension and difficulty sleeping, suggesting persistent difficulty relaxing |
| 5. Restlessness | 0 | MEDIUM | No evidence of physical restlessness. Mental agitation is captured in items 1–3 |
| 6. Irritability | 2 | HIGH | "Estoy muy borde con mi pareja, me enfado por cualquier cosa" — clear and frequent |
| 7. Dread | 1 | LOW | Some worry about the future but no pervasive sense of dread. "A veces tengo miedo de que algo pase con los niños" — intermittent, specific rather than pervasive |

### Example 2: Mild Anxiety (GAD-7 Total = 6)

**Context:** Male patient, 28, in therapy for adjustment difficulties after a job change. Some worry about performance but generally coping. Describes occasional tension and sleep difficulties.

| Item | Score | Confidence | Reasoning |
|------|-------|------------|-----------|
| 1. Nervousness | 1 | HIGH | "Hay días que estoy nervioso, sobre todo cuando tengo reuniones" — intermittent, situational |
| 2. Uncontrollable worry | 1 | MEDIUM | Some rumination about work performance but describes it as manageable: "a veces le doy vueltas" |
| 3. Excessive worry | 1 | MEDIUM | Worry focused on one domain (work). Not multiple domains |
| 4. Difficulty relaxing | 1 | LOW | Mentions some tension but no strong evidence of persistent relaxation difficulty |
| 5. Restlessness | 0 | HIGH | No evidence |
| 6. Irritability | 1 | MEDIUM | Mentions being "algo más irritable" but not as a dominant complaint |
| 7. Dread | 1 | LOW | Some anticipatory anxiety about the future at work, but mild |

## OUTPUT FORMAT

Respond with ONLY a JSON object in the following format. No preamble, no markdown formatting, no explanation outside the JSON:

{
  "items": [
    {
      "item_number": 1,
      "item_text": "Sentirse nervioso/a, intranquilo/a o con los nervios de punta",
      "evidence": "Brief quote or description of relevant evidence from the conversation",
      "frequency_inference": "How often this symptom occurs based on the evidence",
      "score": 0,
      "confidence": "HIGH"
    }
  ],
  "total_score": 0,
  "severity_band": "minimal",
  "overall_confidence": "MEDIUM",
  "scoring_notes": "Brief note on any scoring decisions that were difficult or uncertain"
}
"""

GAD7_SYSTEM_PROMPT_V2 = """You are a clinical psychologist completing the GAD-7 questionnaire based on a therapeutic conversation in Spanish. You must estimate how the patient would respond to each item based on the evidence in the conversation.

## THE GAD-7 QUESTIONNAIRE

The GAD-7 (Generalized Anxiety Disorder – 7) asks the patient:
"During the LAST TWO WEEKS, how often have you been bothered by the following problems?"

### Items

1. Sentirse nervioso/a, intranquilo/a o con los nervios de punta
   (Feeling nervous, anxious, or on edge)

2. No poder dejar de preocuparse o no poder controlar la preocupación
   (Not being able to stop or control worrying)

3. Preocuparse demasiado por diferentes cosas
   (Worrying too much about different things)

4. Dificultad para relajarse
   (Trouble relaxing)

5. Estar tan inquieto/a que es difícil permanecer sentado/a tranquilamente
   (Being so restless that it is hard to sit still)

6. Molestarse o ponerse irritable fácilmente
   (Becoming easily annoyed or irritable)

7. Sentir miedo como si algo terrible pudiera pasar
   (Feeling afraid, as if something awful might happen)

### Response Scale

Each item is scored 0 to 3 based on FREQUENCY over the past two weeks:

| Score | Label (Spanish) | Label (English) | What this means concretely |
|-------|-----------------|-----------------|---------------------------|
| 0 | Nunca | Not at all | The symptom is absent or occurred at most 1 day in two weeks |
| 1 | Varios días | Several days | The symptom occurred on roughly 2-6 days out of 14. It is present but intermittent - not a daily pattern |
| 2 | Más de la mitad de los días | More than half the days | The symptom occurred on roughly 7-11 days out of 14. It is a frequent pattern but with some symptom-free days |
| 3 | Casi todos los días | Nearly every day | The symptom occurred on 12-14 out of 14 days. It is pervasive and near-constant |

### Severity Bands (for reference, NOT for scoring)

| Total Score | Severity |
|-------------|----------|
| 0-4 | Minimal anxiety |
| 5-9 | Mild anxiety |
| 10-14 | Moderate anxiety |
| 15-21 | Severe anxiety |

## SCORING GUIDELINES

### General Principles

- Score each item INDEPENDENTLY based on the evidence for THAT specific symptom. Do not let your overall impression of anxiety severity determine individual item scores.
- The conversation is a WINDOW into the patient's past two weeks, not a complete record. Score based on what the conversation reveals about the patient's daily life and symptom patterns.
- Distinguish between in-session emotional expression (e.g., crying while discussing a difficult topic) and reported daily symptom patterns (e.g., "no puedo dormir por las preocupaciones").
- When the patient describes a symptom but does not specify frequency, default to score 1 (several days) rather than assuming 0 or jumping to 3.
- **Patients in therapy for anxiety typically score in the moderate to severe range (total 10-21).** If the patient is clearly anxious, with multiple symptoms discussed, a total below 10 should prompt you to re-examine whether you are under-scoring.

### Frequency Markers in Spanish (to help calibrate scores)

**Score 3 indicators** (near-daily - requires clear evidence of pervasiveness):
  "todo el tiempo", "siempre", "cada día", "constantemente", "no para nunca", "24/7", "día y noche"

**Score 2 indicators** (more than half the days):
  "muchas veces", "la mayoría de los días", "casi siempre", "muy a menudo", "frecuentemente"

**Score 1 indicators** (several days - the DEFAULT when a symptom is present but frequency unclear):
  "a veces", "de vez en cuando", "algunos días", "hay días que", "no siempre pero..."

**Score 0 indicators** (absent):
  "no me pasa", "no tengo ese problema", "eso no", "para nada"
  Also score 0 when there is NO evidence at all for the item in the conversation.

### Item-Specific Guidance

**Item 2 (Uncontrollable worry):** A patient describing worries or rumination does NOT automatically score 3. Score 3 ONLY if the patient explicitly describes inability to stop worrying as a near-daily, pervasive pattern. Describing worry loops ("un bucle", "ideas dando vueltas") during a therapy session is evidence for the item being present (score >= 1), but does not by itself indicate near-daily frequency. Distinguish between:
  - "A veces me preocupo y no puedo parar" -> Score 1
  - "Muchos días no puedo controlar la preocupación" -> Score 2
  - "Todos los días, desde que me levanto hasta que me acuesto, no paro de preocuparme" -> Score 3

**Item 5 (Restlessness):** This item captures BOTH physical AND mental restlessness that manifests physically. In severely anxious patients, it is often present at high levels. Look for INDIRECT evidence, not just explicit statements about fidgeting:
  - Sleep disruption due to racing mind -> suggests difficulty staying still (score >= 1)
  - Descriptions of agitation, inability to concentrate on tasks, needing to move or leave situations -> score 2
  - Patient describes being unable to sit through meals, work, or conversations; pacing; physical tension that prevents rest -> score 3
  - "No puedo quedarme quieto/a", "me muevo todo el rato", "no puedo estar sentado/a" -> direct evidence
  - "No puedo parar", "me siento acelerado/a", "voy de un lado a otro" -> indirect but valid
  - Note: in severely anxious patients (total >= 15 on other items), score >= 2 is common, not exceptional

**Item 6 (Irritability):** Irritability is extremely common in anxiety disorders but patients often describe it indirectly. Look for:
  - Conflict descriptions: arguments with partner, family, colleagues -> score >= 1
  - "Estoy muy borde", "salto por cualquier cosa", "tengo poca paciencia" -> direct evidence
  - "Me enfado fácilmente", "estoy de mal humor", "no aguanto nada" -> score 2-3
  - Descriptions of interpersonal friction, snapping at children/partner, road rage, workplace conflict -> score >= 1
  - If the patient describes irritability as a recurring theme across conversations -> score 2-3
  - Note: patients may feel shame about irritability and understate it. If there are MULTIPLE references to interpersonal conflict or anger, score >= 2 even without explicit frequency markers.

**Item 7 (Dread/fear):** Distinguish between specific situational fears (which count less toward GAD) and pervasive, non-specific dread ("como si algo malo fuera a pasar"). Score the latter.

### Confidence Assessment

For each item, rate your confidence. This reflects how PRECISELY you can estimate the FREQUENCY, not just whether the symptom exists:
- **HIGH:** The conversation contains direct evidence of both the symptom AND how often it occurs (e.g., "every night I can't sleep" = high confidence in score 3 for that item)
- **MEDIUM:** The symptom is clearly present but the frequency is inferred from context or severity rather than explicit statements
- **LOW:** Scoring is based on indirect evidence, clinical inference from the overall presentation, or absence of counter-evidence

Note: it is possible to give a HIGH score (2 or 3) with MEDIUM or even LOW confidence. Confidence is about evidence quality, not score magnitude.

## EXAMPLE ASSESSMENTS

### Example 1: Moderate Anxiety (GAD-7 Total = 11)

**Context:** Female patient, 35 years old, in ACT therapy. Primary complaint is work-related stress and worry about her children's wellbeing. She describes difficulty sleeping due to worry, tension in her shoulders, and being snappy with her partner.

| Item | Score | Confidence | Reasoning |
|------|-------|------------|-----------|
| 1. Nervousness | 2 | HIGH | "la mayoría de los días estoy tensa, noto la presión en los hombros" - explicit frequency marker for more than half the days |
| 2. Uncontrollable worry | 2 | MEDIUM | "muchas veces me pongo a pensar y me cuesta parar" - frequent but not constant |
| 3. Excessive worry | 2 | HIGH | Multiple worry domains (work, children, partner) with clear distress |
| 4. Difficulty relaxing | 2 | MEDIUM | Reports shoulder tension and difficulty sleeping, suggesting persistent difficulty relaxing |
| 5. Restlessness | 0 | MEDIUM | No evidence of physical restlessness or agitation. Mental tension is captured in items 1-4 |
| 6. Irritability | 2 | HIGH | "Estoy muy borde con mi pareja, me enfado por cualquier cosa" - clear and frequent |
| 7. Dread | 1 | LOW | "A veces tengo miedo de que algo pase con los niños" - intermittent, specific rather than pervasive |

### Example 2: Mild Anxiety (GAD-7 Total = 6)

**Context:** Male patient, 28, in therapy for adjustment difficulties after a job change. Some worry about performance but generally coping. Describes occasional tension and sleep difficulties.

| Item | Score | Confidence | Reasoning |
|------|-------|------------|-----------|
| 1. Nervousness | 1 | HIGH | "Hay días que estoy nervioso, sobre todo cuando tengo reuniones" - intermittent, situational |
| 2. Uncontrollable worry | 1 | MEDIUM | "a veces le doy vueltas" - manageable rumination |
| 3. Excessive worry | 1 | MEDIUM | Worry focused on one domain (work). Not multiple domains |
| 4. Difficulty relaxing | 1 | LOW | Mentions some tension but no strong evidence of persistent relaxation difficulty |
| 5. Restlessness | 0 | HIGH | No evidence |
| 6. Irritability | 1 | MEDIUM | Mentions being "algo más irritable" but not as a dominant complaint |
| 7. Dread | 1 | LOW | Some anticipatory anxiety about the future at work, but mild |

### Example 3: Severe Anxiety (GAD-7 Total = 17)

**Context:** Female patient, 42, in ACT therapy for generalized anxiety with panic features. She describes pervasive worry affecting all life domains, constant physical tension, sleep disruption every night, conflict with her family due to irritability, and avoidance of social situations due to fear. She has been anxious for several years and the current episode has intensified over the past months.

| Item | Score | Confidence | Reasoning |
|------|-------|------------|-----------|
| 1. Nervousness | 3 | HIGH | "Estoy nerviosa todo el día, desde que me levanto ya estoy con el nudo en el estómago" - near-daily, with physical manifestation |
| 2. Uncontrollable worry | 3 | HIGH | "No puedo parar de darle vueltas, me acuesto preocupada y me levanto preocupada. Es todos los días." |
| 3. Excessive worry | 3 | MEDIUM | Worries span work, health, children, finances, social judgment - multiple domains with high frequency |
| 4. Difficulty relaxing | 2 | MEDIUM | Reports sleep disruption and constant tension, but also describes some moments of calm during therapy exercises -> not quite "nearly every day" |
| 5. Restlessness | 2 | MEDIUM | Describes being "acelerada", unable to concentrate at work, needing to get up and move during long meetings. Agitation is clearly frequent |
| 6. Irritability | 2 | HIGH | "Mi marido dice que salto por todo, y tiene razón. Estamos discutiendo mucho." Multiple references to family conflict |
| 7. Dread | 2 | MEDIUM | "Tengo la sensación de que algo malo va a pasar, como si estuviera esperando una mala noticia." Frequent recurring theme |

## OUTPUT FORMAT

Respond with ONLY a JSON object in the following format. No preamble, no markdown formatting, no explanation outside the JSON:

{
  "items": [
    {
      "item_number": 1,
      "item_text": "Sentirse nervioso/a, intranquilo/a o con los nervios de punta",
      "evidence": "Brief quote or description of relevant evidence from the conversation",
      "frequency_inference": "How often this symptom occurs based on the evidence",
      "score": 0,
      "confidence": "HIGH"
    }
  ],
  "total_score": 0,
  "severity_band": "minimal",
  "overall_confidence": "MEDIUM",
  "scoring_notes": "Brief note on any scoring decisions that were difficult or uncertain"
}
"""


PROMPT_VERSIONS = {"v1": GAD7_SYSTEM_PROMPT_V1, "v2": GAD7_SYSTEM_PROMPT_V2}
GAD7_SYSTEM_PROMPT = GAD7_SYSTEM_PROMPT_V1  # legacy alias for unchanged callers


GAD7_USER_PROMPT_TEMPLATE = """Here is the therapeutic conversation so far (in Spanish). Based on this conversation, complete the GAD-7 assessment for this patient.

## CONVERSATION

{transcript}

## INSTRUCTIONS

1. Read the entire conversation carefully
2. For each GAD-7 item, find evidence in the conversation
3. Estimate the frequency of each symptom over the PAST TWO WEEKS
4. Assign a score (0-3) and confidence level (HIGH/MEDIUM/LOW) for each item
5. When frequency is unclear but the symptom is present, default to score 1 (several days)
6. Reserve score 3 (nearly every day) for symptoms with explicit daily/constant frequency markers

Respond with the JSON assessment only.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cohort loaders — test (multi-session per round file), trial / simulated
# (single-session "trial"-keyed round files; simulated = one dir per session)
# ─────────────────────────────────────────────────────────────────────────────
def load_test_rounds(test_dir: Path) -> list[tuple[int, dict]]:
    """Walk round_*.json sorted by round id; return [(round, data_per_session)]."""
    rounds = []
    for fp in sorted(test_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        rnd = int(fp.stem.split("_")[1])
        with open(fp, encoding="utf-8") as fh:
            data = json.load(fh)
        rounds.append((rnd, data))
    return rounds


def build_session_transcripts(rounds: list[tuple[int, dict]]) -> dict[str, list[tuple[int, str, str]]]:
    """Returns {session_id: [(round, role, text), ...]} per spec format_for_prompt."""
    sessions: dict[str, list[tuple[int, str, str]]] = {}
    for rnd, data in rounds:
        for sid, payload in data.items():
            therapist = payload.get("therapist_response")
            patient = payload.get("patient_input", "")
            if therapist:
                sessions.setdefault(sid, []).append((rnd, "therapist", therapist))
            if patient:
                sessions.setdefault(sid, []).append((rnd, "patient", patient))
    return sessions


def load_trial_session(trial_dir: Path, session_id: str) -> list[tuple[int, str, str]]:
    """Read round_*.json with {"trial": {round, patient_input, ...}} format."""
    turns: list[tuple[int, str, str]] = []
    for fp in sorted(trial_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        rnd = int(fp.stem.split("_")[1])
        with open(fp, encoding="utf-8") as fh:
            payload = json.load(fh).get("trial", {})
        therapist = payload.get("therapist_response")
        if therapist:
            turns.append((rnd, "therapist", therapist))
        patient = payload.get("patient_input", "")
        if patient:
            turns.append((rnd, "patient", patient))
    return turns


def load_simulated_task1(sim_root: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Simulated Task 1 sessions: each dir holds round_*.json with {<sid>: {round, patient_input}}."""
    out: dict[str, list[tuple[int, str, str]]] = {}
    if not sim_root.exists():
        return out
    for d in sorted(sim_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "metadata.json").exists():
            continue
        sid = d.name
        turns: list[tuple[int, str, str]] = []
        for fp in sorted(d.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            rnd = int(fp.stem.split("_")[1])
            with open(fp, encoding="utf-8") as fh:
                payload = json.load(fh)
            # Task 1 simulated has the form {<sid>: {round, patient_input, [therapist_response]}}
            entry = payload.get(sid) or next(iter(payload.values()))
            therapist = entry.get("therapist_response")
            if therapist:
                turns.append((rnd, "therapist", therapist))
            patient = entry.get("patient_input", "")
            if patient:
                turns.append((rnd, "patient", patient))
        if turns:
            out[sid] = turns
    return out


def format_transcript_up_to(turns: list[tuple[int, str, str]], up_to_round: int) -> str:
    lines = []
    for r, role, text in turns:
        if r > up_to_round:
            break
        lines.append(f"[Round {r} — {role.upper()}]: {text}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter client (OpenAI SDK)
# ─────────────────────────────────────────────────────────────────────────────
def _openrouter_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def parse_gad7_response(raw: str) -> dict:
    """Parse the JSON, validate, normalise totals."""
    text = _strip_fences(raw)
    data = json.loads(text)
    items = data.get("items", [])
    if len(items) != 7:
        raise ValueError(f"Expected 7 items, got {len(items)}")
    scores = []
    confidences = []
    for entry in items:
        s = int(entry["score"])
        if not 0 <= s <= 3:
            raise ValueError(f"Score out of range: {s}")
        scores.append(s)
        confidences.append(str(entry.get("confidence", "LOW")).upper())
    computed_total = sum(scores)
    if int(data.get("total_score", computed_total)) != computed_total:
        logger.warning("Stated total %s != computed %s; trusting items",
                       data.get("total_score"), computed_total)
    data["scores"] = scores
    data["confidences"] = confidences
    data["total_score"] = computed_total
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_cohort(args: argparse.Namespace) -> tuple[dict[str, list], str]:
    if args.cohort == "test":
        test_dir = REPO_ROOT / "data/MentalRiskES-2026/test/task1/test/data"
        rounds = load_test_rounds(test_dir)
        sessions = build_session_transcripts(rounds)
        logger.info("Loaded %d rounds, %d sessions from %s (cohort=test)",
                    len(rounds), len(sessions), test_dir)
        return sessions, "test"
    if args.cohort == "trial":
        trial_dir = REPO_ROOT / "data/MentalRiskES-2026/task1_trial/data"
        turns = load_trial_session(trial_dir, "trial")
        logger.info("Loaded %d turns from %s (cohort=trial)", len(turns), trial_dir)
        return {"trial": turns}, "trial"
    if args.cohort == "simulated":
        sim_root = REPO_ROOT / "output/mentalriskes/data_prep/simulated/task1"
        sessions = load_simulated_task1(sim_root)
        logger.info("Loaded %d simulated sessions from %s (cohort=simulated)",
                    len(sessions), sim_root)
        return sessions, "simulated"
    raise ValueError(f"Unknown --cohort {args.cohort!r}")


def run(args: argparse.Namespace) -> None:
    session_turns, cohort_label = _resolve_cohort(args)

    # When a patient appears in round R, we score the conversation truncated AT round R.
    # The "live" assessor sees patient input but no therapist response yet for the current round.
    sessions_to_run = list(session_turns.keys())
    if args.max_sessions:
        sessions_to_run = sessions_to_run[: args.max_sessions]

    model = args.model
    prompt_version = args.prompt_version
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown --prompt-version {prompt_version!r}; choose from {list(PROMPT_VERSIONS)}")
    system_prompt = PROMPT_VERSIONS[prompt_version]

    out_root = REPO_ROOT / "output/mentalriskes_gemma_gad7"
    model_short = model.replace("/", "_").replace(":", "_")
    # 'test' + 'v1' keeps the legacy bare-model path so existing analysis still works.
    if cohort_label == "test" and prompt_version == "v1":
        suffix = model_short
    elif cohort_label == "test":
        suffix = f"{model_short}__{prompt_version}"
    else:
        suffix = f"{model_short}__{prompt_version}__{cohort_label}"
    model_dir = out_root / suffix
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / "raw.jsonl"

    # Resume: skip rounds already in raw.jsonl
    done: set[tuple[str, int]] = set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["session"], rec["round"]))
                except Exception:
                    continue
        logger.info("Resume: %d (session, round) pairs already done", len(done))

    client = _openrouter_client()
    raw_fh = open(raw_path, "a", encoding="utf-8")

    # Per-round predictions accumulate so we can write server-format JSONs
    per_round_preds: dict[int, list[dict]] = {}

    total_calls = 0
    started = time.monotonic()

    try:
        for sid in sessions_to_run:
            turns = session_turns[sid]
            patient_round_ids = sorted({r for r, role, _ in turns if role == "patient"})
            for rnd_id in patient_round_ids:
                if (sid, rnd_id) in done:
                    continue
                transcript = format_transcript_up_to(turns, rnd_id)
                t0 = time.monotonic()
                attempt_err = None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        # Gemma instruction-tuned models on Google AI Studio do not
                        # accept a separate system role; merge the system prompt
                        # into the user message.
                        is_gemma = "gemma" in model.lower()
                        if is_gemma:
                            messages = [
                                {
                                    "role": "user",
                                    "content": system_prompt
                                    + "\n\n---\n\n"
                                    + GAD7_USER_PROMPT_TEMPLATE.format(transcript=transcript),
                                }
                            ]
                        else:
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": GAD7_USER_PROMPT_TEMPLATE.format(transcript=transcript)},
                            ]
                        kwargs = {
                            "model": model,
                            "messages": messages,
                            "temperature": 0.1,
                            "max_tokens": args.max_tokens,
                        }
                        if not is_gemma:
                            # Gemma routes don't honour response_format on Google AI Studio
                            kwargs["response_format"] = {"type": "json_object"}
                        resp = client.chat.completions.create(**kwargs)
                        content = resp.choices[0].message.content or ""
                        parsed = parse_gad7_response(content)
                        elapsed = time.monotonic() - t0
                        usage = getattr(resp, "usage", None)
                        record = {
                            "model": model,
                            "prompt_version": prompt_version,
                            "cohort": cohort_label,
                            "session": sid,
                            "round": rnd_id,
                            "elapsed_s": round(elapsed, 2),
                            "scores": parsed["scores"],
                            "confidences": parsed["confidences"],
                            "total_score": parsed["total_score"],
                            "severity_band": parsed.get("severity_band"),
                            "overall_confidence": parsed.get("overall_confidence"),
                            "scoring_notes": parsed.get("scoring_notes"),
                            "raw_items": parsed.get("items"),
                            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                        }
                        raw_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_fh.flush()
                        per_round_preds.setdefault(rnd_id, []).append({
                            "id": sid,
                            "round": rnd_id,
                            "prediction": {"GAD-7": parsed["scores"]},
                        })
                        total_calls += 1
                        if total_calls % 5 == 0:
                            logger.info("done=%d  last=%s/R%d  total=%d  band=%s  elapsed=%.1fs",
                                        total_calls, sid, rnd_id, parsed["total_score"],
                                        parsed.get("severity_band"), elapsed)
                        break
                    except Exception as e:
                        attempt_err = e
                        wait = min(60, 5 * attempt)
                        logger.warning("call failed for %s R%d (attempt %d/%d): %s - sleep %ds",
                                       sid, rnd_id, attempt, args.max_retries, e, wait)
                        time.sleep(wait)
                else:
                    logger.error("giving up on %s R%d after %d attempts: %s", sid, rnd_id, args.max_retries, attempt_err)

                # Be polite to the free tier (20 req/min); paid tiers tolerate higher RPS
                if args.rate_limit_delay > 0:
                    time.sleep(args.rate_limit_delay)
    finally:
        raw_fh.close()

    # Write server-format per-round JSONs (mirror our analysis input format).
    if per_round_preds:
        for rnd_id, preds in per_round_preds.items():
            with open(model_dir / f"round{rnd_id}.json", "w", encoding="utf-8") as fh:
                json.dump([{"predictions": preds, "emissions": {}}], fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %d round JSONs to %s", len(per_round_preds), model_dir)

    elapsed = time.monotonic() - started
    logger.info("Run complete: %d new calls in %.1fs (model=%s)", total_calls, elapsed, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-27b-it",
                        help="OpenRouter model id (omit ':free' suffix here; the API resolves variants)")
    parser.add_argument("--prompt-version", default="v1", choices=list(PROMPT_VERSIONS),
                        help="Which prompt to use. v2 adds a severe example, indirect-evidence "
                             "guidance for items 5/6, and refined confidence framing.")
    parser.add_argument("--cohort", default="test", choices=("test", "trial", "simulated"),
                        help="Which cohort to evaluate on (test/trial/simulated).")
    parser.add_argument("--max-sessions", type=int, default=0, help="Cap sessions for smoke test (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rate-limit-delay", type=float, default=3.5,
                        help="Seconds between calls; 3.5s keeps us under the 20 req/min free tier")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
