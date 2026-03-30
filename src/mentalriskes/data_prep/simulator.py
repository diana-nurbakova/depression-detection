"""TalkDep-style therapeutic conversation simulator for MentalRiskES.

Generates synthetic therapeutic conversations in Spanish following the
MentalRiskES format. Uses LLM to play both patient and therapist roles
with ACT-based therapeutic framework.

Key differences from eRisk TalkDep:
- Language: Spanish (not English)
- Instruments: PHQ-9, GAD-7, CompACT-10 (not BDI-II)
- Therapy: ACT-based (not generic depression interview)
- Format: Round-based with therapist response options (for Task 2)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Patient profiles spanning different severity levels and presentations
PATIENT_PROFILES = [
    {
        "id": "anx_academic",
        "description": "University student with academic anxiety and concentration difficulties",
        "phq9_range": (5, 12),  # mild-moderate depression
        "gad7_range": (10, 18),  # moderate-severe anxiety
        "compact10_profile": "low_flexibility",  # experiential avoidance
        "presenting_issue": "Ansiedad académica, dificultad para concentrarse, presión familiar",
        "personality": "Perfeccionista, autocrítico/a, evitativo/a ante el malestar",
    },
    {
        "id": "dep_loss",
        "description": "Adult with depression following a significant loss",
        "phq9_range": (12, 20),  # moderate-severe depression
        "gad7_range": (5, 10),  # mild anxiety
        "compact10_profile": "mixed_flexibility",
        "presenting_issue": "Duelo, pérdida de interés, aislamiento social, insomnio",
        "personality": "Introvertido/a, reflexivo/a, tendencia a rumiar",
    },
    {
        "id": "anx_social",
        "description": "Young adult with social anxiety and low self-esteem",
        "phq9_range": (8, 15),  # mild-moderate depression
        "gad7_range": (12, 20),  # moderate-severe anxiety
        "compact10_profile": "low_flexibility",
        "presenting_issue": "Ansiedad social, miedo al rechazo, evitación de situaciones sociales",
        "personality": "Tímido/a, sensible a la evaluación, hipervigilante",
    },
    {
        "id": "dep_burnout",
        "description": "Professional with burnout and exhaustion",
        "phq9_range": (10, 18),  # moderate depression
        "gad7_range": (8, 14),  # moderate anxiety
        "compact10_profile": "moderate_flexibility",
        "presenting_issue": "Agotamiento laboral, pérdida de sentido, irritabilidad, fatiga crónica",
        "personality": "Responsable, dificultad para poner límites, autocrítico/a",
    },
    {
        "id": "anx_health",
        "description": "Person with health anxiety and somatic symptoms",
        "phq9_range": (4, 10),  # minimal-mild depression
        "gad7_range": (14, 21),  # severe anxiety
        "compact10_profile": "low_flexibility",
        "presenting_issue": "Ansiedad por la salud, síntomas somáticos, catastrofismo, búsqueda de reaseguramiento",
        "personality": "Controlador/a, necesidad de certeza, hipervigilante con el cuerpo",
    },
    {
        "id": "dep_mild",
        "description": "Adolescent with emerging depressive symptoms",
        "phq9_range": (5, 9),  # mild depression
        "gad7_range": (3, 7),  # minimal-mild anxiety
        "compact10_profile": "moderate_flexibility",
        "presenting_issue": "Apatía, desmotivación escolar, conflictos familiares",
        "personality": "Reservado/a, respuestas cortas, algo defensivo/a",
    },
]

# ACT therapeutic phases
ACT_PHASES = [
    "engagement",  # Building rapport, initial validation
    "creative_hopelessness",  # Exploring failed control strategies
    "acceptance",  # Willingness to experience discomfort
    "defusion",  # Unhooking from thoughts
    "present_moment",  # Mindful contact with experience
    "self_as_context",  # Observer perspective
    "values",  # Clarifying what matters
    "committed_action",  # Concrete behavioral steps
    "integration",  # Consolidating skills
    "closing",  # Summarizing, scheduling follow-up
]

PATIENT_SYSTEM_PROMPT = """\
Eres un paciente simulado en una sesión de terapia en español. Tu perfil:

PROBLEMA PRINCIPAL: {presenting_issue}
PERSONALIDAD: {personality}
NIVEL DE MALESTAR: {distress_level}

INSTRUCCIONES:
- Responde de forma natural, como una persona real en terapia
- Usa lenguaje informal y coloquial español
- Incluye hesitaciones, frases incompletas, expresiones emocionales
- No seas demasiado articulado/a ni demasiado técnico/a
- Varía la longitud de tus respuestas (15-80 palabras)
- Si el terapeuta introduce una técnica ACT, reacciona según tu personalidad:
  * Puedes mostrar resistencia inicial pero ir abriéndote gradualmente
  * Puedes adoptar metáforas que resuenen contigo
  * Puedes hacer preguntas si algo no queda claro
- NO menciones que eres una IA
- Responde SOLO como el paciente, sin instrucciones ni metadatos"""

THERAPIST_SYSTEM_PROMPT = """\
Eres un terapeuta ACT (Terapia de Aceptación y Compromiso) experimentado \
conduciendo una sesión en español.

FASE TERAPÉUTICA ACTUAL: {phase}
PERFIL DEL PACIENTE: {profile_summary}

INSTRUCCIONES:
- Usa un enfoque ACT genuino basado en los 6 procesos del hexaflex
- Adapta tu intervención a la fase terapéutica actual
- Responde en 40-120 palabras
- Usa validación, preguntas abiertas, metáforas cuando sea apropiado
- NO des consejos directivos ("tienes que", "deberías")
- NO hagas más de 2 preguntas por turno
- Crea espacio para la experiencia del paciente
- Responde SOLO como el terapeuta, sin instrucciones ni metadatos"""

DISTRACTOR_SYSTEM_PROMPT = """\
Genera una respuesta de terapeuta que sea PLAUSIBLE pero SUBÓPTIMA para \
esta situación terapéutica en español.

TIPO DE ERROR: {error_type}
FASE ACTUAL: {phase}

INSTRUCCIONES SEGÚN TIPO DE ERROR:
- "premature_advice": Da consejo directivo usando "tienes que" o "deberías". \
  Incluye soluciones concretas sin explorar la experiencia del paciente.
- "phase_mismatch": Usa una técnica ACT correcta pero de una fase incorrecta. \
  Por ejemplo, exploración de valores cuando el paciente está en crisis.
- "question_overload": Haz 4+ preguntas seguidas. Muestra interés pero abruma.
- "surface_validation": Valida emocionalmente pero redirige inmediatamente a acción. \
  "Entiendo que te sientas así... ¿Qué te gustaría hacer ahora?"

Genera SOLO la respuesta del terapeuta, 40-120 palabras, en español."""


@dataclass
class SimulatedTurn:
    round_number: int
    patient_input: str
    therapist_response: str = ""  # Selected/correct response
    options: dict[str, str] = field(default_factory=dict)  # For Task 2 MC
    correct_option: int = 0  # 1, 2, or 3
    phase: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class SimulatedSession:
    session_id: str
    profile: dict
    turns: list[SimulatedTurn] = field(default_factory=list)
    target_scores: dict = field(default_factory=dict)  # PHQ-9, GAD-7, CompACT-10


def generate_session(
    profile: dict,
    llm_client,
    n_rounds: int = 15,
    generate_mc: bool = True,
    seed: int | None = None,
) -> SimulatedSession:
    """Generate a complete simulated therapeutic session.

    Args:
        profile: Patient profile dict from PATIENT_PROFILES.
        llm_client: LLM client with chat_completion() method.
        n_rounds: Number of conversation rounds.
        generate_mc: Whether to generate Task 2 MC options.
        seed: Random seed.

    Returns:
        SimulatedSession with all turns and optional MC options.
    """
    rng = random.Random(seed)

    # Determine target scores
    phq9_total = rng.randint(*profile["phq9_range"])
    gad7_total = rng.randint(*profile["gad7_range"])

    session = SimulatedSession(
        session_id=f"sim_{profile['id']}_{seed or rng.randint(0, 9999)}",
        profile=profile,
        target_scores={
            "phq9_total": phq9_total,
            "gad7_total": gad7_total,
            "compact10_profile": profile["compact10_profile"],
        },
    )

    # Map rounds to ACT phases
    phase_schedule = _assign_phases(n_rounds)

    # Build conversation
    conversation_history: list[dict] = []
    distress_level = _distress_descriptor(phq9_total, gad7_total)

    for round_n in range(1, n_rounds + 1):
        phase = phase_schedule[round_n - 1]

        # Generate patient response
        patient_system = PATIENT_SYSTEM_PROMPT.format(
            presenting_issue=profile["presenting_issue"],
            personality=profile["personality"],
            distress_level=distress_level,
        )
        patient_messages = [{"role": "system", "content": patient_system}]

        if round_n == 1:
            # First turn: patient initiates
            patient_messages.append({
                "role": "user",
                "content": "La sesión comienza. Preséntate brevemente y explica por qué has venido a terapia."
            })
        else:
            # Subsequent turns: respond to therapist
            for entry in conversation_history:
                patient_messages.append(entry)
            patient_messages.append({
                "role": "user",
                "content": "Responde al terapeuta de forma natural."
            })

        patient_text = llm_client.complete(patient_messages)

        # Generate therapist response (correct)
        therapist_system = THERAPIST_SYSTEM_PROMPT.format(
            phase=phase,
            profile_summary=f"{profile['presenting_issue']} — {profile['personality']}",
        )
        therapist_messages = [{"role": "system", "content": therapist_system}]
        for entry in conversation_history:
            therapist_messages.append(entry)
        therapist_messages.append({
            "role": "user",
            "content": patient_text,
        })

        therapist_text = llm_client.complete(therapist_messages)

        turn = SimulatedTurn(
            round_number=round_n,
            patient_input=patient_text,
            therapist_response=therapist_text,
            phase=phase,
        )

        # Generate MC options if requested
        if generate_mc and round_n > 1:
            options, correct_pos = _generate_mc_options(
                therapist_text, patient_text, phase, profile,
                conversation_history, llm_client, rng,
            )
            turn.options = options
            turn.correct_option = correct_pos

        session.turns.append(turn)

        # Update conversation history
        conversation_history.append({"role": "assistant", "content": patient_text})
        if round_n < n_rounds:
            conversation_history.append({"role": "user", "content": therapist_text})

    logger.info(
        "Generated session %s: %d rounds, profile=%s",
        session.session_id, len(session.turns), profile["id"],
    )
    return session


def _assign_phases(n_rounds: int) -> list[str]:
    """Assign ACT therapeutic phases to rounds."""
    if n_rounds <= len(ACT_PHASES):
        return ACT_PHASES[:n_rounds]

    # Spread phases across rounds, repeating middle phases
    phases = []
    phase_budget = {
        "engagement": 1,
        "creative_hopelessness": 1,
        "acceptance": max(1, n_rounds // 8),
        "defusion": max(1, n_rounds // 8),
        "present_moment": max(1, n_rounds // 8),
        "self_as_context": max(1, n_rounds // 10),
        "values": max(1, n_rounds // 8),
        "committed_action": max(1, n_rounds // 8),
        "integration": max(1, n_rounds // 10),
        "closing": 1,
    }

    for phase in ACT_PHASES:
        count = phase_budget.get(phase, 1)
        phases.extend([phase] * count)

    # Pad or trim to n_rounds
    while len(phases) < n_rounds:
        phases.insert(-1, "integration")
    return phases[:n_rounds]


def _distress_descriptor(phq9: int, gad7: int) -> str:
    """Generate a Spanish distress level description."""
    if phq9 >= 15 or gad7 >= 15:
        return "Alto: malestar significativo, dificultad para funcionar en el día a día"
    elif phq9 >= 10 or gad7 >= 10:
        return "Moderado: malestar notable que interfiere con algunas actividades"
    elif phq9 >= 5 or gad7 >= 5:
        return "Leve: malestar presente pero funcional en general"
    else:
        return "Mínimo: algunas preocupaciones pero buen funcionamiento general"


def _generate_mc_options(
    correct_response: str,
    patient_text: str,
    phase: str,
    profile: dict,
    history: list[dict],
    llm_client,
    rng: random.Random,
) -> tuple[dict[str, str], int]:
    """Generate 3 MC options (1 correct + 2 distractors)."""
    error_types = ["premature_advice", "phase_mismatch", "question_overload", "surface_validation"]
    chosen_errors = rng.sample(error_types, 2)

    distractors = []
    for error_type in chosen_errors:
        dist_system = DISTRACTOR_SYSTEM_PROMPT.format(
            error_type=error_type,
            phase=phase,
        )
        dist_messages = [{"role": "system", "content": dist_system}]

        # Include brief context
        context_str = ""
        for entry in history[-4:]:
            role_label = "Paciente" if entry["role"] == "assistant" else "Terapeuta"
            context_str += f"{role_label}: {entry['content'][:200]}\n\n"

        dist_messages.append({
            "role": "user",
            "content": f"Contexto reciente:\n{context_str}\nMensaje actual del paciente:\n{patient_text}\n\nGenera la respuesta del terapeuta con el error tipo '{error_type}'.",
        })

        dist_text = llm_client.complete(dist_messages)
        distractors.append(dist_text)

    # Shuffle positions
    options_list = [correct_response] + distractors
    order = list(range(3))
    rng.shuffle(order)
    shuffled = [options_list[i] for i in order]
    correct_pos = order.index(0) + 1

    return {
        "option_1": shuffled[0],
        "option_2": shuffled[1],
        "option_3": shuffled[2],
    }, correct_pos


def save_session_task1(session: SimulatedSession, output_dir: str | Path) -> None:
    """Save simulated session in MentalRiskES Task 1 format (round files)."""
    output_dir = Path(output_dir) / session.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for turn in session.turns:
        round_data: dict = {
            "round": turn.round_number,
            "patient_input": turn.patient_input,
        }
        if turn.round_number > 1:
            # Include previous therapist response
            prev_turn = session.turns[turn.round_number - 2]
            round_data["therapist_response"] = prev_turn.therapist_response

        # Wrap in session key (MentalRiskES server format)
        wrapped = {session.session_id: round_data}

        path = output_dir / f"round_{turn.round_number}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, ensure_ascii=False, indent=2)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session.session_id,
            "profile": session.profile,
            "target_scores": session.target_scores,
            "n_rounds": len(session.turns),
        }, f, ensure_ascii=False, indent=2)

    logger.info("Saved Task 1 data for %s (%d rounds) to %s",
                session.session_id, len(session.turns), output_dir)


def save_session_task2(session: SimulatedSession, output_dir: str | Path) -> None:
    """Save simulated session in MentalRiskES Task 2 format (round files with options)."""
    output_dir = Path(output_dir) / session.session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for turn in session.turns:
        if not turn.options:
            continue

        round_data = {
            "round": turn.round_number,
            "patient_input": turn.patient_input,
            **turn.options,
        }

        # Wrap in trial format
        wrapped = {"trial": round_data}

        path = output_dir / f"round_{turn.round_number}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, ensure_ascii=False, indent=2)

    # Save labels
    labels = {
        turn.round_number: turn.correct_option
        for turn in session.turns
        if turn.correct_option > 0
    }
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session.session_id,
            "profile": session.profile,
            "target_scores": session.target_scores,
            "n_rounds": len(session.turns),
            "rounds_with_mc": len(labels),
        }, f, ensure_ascii=False, indent=2)

    logger.info("Saved Task 2 data for %s (%d MC rounds) to %s",
                session.session_id, len(labels), output_dir)
