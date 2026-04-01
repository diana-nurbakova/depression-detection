"""Prompt templates for Task 2 state tracking and response selection.

All prompts are in Spanish (primary) with English variants available.
Organized by pipeline step and evaluation framing variant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1: State Update
# ---------------------------------------------------------------------------

STATE_UPDATE_SYSTEM = {
    "es": """\
Eres un sistema de seguimiento clínico para conversaciones terapéuticas basadas en \
Terapia de Aceptación y Compromiso (ACT). Tu tarea es actualizar el estado de la \
conversación tras cada turno del paciente.

Debes rastrear:
1. FASE TERAPÉUTICA: crisis / exploración / aceptación / defusión / activación / \
integración / cierre.
2. ESTADO EMOCIONAL: valencia (negativa/mixta/neutra/positiva), intensidad \
(alta/media/baja), orientación a la acción (evitativa/pasiva/tentativa/activa).
3. PROCESOS ACT activos (0.0–1.0 cada uno): defusión, aceptación, momento_presente, \
valores, acción_comprometida, yo_como_contexto.
4. METÁFORAS ACTIVAS: Metáforas que el paciente ha adoptado.
5. MARCADORES DE RAPPORT: Indicadores de alianza terapéutica.

Responde SOLO con un objeto JSON válido.""",

    "en": """\
You are a clinical tracking system for ACT-based therapeutic conversations. Your task \
is to update the conversation state after each patient turn.

Track:
1. THERAPEUTIC PHASE: crisis / exploration / acceptance / defusion / activation / \
integration / closing.
2. EMOTIONAL STATE: valence (negative/mixed/neutral/positive), intensity \
(high/medium/low), action orientation (avoidant/passive/tentative/active).
3. ACT PROCESSES (0.0–1.0 each): defusion, acceptance, present_moment, values, \
committed_action, self_as_context.
4. ACTIVE METAPHORS: Metaphors the patient has adopted.
5. RAPPORT MARKERS: Therapeutic alliance indicators.

Respond ONLY with a valid JSON object.""",
}


def build_state_update_user(
    previous_state_json: str,
    selected_response_text: str | None,
    patient_input: str,
    round_number: int,
    lang: str = "es",
) -> str:
    """Build the user prompt for Step 1 state update."""
    if lang == "es":
        prev_label = "ESTADO ANTERIOR"
        resp_label = "RESPUESTA DEL TERAPEUTA SELECCIONADA EN EL TURNO ANTERIOR"
        msg_label = f"NUEVO MENSAJE DEL PACIENTE (Turno {round_number})"
        instruction = "Actualiza el estado. Responde con JSON:"
        no_prev = "Primera interacción — no hay estado previo."
        no_resp = "Primer turno — no hay respuesta previa del terapeuta."
    else:
        prev_label = "PREVIOUS STATE"
        resp_label = "THERAPIST RESPONSE SELECTED IN PREVIOUS TURN"
        msg_label = f"NEW PATIENT MESSAGE (Round {round_number})"
        instruction = "Update the state. Respond with JSON:"
        no_prev = "First interaction — no previous state."
        no_resp = "First turn — no previous therapist response."

    prev_text = previous_state_json if previous_state_json else no_prev
    resp_text = selected_response_text if selected_response_text else no_resp

    return f"""{prev_label}:
{prev_text}

{resp_label}:
{resp_text}

{msg_label}:
{patient_input}

{instruction}
{{
  "fase_terapeutica": "...",
  "estado_emocional": {{"valencia": "...", "intensidad": "...", "orientacion_accion": "..."}},
  "procesos_act": {{"defusion": 0.0, "aceptacion": 0.0, "momento_presente": 0.0,
                    "valores": 0.0, "accion_comprometida": 0.0, "yo_como_contexto": 0.0}},
  "metaforas_activas": [],
  "marcadores_rapport": [],
  "resumen_acumulado": "Resumen breve en 2–3 frases."
}}"""


# ---------------------------------------------------------------------------
# Step 2: Evaluation + Selection — FUNC variant
# ---------------------------------------------------------------------------

FUNC_SYSTEM = {
    "es": """\
Eres un sistema de apoyo a la decisión terapéutica basado en ACT. Debes seleccionar \
la respuesta terapéutica más apropiada entre tres opciones en español.

## PRINCIPIO CLAVE: Análisis funcional, no patrones superficiales

Las tres opciones pueden parecer similares en la superficie. La diferencia está en la \
FUNCIÓN del comportamiento del terapeuta, no en las palabras exactas.

Pregúntate: ¿Cuál es la FUNCIÓN de esta respuesta en el contexto terapéutico?
- ¿Valida para CREAR ESPACIO? → consistente con ACT
- ¿Valida para PASAR DE LARGO el malestar? → inconsistente
- ¿Hace preguntas para que el paciente EXPLORE? → consistente
- ¿Hace preguntas para DIRIGIR al paciente? → inconsistente

## EJEMPLOS DE ANÁLISIS FUNCIONAL

Ejemplo 1 — Defusión vs. Exploración prematura de valores (basado en Turno 5):
Paciente: "Es como... una bola en el estómago. Pesada, grisácea..."
Opción A (defusión_experiencial): "¿Qué sucede si simplemente la dejas pasar, como \
  si fuera un coche que pasa por la calle?" → FUNCIÓN: facilita defusión experiencial.
Opción B (activación_prematura): "¿Qué valores son importantes para ti? ¿Qué tipo \
  de persona quieres ser?" → FUNCIÓN: salta a valores antes de trabajar aceptación.
Ambas son "ACT" en contenido. La diferencia es funcional: la fase requiere defusión.

Ejemplo 2 — Validación con diferentes funciones (basado en Turno 8):
Opción A (permanencia_con_dificultad): "Ese respiro es indicador de que estás creando \
  espacio. No tienes que eliminar la tensión, solo coexistir con ella." → Valida para \
  REFORZAR la aceptación.
Opción B (reaseguramiento_prematuro + activación_prematura): "Es un gesto de valentía. \
  ¿Qué te gustaría hacer ahora?" → Valida pero REDIRIGE a acción prematuramente.

Ejemplo 3 — Espacio vs. Sobrecarga (basado en Turno 16):
Paciente: "Me gustaría cerrar los ojos y simplemente observar. Gracias."
Opción A (momento_presente_atento): "Cierra los ojos, observa lo que surge. No intentes \
  cambiar nada. Estoy aquí contigo." → Breve, da espacio, acompaña.
Opción B (sobrecarga_preguntas + conceptual_excesivo): "Imagina un lugar tranquilo... \
  ¿Qué te gustaría hacer ahora? No tienes que hacerlo solo..." → Demasiado largo, \
  demasiadas preguntas cuando el paciente ha pedido silencio.

## PROCEDIMIENTO

### PASO 1 — CARACTERIZACIÓN
Asigna a cada opción etiquetas terapéuticas del vocabulario controlado.
Evalúa la FUNCIÓN, no la forma.

Etiquetas de consistencia: validación_empática, defusión_experiencial, \
  aceptación_compasiva, momento_presente_atento, exploración_valores, \
  acción_comprometida_gradual, yo_contexto_observador, normalización_experiencia, \
  permanencia_con_dificultad.

Etiquetas de inconsistencia: consejo_directivo, reaseguramiento_prematuro, \
  activación_prematura, sobrecarga_preguntas, conceptual_excesivo, \
  positivismo_forzado, control_emocional, mindfulness_como_control, \
  imposición_valores.

### PASO 2 — ELIMINACIÓN
Para opciones con etiquetas de inconsistencia: ¿La inconsistencia es la FUNCIÓN \
PRINCIPAL o un elemento menor? Si es la función principal → ELIMINADA.

### PASO 3 — EVALUACIÓN DE ADECUACIÓN
Para opciones no eliminadas:
- fase_terapéutica: ¿Apropiada para la fase actual?
- coherencia_metáforas: ¿Construye sobre metáforas adoptadas?
- proporcionalidad: ¿Extensión proporcionada al estado emocional?
- carga_de_preguntas: 0 (solo en cierre), 1–2 (óptimo), 3+ (excesivo).
- validación_primero: ¿Valida antes de explorar o proponer?

### PASO 4 — SELECCIÓN
Elige la opción con mayor consistencia + mejor adecuación a la fase.
Si empate: prefiere coherencia con metáforas y proporcionalidad.
Si todas eliminadas: selecciona la menos inconsistente.

## FORMATO DE RESPUESTA
Responde SOLO con un JSON válido con la siguiente estructura:
{
  "caracterización": {
    "opcion_1": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."},
    "opcion_2": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."},
    "opcion_3": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."}
  },
  "eliminación": {
    "opcion_1": {"eliminada": false, "debilidades": []},
    "opcion_2": {"eliminada": false, "debilidades": []},
    "opcion_3": {"eliminada": false, "debilidades": []}
  },
  "evaluación_adecuación": {
    "opcion_N": {
      "fase_apropiada": true,
      "coherencia_metáforas": "construye/neutra/compite",
      "proporcionalidad": "adecuada/excesiva/insuficiente",
      "num_preguntas": 0,
      "valida_primero": true
    }
  },
  "selección": {
    "opcion_elegida": {
      "numero": 1,
      "etiqueta_principal": "...",
      "fase_match": "...",
      "razón_principal": "..."
    },
    "razonamiento": "Justificación funcional en 2–3 frases."
  }
}""",

    "en": """\
You are an ACT-based therapeutic decision support system. Select the most appropriate \
therapist response from three options.

## KEY PRINCIPLE: Functional analysis, not surface patterns

The three options may look similar on the surface. The difference is in the FUNCTION \
of the therapist's behavior, not the exact words.

Ask yourself: What is the FUNCTION of this response in the therapeutic context?
- Validates to CREATE SPACE? → ACT-consistent
- Validates to BYPASS distress? → inconsistent
- Asks questions so the patient EXPLORES? → consistent
- Asks questions to DIRECT the patient? → inconsistent

## PROCEDURE

### STEP 1 — CHARACTERIZATION
Assign therapeutic tags from the controlled vocabulary.
Evaluate FUNCTION, not form.

Consistency tags: empathic_validation, experiential_defusion, \
  compassionate_acceptance, mindful_present_moment, values_exploration, \
  gradual_committed_action, observer_self, experience_normalization, \
  staying_with_difficulty.

Inconsistency tags: directive_advice, premature_reassurance, \
  premature_activation, question_overload, excessively_conceptual, \
  forced_positivity, emotional_control, mindfulness_as_control, \
  values_imposition.

### STEP 2 — ELIMINATION
For options with inconsistency tags: Is inconsistency the PRIMARY FUNCTION \
or a minor element? If primary → ELIMINATED.

### STEP 3 — FIT EVALUATION
For non-eliminated options:
- therapeutic_phase: Appropriate for current phase?
- metaphor_coherence: Builds on adopted metaphors?
- proportionality: Length proportional to emotional state?
- question_load: 0 (closing only), 1–2 (optimal), 3+ (excessive).
- validation_first: Validates before exploring or proposing?

### STEP 4 — SELECTION
Choose the option with highest consistency + best phase fit.
Ties: prefer metaphor coherence and proportionality.
All eliminated: select the least inconsistent.

## RESPONSE FORMAT
Respond ONLY with valid JSON:
{
  "characterization": {
    "option_1": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."},
    "option_2": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."},
    "option_3": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."}
  },
  "elimination": {
    "option_1": {"eliminated": false, "weaknesses": []},
    "option_2": {"eliminated": false, "weaknesses": []},
    "option_3": {"eliminated": false, "weaknesses": []}
  },
  "fit_evaluation": {
    "option_N": {
      "phase_appropriate": true,
      "metaphor_coherence": "builds/neutral/competes",
      "proportionality": "adequate/excessive/insufficient",
      "num_questions": 0,
      "validation_first": true
    }
  },
  "selection": {
    "chosen_option": {
      "number": 1,
      "primary_tag": "...",
      "phase_match": "...",
      "primary_reason": "..."
    },
    "reasoning": "Functional justification in 2–3 sentences."
  }
}""",
}


# ---------------------------------------------------------------------------
# ToM evaluation module (shared by HYB, TOM-B, TOM-C)
# ---------------------------------------------------------------------------

TOM_EVAL_MODULE = {
    "es": """\
## Teoría de la Mente Terapéutica

Para evaluar cada opción, aplica razonamiento de Teoría de la Mente en dos niveles:

NIVEL 1 — MODELO MENTAL DEL TERAPEUTA SOBRE EL PACIENTE
  ¿Qué CREE el terapeuta sobre el estado actual del paciente?
  ¿Qué cree que el paciente necesita ahora mismo?
  ¿Es este modelo mental PRECISO dado lo que sabemos?

NIVEL 2 — PREDICCIÓN DEL EFECTO EN EL PACIENTE
  Si el paciente recibe esta respuesta:
  - ¿Se sentirá escuchado y validado, o dirigido y presionado?
  - ¿Tendrá espacio para explorar, o se sentirá abrumado?
  - ¿Se abrirá más, o se cerrará?
  - ¿La respuesta profundizará el proceso ACT actual, o lo interrumpirá?
  - ¿Las metáforas resonarán con su experiencia?

  Un buen terapeuta ACT PREDICE que su intervención:
  - Creará espacio para contactar con la experiencia (no evitarla).
  - Mantendrá o fortalecerá la alianza terapéutica.
  - Será coherente con lo que al paciente le funciona.
  - Avanzará al ritmo del paciente, no del terapeuta.""",

    "en": """\
## Therapeutic Theory of Mind

Evaluate each option using two-level Theory of Mind reasoning:

LEVEL 1 — THERAPIST'S MENTAL MODEL OF THE PATIENT
  What does the therapist BELIEVE about the patient's current state?
  What does the therapist think the patient needs right now?
  Is this mental model ACCURATE given what we know?

LEVEL 2 — PREDICTED EFFECT ON THE PATIENT
  If the patient receives this response:
  - Will they feel heard and validated, or directed and pressured?
  - Will they have space to explore, or feel overwhelmed?
  - Will they open up more, or shut down?
  - Will the response deepen the current ACT process, or interrupt it?
  - Will the metaphors resonate with their experience?

  A good ACT therapist PREDICTS their intervention will:
  - Create space to contact experience (not avoid it).
  - Maintain or strengthen the therapeutic alliance.
  - Be coherent with what works for the patient.
  - Advance at the patient's pace, not the therapist's.""",
}


# ---------------------------------------------------------------------------
# ToM elimination module (used in TOM-B and TOM-C)
# ---------------------------------------------------------------------------

TOM_ELIM_MODULE = {
    "es": """\
### ELIMINACIÓN POR MODELO MENTAL ERRÓNEO

Para cada opción, evalúa si el terapeuta opera desde un modelo mental \
FUNDAMENTALMENTE INCORRECTO del paciente.

Un modelo mental erróneo se manifiesta cuando el terapeuta asume implícitamente:

  modelo_urgencia: El paciente necesita sentirse mejor AHORA MISMO, en lugar de \
    aprender a estar con la experiencia difícil.
    Señal: la respuesta se orienta a reducir el malestar como prioridad.

  modelo_instrucción: El paciente necesita que le DIGAN qué hacer o pensar, en \
    lugar de descubrir por sí mismo.
    Señal: la respuesta da consejos, instrucciones o soluciones.

  modelo_ritmo_terapeuta: El paciente debe avanzar al RITMO DEL TERAPEUTA, no \
    al suyo propio.
    Señal: la respuesta empuja hacia una técnica o fase para la que el paciente \
    no ha mostrado preparación.

  modelo_emoción_problema: Las emociones difíciles SON EL PROBLEMA a resolver, \
    en lugar del contexto para el crecimiento.
    Señal: la respuesta busca eliminar, reducir o reemplazar la emoción.

  modelo_fragilidad: El paciente es FRÁGIL y necesita protección, en lugar de \
    capaz de observar y estar con su experiencia.
    Señal: la respuesta evita profundizar, desvía la atención, o reasegura \
    prematuramente.

Para cada opción:
- ¿Opera desde alguno de estos modelos mentales erróneos?
- Si el modelo erróneo es la BASE de la respuesta → ELIMINADA.
- Si es un elemento menor → anotar como debilidad, no eliminar.""",

    "en": """\
### ELIMINATION BY FLAWED MENTAL MODEL

For each option, evaluate whether the therapist operates from a FUNDAMENTALLY \
INCORRECT mental model of the patient.

A flawed mental model manifests when the therapist implicitly assumes:

  urgency_model: The patient needs to feel better RIGHT NOW, instead of learning \
    to sit with difficult experience.
    Signal: the response prioritizes distress reduction.

  instruction_model: The patient needs to be TOLD what to do or think, instead \
    of discovering on their own.
    Signal: the response gives advice, instructions, or solutions.

  therapist_pace_model: The patient should advance at the THERAPIST'S pace, not \
    their own.
    Signal: the response pushes toward a technique or phase the patient hasn't \
    shown readiness for.

  emotion_as_problem_model: Difficult emotions ARE THE PROBLEM to solve, instead \
    of the context for growth.
    Signal: the response seeks to eliminate, reduce, or replace the emotion.

  fragility_model: The patient is FRAGILE and needs protection, instead of being \
    capable of observing and sitting with their experience.
    Signal: the response avoids deepening, diverts attention, or reassures \
    prematurely.

For each option:
- Does it operate from any of these flawed mental models?
- If the flawed model is the BASE of the response → ELIMINATED.
- If it's a minor element → note as weakness, don't eliminate.""",
}


# ---------------------------------------------------------------------------
# Structured ACT-FM selection module (used in TOM-B)
# ---------------------------------------------------------------------------

TOM_B_SELECTION_MODULE = {
    "es": """\
### SELECCIÓN ESTRUCTURADA (basada en ACT-FM, informada por análisis ToM)

Usando los resultados del análisis ToM, aplica los siguientes criterios:

CRITERIO 1 — Precisión del modelo mental (peso: alto)
  ¿La opción con el modelo mental más preciso se alinea con las etiquetas de \
  consistencia ACT más relevantes para la fase actual?

  Mapeo fase → etiquetas prioritarias:
  - crisis → validación_empática, normalización_experiencia
  - exploración → métodos_experienciales, notar_interacción_pensamientos
  - aceptación → aceptación_compasiva, permanecer_con_dolor
  - defusión → defusión_experiencial, pensamientos_separados
  - activación → acción_comprometida_gradual, respuestas_viables_inviables
  - integración → foco_momento_presente, más_grande_que_experiencias
  - cierre → planes_alineados_valores, clarificar_valores

CRITERIO 2 — Efecto predicho en el paciente (peso: alto)
  Entre las opciones con modelo mental preciso, ¿cuál produce el efecto \
  predicho más beneficioso?
  Prioridad: fortalece alianza > avanza proceso > mantiene proceso.

CRITERIO 3 — Coherencia con metáforas activas (peso: medio)
CRITERIO 4 — Proporcionalidad (peso: medio)
CRITERIO 5 — Validación primero (peso: bajo)

Selecciona la opción con mayor puntuación ponderada.""",

    "en": """\
### STRUCTURED SELECTION (ACT-FM based, informed by ToM analysis)

Using the ToM analysis results, apply the following criteria:

CRITERION 1 — Mental model accuracy (weight: high)
  Does the option with the most accurate mental model align with the ACT \
  consistency tags most relevant for the current phase?

  Phase → priority tags mapping:
  - crisis → empathic_validation, experience_normalization
  - exploration → experiential_methods, notice_thought_interaction
  - acceptance → compassionate_acceptance, staying_with_pain
  - defusion → experiential_defusion, thoughts_as_separate
  - activation → gradual_committed_action, workable_responses
  - integration → present_moment_focus, larger_than_experiences
  - closing → values_aligned_plans, clarify_values

CRITERION 2 — Predicted effect on patient (weight: high)
  Among options with accurate mental model, which produces the most \
  beneficial predicted effect?
  Priority: strengthens alliance > advances process > maintains process.

CRITERION 3 — Coherence with active metaphors (weight: medium)
CRITERION 4 — Proportionality (weight: medium)
CRITERION 5 — Validation first (weight: low)

Select the option with highest weighted score.""",
}


# ---------------------------------------------------------------------------
# Single-prompt CoT (Variant A) — combines state + evaluation in one call
# ---------------------------------------------------------------------------

SINGLE_COT_SYSTEM = {
    "es": """\
Eres un sistema de apoyo a la decisión terapéutica basado en Terapia de Aceptación \
y Compromiso (ACT). Recibes una conversación terapéutica en español y debes:

1. Analizar el ESTADO actual del paciente (fase terapéutica, emoción, procesos ACT).
2. EVALUAR tres opciones de respuesta del terapeuta usando análisis funcional.
3. SELECCIONAR la opción más apropiada.

Responde con un JSON que incluya tanto el estado como la selección.""",

    "en": """\
You are an ACT-based therapeutic decision support system. You receive a therapeutic \
conversation in Spanish and must:

1. Analyze the patient's current STATE (therapeutic phase, emotion, ACT processes).
2. EVALUATE three therapist response options using functional analysis.
3. SELECT the most appropriate option.

Respond with JSON including both state and selection.""",
}


# ---------------------------------------------------------------------------
# Step 1.5: Characterization-only (Variant B+)
# ---------------------------------------------------------------------------

CHARACTERIZATION_SYSTEM = {
    "es": """\
Eres un sistema de etiquetado terapéutico. Para cada una de las tres opciones de \
respuesta del terapeuta, asigna etiquetas terapéuticas del vocabulario controlado. \
Evalúa la FUNCIÓN de cada respuesta, no su forma superficial.

Etiquetas de consistencia: validación_empática, defusión_experiencial, \
  aceptación_compasiva, momento_presente_atento, exploración_valores, \
  acción_comprometida_gradual, yo_contexto_observador, normalización_experiencia, \
  permanencia_con_dificultad.

Etiquetas de inconsistencia: consejo_directivo, reaseguramiento_prematuro, \
  activación_prematura, sobrecarga_preguntas, conceptual_excesivo, \
  positivismo_forzado, control_emocional, mindfulness_como_control, \
  imposición_valores.

Responde SOLO con JSON:
{
  "opcion_1": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."},
  "opcion_2": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."},
  "opcion_3": {"etiquetas_consistencia": [], "etiquetas_inconsistencia": [], "función_principal": "..."}
}""",

    "en": """\
You are a therapeutic tagging system. For each of the three therapist response options, \
assign therapeutic tags from the controlled vocabulary. Evaluate the FUNCTION of each \
response, not its surface form.

Consistency tags: empathic_validation, experiential_defusion, \
  compassionate_acceptance, mindful_present_moment, values_exploration, \
  gradual_committed_action, observer_self, experience_normalization, \
  staying_with_difficulty.

Inconsistency tags: directive_advice, premature_reassurance, \
  premature_activation, question_overload, excessively_conceptual, \
  forced_positivity, emotional_control, mindfulness_as_control, \
  values_imposition.

Respond ONLY with JSON:
{
  "option_1": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."},
  "option_2": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."},
  "option_3": {"consistency_tags": [], "inconsistency_tags": [], "primary_function": "..."}
}""",
}


# ---------------------------------------------------------------------------
# Experiential tiebreaker calibration (§4.2 of v1.2 update)
# ---------------------------------------------------------------------------

EXPERIENTIAL_CALIBRATION = {
    "es": """\

## CALIBRACIÓN DE SELECCIÓN

Cuando dos opciones son similares en consistencia ACT y adecuación a la fase:

1. Prefiere la opción que usa un enfoque más EXPERIENCIAL y METAFÓRICO \
sobre la que es más directa o conceptual.
(Ejemplo: "observa cómo se siente en tu cuerpo" > "¿qué piensas sobre eso?")

2. Prefiere la opción que INVITA a la observación sobre la que DIRIGE \
la atención.
(Ejemplo: "¿qué notas?" > "fíjate en que...")

3. Prefiere la opción que avanza al RITMO DEL PACIENTE sobre la que \
introduce técnicas nuevas o empuja hacia una fase siguiente.
(Ejemplo: acompañar en la experiencia actual > proponer un ejercicio nuevo)

4. Prefiere la opción que es más BREVE y da más ESPACIO cuando el \
paciente ha mostrado señales de necesitar procesamiento interno \
(pausas, "gracias", turnos cortos).

Esta calibración refleja principios clínicos ACT: los métodos \
experienciales son preferibles a los conceptuales (ACT-FM C2), y el \
ritmo debe ser del paciente, no del terapeuta.""",

    "en": """\

## SELECTION CALIBRATION

When two options are similar in ACT consistency and phase fit:

1. Prefer the option using a more EXPERIENTIAL and METAPHORICAL approach \
over a more direct or conceptual one.
(Example: "notice how that feels in your body" > "what do you think about that?")

2. Prefer the option that INVITES observation over one that DIRECTS attention.
(Example: "what do you notice?" > "pay attention to...")

3. Prefer the option that follows the PATIENT'S PACE over one that introduces \
new techniques or pushes toward the next phase.
(Example: accompanying the current experience > proposing a new exercise)

4. Prefer the option that is more BRIEF and gives more SPACE when the patient \
has shown signs of needing internal processing (pauses, "thank you", short turns).

This calibration reflects core ACT clinical principles: experiential methods \
are preferable to conceptual ones (ACT-FM C2), and pacing should follow the \
patient, not the therapist.""",
}


def build_selection_system(
    framing: str, lang: str = "es", calibration: bool = False,
) -> str:
    """Assemble the full system prompt for Step 2 based on framing variant.

    Args:
        framing: FUNC, HYB, TOM-B, or TOM-C.
        lang: "es" or "en".
        calibration: If True, append experiential tiebreaker calibration.

    Returns:
        Full system prompt string.
    """
    system = _build_selection_system_base(framing, lang)
    if calibration:
        system += EXPERIENTIAL_CALIBRATION[lang]
    return system


def _build_selection_system_base(framing: str, lang: str = "es") -> str:
    """Assemble the base system prompt for Step 2 (without calibration).

    Variants:
      FUNC  — Full functional analysis (§5.5)
      HYB   — FUNC elimination + ToM selection
      TOM-B — ToM evaluation + ToM elimination + structured ACT-FM selection
      TOM-C — ToM evaluation + ToM elimination + ToM selection
    """
    if framing == "FUNC":
        return FUNC_SYSTEM[lang]

    elif framing == "HYB":
        base = FUNC_SYSTEM[lang]
        tom = TOM_EVAL_MODULE[lang]
        return f"""{base}

{tom}

IMPORTANTE: Usa el análisis funcional (PASOS 1-2) para la ELIMINACIÓN. \
Luego usa el análisis de Teoría de la Mente para la SELECCIÓN final entre \
las opciones supervivientes. La selección se basa en cuál opción produce \
el mejor efecto predicho en el paciente."""

    elif framing == "TOM-B":
        tom_eval = TOM_EVAL_MODULE[lang]
        tom_elim = TOM_ELIM_MODULE[lang]
        tom_sel = TOM_B_SELECTION_MODULE[lang]

        if lang == "es":
            intro = """\
Eres un sistema de apoyo a la decisión terapéutica basado en ACT. Debes seleccionar \
la respuesta terapéutica más apropiada entre tres opciones en español.

Usa Teoría de la Mente como LENTE de evaluación y criterios ACT-FM estructurados \
como MARCO de decisión."""
            fmt = """\
## FORMATO DE RESPUESTA
Responde SOLO con un JSON válido:
{
  "análisis_tom": {
    "opcion_1": {
      "modelo_mental_terapeuta": "...",
      "modelo_mental_preciso": true,
      "efecto_predicho_paciente": "...",
      "efecto_alianza": "fortalece/mantiene/debilita",
      "efecto_proceso_act": "avanza/mantiene/interrumpe",
      "etiquetas_consistencia": [],
      "etiquetas_inconsistencia": []
    },
    "opcion_2": {"...": "..."},
    "opcion_3": {"...": "..."}
  },
  "eliminación_tom": {
    "opcion_1": {"eliminada": false, "modelos_erróneos": [], "debilidades": []},
    "opcion_2": {"eliminada": false, "modelos_erróneos": [], "debilidades": []},
    "opcion_3": {"eliminada": false, "modelos_erróneos": [], "debilidades": []}
  },
  "selección_estructurada": {
    "opcion_elegida": {
      "numero": 1,
      "etiqueta_principal": "...",
      "razón_principal": "..."
    }
  }
}"""
        else:
            intro = """\
You are an ACT-based therapeutic decision support system. Select the most appropriate \
therapist response from three options.

Use Theory of Mind as the evaluation LENS and structured ACT-FM criteria as the \
DECISION FRAMEWORK."""
            fmt = """\
## RESPONSE FORMAT
Respond ONLY with valid JSON:
{
  "tom_analysis": {
    "option_1": {
      "therapist_mental_model": "...",
      "mental_model_accurate": true,
      "predicted_patient_effect": "...",
      "alliance_effect": "strengthens/maintains/weakens",
      "act_process_effect": "advances/maintains/interrupts",
      "consistency_tags": [],
      "inconsistency_tags": []
    },
    "option_2": {"...": "..."},
    "option_3": {"...": "..."}
  },
  "tom_elimination": {
    "option_1": {"eliminated": false, "flawed_models": [], "weaknesses": []},
    "option_2": {"eliminated": false, "flawed_models": [], "weaknesses": []},
    "option_3": {"eliminated": false, "flawed_models": [], "weaknesses": []}
  },
  "structured_selection": {
    "chosen_option": {
      "number": 1,
      "primary_tag": "...",
      "primary_reason": "..."
    }
  }
}"""
        return f"{intro}\n\n{tom_eval}\n\n{tom_elim}\n\n{tom_sel}\n\n{fmt}"

    elif framing == "TOM-C":
        tom_eval = TOM_EVAL_MODULE[lang]
        tom_elim = TOM_ELIM_MODULE[lang]

        if lang == "es":
            intro = """\
Eres un sistema de apoyo a la decisión terapéutica basado en ACT. Debes seleccionar \
la respuesta terapéutica más apropiada entre tres opciones en español.

Usa razonamiento de Teoría de la Mente completo: evalúa, elimina y selecciona \
basándote en la precisión del modelo mental del terapeuta y el efecto predicho \
en el paciente."""
            selection = """\
### SELECCIÓN POR TEORÍA DE LA MENTE

Entre las opciones no eliminadas, selecciona la que tiene:
1. El modelo mental más PRECISO del estado actual del paciente.
2. El efecto predicho más BENEFICIOSO para el paciente.
3. La mayor probabilidad de FORTALECER la alianza terapéutica.

Si empate en modelo mental: prefiere mejor efecto predicho.
Si empate en efecto: prefiere coherencia con metáforas activas."""
            fmt = """\
## FORMATO DE RESPUESTA
Responde SOLO con un JSON válido:
{
  "análisis_tom": {
    "opcion_1": {
      "modelo_mental_terapeuta": "...",
      "modelo_mental_preciso": true,
      "efecto_predicho_paciente": "...",
      "efecto_alianza": "fortalece/mantiene/debilita",
      "efecto_proceso_act": "avanza/mantiene/interrumpe"
    },
    "opcion_2": {"...": "..."},
    "opcion_3": {"...": "..."}
  },
  "eliminación_tom": {
    "opcion_1": {"eliminada": false, "modelos_erróneos": [], "debilidades": []},
    "opcion_2": {"eliminada": false, "modelos_erróneos": [], "debilidades": []},
    "opcion_3": {"eliminada": false, "modelos_erróneos": [], "debilidades": []}
  },
  "selección": {
    "opcion_elegida": {
      "numero": 1,
      "razón_principal": "..."
    },
    "razonamiento": "Justificación basada en ToM en 2–3 frases."
  }
}"""
        else:
            intro = """\
You are an ACT-based therapeutic decision support system. Select the most appropriate \
therapist response from three options.

Use full Theory of Mind reasoning: evaluate, eliminate, and select based on the \
accuracy of the therapist's mental model and the predicted effect on the patient."""
            selection = """\
### THEORY OF MIND SELECTION

Among non-eliminated options, select the one with:
1. The most ACCURATE mental model of the patient's current state.
2. The most BENEFICIAL predicted effect on the patient.
3. The highest probability of STRENGTHENING the therapeutic alliance.

If tied on mental model: prefer better predicted effect.
If tied on effect: prefer coherence with active metaphors."""
            fmt = """\
## RESPONSE FORMAT
Respond ONLY with valid JSON:
{
  "tom_analysis": {
    "option_1": {
      "therapist_mental_model": "...",
      "mental_model_accurate": true,
      "predicted_patient_effect": "...",
      "alliance_effect": "strengthens/maintains/weakens",
      "act_process_effect": "advances/maintains/interrupts"
    },
    "option_2": {"...": "..."},
    "option_3": {"...": "..."}
  },
  "tom_elimination": {
    "option_1": {"eliminated": false, "flawed_models": [], "weaknesses": []},
    "option_2": {"eliminated": false, "flawed_models": [], "weaknesses": []},
    "option_3": {"eliminated": false, "flawed_models": [], "weaknesses": []}
  },
  "selection": {
    "chosen_option": {
      "number": 1,
      "primary_reason": "..."
    },
    "reasoning": "ToM-based justification in 2–3 sentences."
  }
}"""
        return f"{intro}\n\n{tom_eval}\n\n{tom_elim}\n\n{selection}\n\n{fmt}"

    else:
        raise ValueError(f"Unknown framing variant: {framing}")


# Remove old build_selection_system (replaced above)


def build_selection_user(
    state_json: str,
    recent_transcript: str,
    patient_input: str,
    options: dict[str, str],
    selection_log: str,
    round_number: int,
    lang: str = "es",
    characterization_tags: str | None = None,
) -> str:
    """Build the user prompt for Step 2 (or Step 2 in B+ after characterization)."""
    if lang == "es":
        parts = [
            f"ESTADO ACTUAL DE LA CONVERSACIÓN:\n{state_json}",
            f"HISTORIAL RECIENTE:\n{recent_transcript}",
            f"SELECCIONES PREVIAS:\n{selection_log}",
            f"MENSAJE DEL PACIENTE (Turno {round_number}):\n{patient_input}",
            f"OPCIÓN 1:\n{options['option_1']}",
            f"OPCIÓN 2:\n{options['option_2']}",
            f"OPCIÓN 3:\n{options['option_3']}",
        ]
        if characterization_tags:
            parts.append(f"CARACTERIZACIÓN PREVIA DE OPCIONES:\n{characterization_tags}")
        parts.append("Evalúa las tres opciones y selecciona la más apropiada. Responde con JSON.")
    else:
        parts = [
            f"CURRENT CONVERSATION STATE:\n{state_json}",
            f"RECENT HISTORY:\n{recent_transcript}",
            f"PREVIOUS SELECTIONS:\n{selection_log}",
            f"PATIENT MESSAGE (Round {round_number}):\n{patient_input}",
            f"OPTION 1:\n{options['option_1']}",
            f"OPTION 2:\n{options['option_2']}",
            f"OPTION 3:\n{options['option_3']}",
        ]
        if characterization_tags:
            parts.append(f"PRIOR OPTION CHARACTERIZATION:\n{characterization_tags}")
        parts.append("Evaluate the three options and select the most appropriate. Respond with JSON.")

    return "\n\n".join(parts)
