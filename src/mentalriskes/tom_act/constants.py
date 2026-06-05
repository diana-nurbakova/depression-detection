"""Single source of truth for label sets, instrument definitions, and mappings.

Mirrors spec Appendix B (label constants) and §2.5 (instrument wordings,
response scales, CompACT-10 Triflex subscale grouping + reverse-scoring).

Spanish labels are canonical (emitted in JSON output and used downstream);
English forms are used in English-language paper text and code comments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ToM-stance label sets (spec §5.5, Appendix B)
# ---------------------------------------------------------------------------

# Spanish labels are canonical; emitted in JSON output and used downstream.
TOM_STANCE_LABELS_ES = [
    "reflejo",
    "reformulación",
    "invitación-a-tomar-perspectiva",
    "defusión",
]

# English forms used in English-language paper text and code comments.
TOM_STANCE_EN_FROM_ES = {
    "reflejo": "reflecting",
    "reformulación": "reframing",
    "invitación-a-tomar-perspectiva": "perspective-taking-invitation",
    "defusión": "defusing",
}
TOM_STANCE_ES_FROM_EN = {v: k for k, v in TOM_STANCE_EN_FROM_ES.items()}

# Fuzzy-matcher alias set: accept English forms as low-priority fallbacks
# during Stage 3 recovery, mapped to Spanish canonical forms.
TOM_STANCE_FALLBACK_ALIASES = TOM_STANCE_ES_FROM_EN

# ToM-tier and presencia closed sets.
TOM_TIER_LABELS_ES = ["somatico", "cognitivo", "afectivo"]
PRESENCIA_LABELS_ES = ["alta", "media", "baja"]

# ---------------------------------------------------------------------------
# Sessions (spec §2.1)
# ---------------------------------------------------------------------------

# 10 sessions present in the released test data.
SESSIONS = ["S01", "S03", "S04", "S05", "S06", "S07", "S09", "S12", "S15", "S16"]
# Sessions in gold_label.json but absent from the released test data — excluded.
EXCLUDED_SESSIONS = ["S02", "S08", "S10", "S11", "S13", "S14", "S17"]

# ---------------------------------------------------------------------------
# Instruments (spec §2.5)
# ---------------------------------------------------------------------------

# n_items, integer score range [0, max_val], scale range for W1 weighting.
INSTRUMENTS = {
    "PHQ-9": {"n_items": 9, "max_val": 3, "scale_range": 27},
    "GAD-7": {"n_items": 7, "max_val": 3, "scale_range": 21},
    "CompACT-10": {"n_items": 10, "max_val": 6, "scale_range": 60},
}

# Likert anchor labels (Spanish), index == score.
PHQ9_ANCHORS = ["Para nada", "Varios días", "Más de la mitad de los días", "Casi todos los días"]
GAD7_ANCHORS = ["Nunca", "Varios días", "Más de la mitad de los días", "Casi todos los días"]
COMPACT10_ANCHORS = [
    "Totalmente en desacuerdo",
    "Bastante en desacuerdo",
    "Algo en desacuerdo",
    "Ni de acuerdo ni en desacuerdo",
    "Algo de acuerdo",
    "Bastante de acuerdo",
    "Totalmente de acuerdo",
]
INSTRUMENT_ANCHORS = {
    "PHQ-9": PHQ9_ANCHORS,
    "GAD-7": GAD7_ANCHORS,
    "CompACT-10": COMPACT10_ANCHORS,
}

# Instrument item text (Spanish), 1-indexed in the list comments.
PHQ9_ITEMS = [
    "Poco interés o placer en hacer las cosas",
    "Se ha sentido decaído(a), deprimido(a), o sin esperanzas",
    "Dificultad para dormir o permanecer dormido(a), o ha dormido demasiado",
    "Se ha sentido cansado(a) o con poca energía",
    "Con poco apetito o ha comido en exceso",
    "Se ha sentido mal con usted mismo(a) – o que es un fracaso o que ha quedado mal con usted mismo(a) o con su familia",
    "Ha tenido dificultad para concentrarse en cosas tales como leer el periódico o ver televisión",
    "Se ha estado moviendo o hablando tan lento que otras personas podrían notarlo, o por el contrario – ha estado tan inquieto(a) o agitado(a), que se ha estado moviendo mucho más de lo normal",
    "Ha pensado que estaría mejor muerto(a) o se le ha ocurrido lastimarse de alguna manera",
]
GAD7_ITEMS = [
    "Sentirse nervioso/a, intranquilo/a o con los nervios de punta",
    "No poder dejar de preocuparse o no poder controlar la preocupación",
    "Preocuparse demasiado por diferentes cosas",
    "Dificultad para relajarse",
    "Estar tan inquieto/a que es difícil permanecer sentado/a tranquilamente",
    "Molestarse o ponerse irritable fácilmente",
    "Sentir miedo como si algo terrible pudiera pasar",
]
COMPACT10_ITEMS = [
    "Hago apresuradamente actividades significativas para mí, sin prestarles realmente atención.",
    "Actúo de forma coherente con cómo deseo vivir mi vida.",
    "Me digo a mí mismo/a que no debería tener ciertos pensamientos.",
    "Me comporto de acuerdo con mis valores personales.",
    "Me esfuerzo mucho por evitar situaciones que puedan traerme pensamientos, sentimientos o sensaciones difíciles.",
    "Incluso cuando hago las cosas que me importan, me encuentro haciéndolas sin prestar atención.",
    "Acometo las cosas que son significativas para mí, incluso cuando me resulta difícil hacerlo.",
    "Me esfuerzo mucho por mantener alejados los sentimientos molestos.",
    "Parece que voy 'en piloto automático' sin ser muy consciente de lo que estoy haciendo.",
    "Puedo seguir adelante con algo cuando es importante para mí.",
]
INSTRUMENT_ITEMS = {
    "PHQ-9": PHQ9_ITEMS,
    "GAD-7": GAD7_ITEMS,
    "CompACT-10": COMPACT10_ITEMS,
}

# ---------------------------------------------------------------------------
# CompACT-10 Triflex subscales (spec §2.5)
# ---------------------------------------------------------------------------
# 1-indexed item numbers. OE and BA items are all reverse-scored; VA is not.
COMPACT10_SUBSCALES = {
    "OE": [3, 5, 8],        # Openness to Experience  (reverse)
    "BA": [1, 6, 9],        # Behavioural Awareness   (reverse)
    "VA": [2, 4, 7, 10],    # Valued Action           (direct)
}
COMPACT10_REVERSE_ITEMS = {1, 3, 5, 6, 8, 9}  # all OE + BA items
COMPACT10_MAX = 6


def reverse_score(raw: int) -> int:
    """CompACT-10 reverse-scoring formula (spec §2.5): reversed = 6 - raw."""
    return COMPACT10_MAX - raw


def compact10_subscale_scores(item_scores: list[int]) -> dict[str, float]:
    """Compute OE / BA / VA subscale sums from a 10-element raw item array.

    Raw arrays (e.g. gold ``compact_gold_items`` or Llama-inferred items) are in
    original form; reverse-scoring is applied here for OE and BA items.
    """
    if len(item_scores) != 10:
        raise ValueError(f"CompACT-10 expects 10 items, got {len(item_scores)}")
    out: dict[str, float] = {}
    for subscale, items_1idx in COMPACT10_SUBSCALES.items():
        total = 0.0
        for item_1idx in items_1idx:
            raw = item_scores[item_1idx - 1]
            total += reverse_score(raw) if item_1idx in COMPACT10_REVERSE_ITEMS else raw
        out[subscale] = total
    return out


# ---------------------------------------------------------------------------
# ACT hexaflex process keys (spec §2.4 / §5.8) — match task2 ACTProcesses
# ---------------------------------------------------------------------------
ACT_PROCESS_KEYS = [
    "defusion",
    "aceptacion",
    "momento_presente",
    "valores",
    "accion_comprometida",
    "yo_como_contexto",
]

# ---------------------------------------------------------------------------
# Therapeutic phase canonicalisation (spec §5.8 ``fase_terapeutica``)
# ---------------------------------------------------------------------------
# Llama emits the same phase in both accented and unaccented Spanish forms
# (e.g. ``defusion`` and ``defusión``), inflating RQ4's phase-conditional terms.
# The canonical set keeps Spanish accents; the lookup collapses unaccented
# variants and casing back to canonical form.

THERAPEUTIC_PHASES_ES = [
    "crisis",
    "exploración",
    "defusión",
    "aceptación",
    "activación",
    "integración",
    "cierre",
]


def _strip_accents_lower(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower().strip()


_PHASE_LOOKUP = {_strip_accents_lower(p): p for p in THERAPEUTIC_PHASES_ES}


def canonical_phase(value):
    """Normalise a ``fase_terapeutica`` string to its canonical accented form.

    - Accented and unaccented variants of a known phase collapse to the
      accented canonical (e.g. ``"defusion"`` → ``"defusión"``).
    - Casing is normalised (lowercased).
    - Unknown phases fall back to the accent-stripped lowercase form so
      future Llama outputs remain comparable across rows.
    - ``None`` / empty / non-string inputs pass through unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    key = _strip_accents_lower(value)
    return _PHASE_LOOKUP.get(key, key)

# ---------------------------------------------------------------------------
# Signal types (spec §5.2 / §6.1)
# ---------------------------------------------------------------------------
LLAMA_SIGNAL = "llama_state_snapshot"
GEMMA_VIEW_SIGNALS = ["self_a", "self_b", "observer_p", "observer_pt"]
GEMMA_TIER_SIGNAL = "tom_tier_patient"
GEMMA_STANCE_SIGNAL = "tom_stance"
GEMMA_PRESENCIA_SIGNAL = "presencia"
GEMMA_SIGNALS = GEMMA_VIEW_SIGNALS + [GEMMA_TIER_SIGNAL, GEMMA_STANCE_SIGNAL, GEMMA_PRESENCIA_SIGNAL]
ALL_SIGNALS = [LLAMA_SIGNAL] + GEMMA_SIGNALS

# Candidate options for per-candidate signals (stance, presencia).
CANDIDATE_OPTIONS = [1, 2, 3]
