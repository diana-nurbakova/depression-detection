"""Data models for Task 2 state tracking and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmotionalState:
    valencia: str = "negativa"  # negativa | mixta | neutra | positiva
    intensidad: str = "alta"  # alta | media | baja
    orientacion_accion: str = "pasiva"  # evitativa | pasiva | tentativa | activa


@dataclass
class ACTProcesses:
    defusion: float = 0.0
    aceptacion: float = 0.0
    momento_presente: float = 0.0
    valores: float = 0.0
    accion_comprometida: float = 0.0
    yo_como_contexto: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "defusion": self.defusion,
            "aceptacion": self.aceptacion,
            "momento_presente": self.momento_presente,
            "valores": self.valores,
            "accion_comprometida": self.accion_comprometida,
            "yo_como_contexto": self.yo_como_contexto,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ACTProcesses:
        return cls(**{k: float(v) for k, v in d.items() if hasattr(cls, k)})


@dataclass
class RoundRecord:
    round_id: int
    patient_message: str
    options: dict[str, str]  # {"option_1": ..., "option_2": ..., "option_3": ...}
    selected_option: int | None = None
    selected_response_text: str | None = None


@dataclass
class SharedState:
    """Accumulated conversation state across rounds."""

    transcript: list[RoundRecord] = field(default_factory=list)
    fase_terapeutica: str = "crisis"
    estado_emocional: EmotionalState = field(default_factory=EmotionalState)
    procesos_act: ACTProcesses = field(default_factory=ACTProcesses)
    metaforas_activas: list[str] = field(default_factory=list)
    marcadores_rapport: list[str] = field(default_factory=list)
    resumen_acumulado: str = ""
    selection_log: list[dict] = field(default_factory=list)
    # Phase transition tracking (v2.0 — D2)
    transicion: dict = field(default_factory=lambda: {
        "señales_integración": [],
        "señales_cierre": [],
        "rondas_en_fase_actual": 1,
        "fase_siguiente_probable": "",
    })

    def to_state_json(self) -> dict:
        """Serialize state for prompt injection (excludes transcript)."""
        return {
            "fase_terapeutica": self.fase_terapeutica,
            "estado_emocional": {
                "valencia": self.estado_emocional.valencia,
                "intensidad": self.estado_emocional.intensidad,
                "orientacion_accion": self.estado_emocional.orientacion_accion,
            },
            "procesos_act": self.procesos_act.to_dict(),
            "metaforas_activas": self.metaforas_activas,
            "marcadores_rapport": self.marcadores_rapport,
            "resumen_acumulado": self.resumen_acumulado,
            "transicion": self.transicion,
        }

    def update_from_llm(self, state_json: dict) -> None:
        """Update state from LLM state-update output."""
        old_phase = self.fase_terapeutica
        if "fase_terapeutica" in state_json:
            self.fase_terapeutica = state_json["fase_terapeutica"]
        if "estado_emocional" in state_json:
            emo = state_json["estado_emocional"]
            self.estado_emocional = EmotionalState(
                valencia=emo.get("valencia", self.estado_emocional.valencia),
                intensidad=emo.get("intensidad", self.estado_emocional.intensidad),
                orientacion_accion=emo.get("orientacion_accion", self.estado_emocional.orientacion_accion),
            )
        if "procesos_act" in state_json:
            self.procesos_act = ACTProcesses.from_dict(state_json["procesos_act"])
        if "metaforas_activas" in state_json:
            self.metaforas_activas = state_json["metaforas_activas"]
        if "marcadores_rapport" in state_json:
            self.marcadores_rapport = state_json["marcadores_rapport"]
        if "resumen_acumulado" in state_json:
            self.resumen_acumulado = state_json["resumen_acumulado"]
        if "transicion" in state_json:
            self.transicion = state_json["transicion"]
        # Auto-track rounds in current phase
        if self.fase_terapeutica != old_phase:
            self.transicion["rondas_en_fase_actual"] = 1
        else:
            self.transicion["rondas_en_fase_actual"] = self.transicion.get("rondas_en_fase_actual", 0) + 1

    def get_recent_transcript(self, window: int) -> str:
        """Format last N rounds as full transcript text."""
        recent = self.transcript[-window:] if window > 0 else self.transcript
        lines = []
        for r in recent:
            lines.append(f"[Turno {r.round_id} — PACIENTE]: {r.patient_message}")
            if r.selected_response_text:
                lines.append(f"[Turno {r.round_id} — TERAPEUTA]: {r.selected_response_text}")
        return "\n\n".join(lines)

    def get_selection_log_text(self) -> str:
        """Format selection log for prompt context."""
        if not self.selection_log:
            return "Ninguna selección previa."
        lines = []
        for entry in self.selection_log:
            lines.append(
                f"Turno {entry.get('round', '?')}: opción {entry.get('chosen', '?')} "
                f"({entry.get('tag', 'sin etiqueta')})"
            )
        return "\n".join(lines)


@dataclass
class SelectionResult:
    """Output from the evaluation/selection step."""

    round_id: int
    chosen_option: int  # 1, 2, or 3
    primary_tag: str
    reasoning: str
    raw_evaluation: dict = field(default_factory=dict)
    permutation_votes: list[int] | None = None  # for PERM mode
