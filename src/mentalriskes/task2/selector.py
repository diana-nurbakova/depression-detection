"""Response selector: evaluates 3 options and selects the best one.

Supports pipeline variants A (single CoT), B (2-step), B+ (2.5-step)
and evaluation framings FUNC, HYB, TOM-B, TOM-C.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from ..llm_client import LLMClient, parse_json_response
from .models import SelectionResult, SharedState
from .prompts import (
    CHARACTERIZATION_SYSTEM,
    SINGLE_COT_SYSTEM,
    STATE_UPDATE_SYSTEM,
    build_selection_system,
    build_selection_user,
    build_state_update_user,
)

logger = logging.getLogger(__name__)


class Task2Selector:
    """Orchestrates state tracking and response selection for a single session."""

    def __init__(
        self,
        llm: LLMClient,
        framing: str = "FUNC",
        pipeline: str = "B",
        lang: str = "es",
        lookback_window: int = 3,
    ) -> None:
        self.llm = llm
        self.framing = framing
        self.pipeline = pipeline
        self.lang = lang
        self.lookback_window = lookback_window
        self.state = SharedState()

    def process_round(
        self,
        round_id: int,
        patient_message: str,
        options: dict[str, str],
    ) -> SelectionResult:
        """Process one round: update state → evaluate → select."""
        # Store round in transcript
        from .models import RoundRecord
        record = RoundRecord(
            round_id=round_id,
            patient_message=patient_message,
            options=options,
        )
        self.state.transcript.append(record)

        if self.pipeline == "A":
            result = self._single_cot(round_id, patient_message, options)
        elif self.pipeline == "B":
            self._step1_state_update(round_id, patient_message)
            result = self._step2_evaluate_select(round_id, patient_message, options)
        elif self.pipeline == "B+":
            self._step1_state_update(round_id, patient_message)
            char_tags = self._step1_5_characterize(round_id, patient_message, options)
            result = self._step2_evaluate_select(
                round_id, patient_message, options, characterization_tags=char_tags
            )
        else:
            raise ValueError(f"Unknown pipeline variant: {self.pipeline}")

        # Record selection in state
        record.selected_option = result.chosen_option
        record.selected_response_text = options[f"option_{result.chosen_option}"]
        self.state.selection_log.append({
            "round": round_id,
            "chosen": result.chosen_option,
            "tag": result.primary_tag,
        })

        return result

    def _step1_state_update(self, round_id: int, patient_message: str) -> None:
        """Step 1: Update shared state from patient message."""
        prev_state = json.dumps(self.state.to_state_json(), ensure_ascii=False, indent=2)
        prev_response = None
        if len(self.state.transcript) > 1:
            prev_record = self.state.transcript[-2]
            prev_response = prev_record.selected_response_text

        system = STATE_UPDATE_SYSTEM[self.lang]
        user = build_state_update_user(
            previous_state_json=prev_state,
            selected_response_text=prev_response,
            patient_input=patient_message,
            round_number=round_id,
            lang=self.lang,
        )

        response = self.llm.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        parsed = parse_json_response(response)
        if parsed:
            self.state.update_from_llm(parsed)
            logger.info("Round %d state updated: phase=%s", round_id, self.state.fase_terapeutica)
        else:
            logger.warning("Round %d: failed to parse state update response", round_id)

    def _step1_5_characterize(
        self,
        round_id: int,
        patient_message: str,
        options: dict[str, str],
    ) -> str:
        """Step 1.5 (B+ only): Tag each option without scoring."""
        system = CHARACTERIZATION_SYSTEM[self.lang]
        user_parts = [
            f"Turno {round_id}",
            f"Paciente: {patient_message}",
            f"Opción 1: {options['option_1']}",
            f"Opción 2: {options['option_2']}",
            f"Opción 3: {options['option_3']}",
        ]

        response = self.llm.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ])

        return response  # raw text, passed as context to step 2

    def _step2_evaluate_select(
        self,
        round_id: int,
        patient_message: str,
        options: dict[str, str],
        characterization_tags: str | None = None,
    ) -> SelectionResult:
        """Step 2: Evaluate options and select best."""
        system = build_selection_system(self.framing, self.lang)
        state_json = json.dumps(self.state.to_state_json(), ensure_ascii=False, indent=2)
        recent = self.state.get_recent_transcript(self.lookback_window)
        sel_log = self.state.get_selection_log_text()

        user = build_selection_user(
            state_json=state_json,
            recent_transcript=recent,
            patient_input=patient_message,
            options=options,
            selection_log=sel_log,
            round_number=round_id,
            lang=self.lang,
            characterization_tags=characterization_tags,
        )

        response = self.llm.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        return self._parse_selection(round_id, response)

    def _single_cot(
        self,
        round_id: int,
        patient_message: str,
        options: dict[str, str],
    ) -> SelectionResult:
        """Variant A: single prompt with chain-of-thought."""
        system = SINGLE_COT_SYSTEM[self.lang]
        transcript = self.state.get_recent_transcript(self.lookback_window)
        sel_log = self.state.get_selection_log_text()

        user_parts = [
            f"HISTORIAL:\n{transcript}" if transcript else "Primera interacción.",
            f"SELECCIONES PREVIAS:\n{sel_log}",
            f"MENSAJE DEL PACIENTE (Turno {round_id}):\n{patient_message}",
            f"OPCIÓN 1:\n{options['option_1']}",
            f"OPCIÓN 2:\n{options['option_2']}",
            f"OPCIÓN 3:\n{options['option_3']}",
            "Analiza el estado, evalúa las opciones y selecciona. Responde con JSON.",
        ]

        response = self.llm.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ])

        # Also try to extract state from the CoT response
        parsed = parse_json_response(response)
        if parsed:
            # Update state from any state fields in the response
            for key in ("fase_terapeutica", "estado_emocional", "procesos_act",
                        "metaforas_activas", "resumen_acumulado"):
                if key in parsed:
                    self.state.update_from_llm({key: parsed[key]})

        return self._parse_selection(round_id, response)

    def _parse_selection(self, round_id: int, raw_response: str) -> SelectionResult:
        """Parse LLM response into SelectionResult."""
        parsed = parse_json_response(raw_response)

        if not parsed:
            logger.warning("Round %d: no JSON in selection response, defaulting to option 1", round_id)
            return SelectionResult(
                round_id=round_id,
                chosen_option=1,
                primary_tag="parse_error",
                reasoning="Failed to parse LLM response",
                raw_evaluation={"raw": raw_response[:500]},
            )

        # Extract chosen option number from various response formats
        chosen = self._extract_chosen_option(parsed)
        tag = self._extract_primary_tag(parsed)
        reasoning = self._extract_reasoning(parsed)

        return SelectionResult(
            round_id=round_id,
            chosen_option=chosen,
            primary_tag=tag,
            reasoning=reasoning,
            raw_evaluation=parsed,
        )

    @staticmethod
    def _get_key(d: dict, *candidates: str, default=None):
        """Get value from dict trying multiple key variants (accent-tolerant)."""
        for key in candidates:
            if key in d:
                return d[key]
        return default if default is not None else {}

    @classmethod
    def _extract_chosen_option(cls, parsed: dict) -> int:
        """Extract the chosen option number from parsed JSON."""
        # Try all selection key variants (accented/unaccented/English)
        sel = cls._get_key(parsed, "selección", "seleccion", "selection")
        chosen = cls._get_key(sel, "opcion_elegida", "chosen_option") if isinstance(sel, dict) else {}
        if isinstance(chosen, dict):
            num = chosen.get("numero", chosen.get("number", 1))
        elif isinstance(chosen, (int, float)):
            num = int(chosen)
        else:
            num = 1

        # Also check structured_selection (TOM-B)
        if num == 1:
            ss = cls._get_key(parsed, "selección_estructurada", "seleccion_estructurada", "structured_selection")
            if isinstance(ss, dict):
                oe = cls._get_key(ss, "opcion_elegida", "chosen_option")
                if isinstance(oe, dict):
                    num = oe.get("numero", oe.get("number", 1))

        return int(num) if num in (1, 2, 3) else 1

    @classmethod
    def _extract_primary_tag(cls, parsed: dict) -> str:
        """Extract the primary therapeutic tag."""
        sel = cls._get_key(parsed, "selección", "seleccion", "selection")
        if isinstance(sel, dict):
            chosen = cls._get_key(sel, "opcion_elegida", "chosen_option")
            if isinstance(chosen, dict):
                return chosen.get("etiqueta_principal", chosen.get("primary_tag", "unknown"))

        # TOM-B
        ss = cls._get_key(parsed, "selección_estructurada", "seleccion_estructurada", "structured_selection")
        if isinstance(ss, dict):
            oe = cls._get_key(ss, "opcion_elegida", "chosen_option")
            if isinstance(oe, dict):
                return oe.get("etiqueta_principal", oe.get("primary_tag", "unknown"))

        return "unknown"

    @classmethod
    def _extract_reasoning(cls, parsed: dict) -> str:
        """Extract the reasoning string."""
        sel = cls._get_key(parsed, "selección", "seleccion", "selection")
        if isinstance(sel, dict):
            return sel.get("razonamiento", sel.get("reasoning", ""))
        return ""


def run_with_permutation_voting(
    llm: LLMClient,
    state: SharedState,
    round_id: int,
    patient_message: str,
    options: dict[str, str],
    framing: str = "FUNC",
    pipeline: str = "B",
    lang: str = "es",
    lookback_window: int = 3,
) -> SelectionResult:
    """Run 3 permutations and take majority vote.

    Permutations:
    - Original: [1, 2, 3]
    - Rotation A: [2, 3, 1] → presented as [1, 2, 3]
    - Rotation B: [3, 1, 2] → presented as [1, 2, 3]
    """
    permutations = [
        {"option_1": "option_1", "option_2": "option_2", "option_3": "option_3"},  # original
        {"option_1": "option_2", "option_2": "option_3", "option_3": "option_1"},  # rotation A
        {"option_1": "option_3", "option_2": "option_1", "option_3": "option_2"},  # rotation B
    ]
    reverse_maps = [
        {1: 1, 2: 2, 3: 3},
        {1: 2, 2: 3, 3: 1},
        {1: 3, 2: 1, 3: 2},
    ]

    votes: list[int] = []

    for perm, rev_map in zip(permutations, reverse_maps):
        # Create permuted options
        perm_options = {
            "option_1": options[perm["option_1"]],
            "option_2": options[perm["option_2"]],
            "option_3": options[perm["option_3"]],
        }

        # Create a temporary selector with shared state (snapshot)
        selector = Task2Selector(
            llm=llm, framing=framing, pipeline=pipeline,
            lang=lang, lookback_window=lookback_window,
        )
        # Copy the state (but don't add this round to transcript again)
        selector.state = SharedState(
            transcript=list(state.transcript),
            fase_terapeutica=state.fase_terapeutica,
            estado_emocional=state.estado_emocional,
            procesos_act=state.procesos_act,
            metaforas_activas=list(state.metaforas_activas),
            marcadores_rapport=list(state.marcadores_rapport),
            resumen_acumulado=state.resumen_acumulado,
            selection_log=list(state.selection_log),
        )

        # Run evaluation only (skip state update — already done)
        result = selector._step2_evaluate_select(
            round_id, patient_message, perm_options
        )

        # Map back to original numbering
        original_choice = rev_map[result.chosen_option]
        votes.append(original_choice)
        logger.info(
            "Permutation vote round %d: perm chose %d → original %d",
            round_id, result.chosen_option, original_choice,
        )

    # Majority vote
    counter = Counter(votes)
    winner, count = counter.most_common(1)[0]
    if count == 1:
        # 3-way tie: use original ordering's answer
        winner = votes[0]
        logger.info("Round %d: 3-way tie, using original answer %d", round_id, winner)
    else:
        logger.info("Round %d: majority vote %d (%d/3)", round_id, winner, count)

    return SelectionResult(
        round_id=round_id,
        chosen_option=winner,
        primary_tag="permutation_vote",
        reasoning=f"Majority vote: {votes} → {winner}",
        permutation_votes=votes,
    )
