"""ESConv data loader, translator, and synthetic MC generator.

ESConv (Emotional Support Conversation) contains 1,300 English dialogues
with strategy annotations. We use it for:
1. Translating dialogues to Spanish via DeepL
2. Generating synthetic multiple-choice dev sets for Task 2
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Strategy labels in ESConv
STRATEGIES = [
    "Question",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Self-disclosure",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Information",
    "Others",
]


@dataclass
class ESConvTurn:
    speaker: str  # "seeker" or "supporter"
    content: str
    strategy: str = ""  # Only for supporter turns
    content_es: str = ""  # Spanish translation


@dataclass
class ESConvDialogue:
    dialogue_id: int
    emotion_type: str
    problem_type: str
    situation: str
    situation_es: str = ""
    turns: list[ESConvTurn] = field(default_factory=list)


@dataclass
class MCInstance:
    """A multiple-choice instance for Task 2 development."""
    dialogue_id: int
    turn_index: int
    context: list[dict]  # Previous turns as {"speaker": ..., "text": ...}
    patient_message: str  # The seeker message before the supporter turn
    correct_response: str  # Original supporter response
    distractor_1: str  # From a different dialogue
    distractor_2: str  # From a different dialogue
    correct_option: int  # 1, 2, or 3 (shuffled position)
    options: dict[str, str] = field(default_factory=dict)  # {"option_1": ..., ...}
    metadata: dict = field(default_factory=dict)


def load_esconv(data_path: str | Path) -> list[ESConvDialogue]:
    """Load ESConv dialogues from the JSON file."""
    data_path = Path(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dialogues = []
    for i, d in enumerate(raw):
        turns = []
        for t in d["dialog"]:
            strategy = ""
            if t["speaker"] == "supporter" and "annotation" in t:
                strategy = t["annotation"].get("strategy", "")
            turns.append(ESConvTurn(
                speaker=t["speaker"],
                content=t["content"].strip(),
                strategy=strategy,
            ))
        dialogues.append(ESConvDialogue(
            dialogue_id=i,
            emotion_type=d.get("emotion_type", ""),
            problem_type=d.get("problem_type", ""),
            situation=d.get("situation", ""),
            turns=turns,
        ))

    logger.info("Loaded %d ESConv dialogues", len(dialogues))
    return dialogues


def filter_therapeutic_dialogues(
    dialogues: list[ESConvDialogue],
    emotions: set[str] | None = None,
    min_turns: int = 10,
) -> list[ESConvDialogue]:
    """Filter dialogues to those most relevant for MentalRiskES."""
    if emotions is None:
        # Focus on anxiety + depression (closest to MentalRiskES)
        emotions = {"anxiety", "depression", "sadness"}

    filtered = [
        d for d in dialogues
        if d.emotion_type in emotions and len(d.turns) >= min_turns
    ]
    logger.info(
        "Filtered %d -> %d dialogues (emotions=%s, min_turns=%d)",
        len(dialogues), len(filtered), emotions, min_turns,
    )
    return filtered


def translate_dialogues(
    dialogues: list[ESConvDialogue],
    translator,
    max_dialogues: int | None = None,
) -> list[ESConvDialogue]:
    """Translate ESConv dialogues from English to Spanish using DeepL.

    Args:
        dialogues: List of ESConvDialogue objects.
        translator: DeepLTranslator instance.
        max_dialogues: Limit number of dialogues to translate.

    Returns:
        Same dialogues with content_es and situation_es filled in.
    """
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    # Collect all texts to translate in one batch
    all_texts: list[str] = []
    text_map: list[tuple[int, str, int | None]] = []  # (dialogue_idx, field, turn_idx)

    for i, d in enumerate(dialogues):
        all_texts.append(d.situation)
        text_map.append((i, "situation", None))
        for j, t in enumerate(d.turns):
            all_texts.append(t.content)
            text_map.append((i, "turn", j))

    logger.info("Translating %d texts from %d dialogues", len(all_texts), len(dialogues))
    translated = translator.translate_batch(all_texts)

    # Assign translations
    for (d_idx, field_name, t_idx), trans in zip(text_map, translated):
        if field_name == "situation":
            dialogues[d_idx].situation_es = trans
        else:
            dialogues[d_idx].turns[t_idx].content_es = trans

    logger.info("Translation complete. Stats: %s", translator.stats())
    return dialogues


def _find_supporter_turns(dialogue: ESConvDialogue) -> list[int]:
    """Find supporter turn indices that are preceded by a seeker turn."""
    indices = []
    for i, turn in enumerate(dialogue.turns):
        if turn.speaker == "supporter" and i > 0 and dialogue.turns[i - 1].speaker == "seeker":
            indices.append(i)
    return indices


def generate_cross_dialogue_mc(
    dialogues: list[ESConvDialogue],
    n_instances: int = 100,
    min_context_turns: int = 4,
    use_spanish: bool = True,
    seed: int = 42,
) -> list[MCInstance]:
    """Generate MC instances using cross-dialogue distractors (Strategy B from spec).

    For each selected supporter turn:
    - Correct answer: the original supporter response
    - Distractor 1: a supporter response from a different dialogue at a similar position
    - Distractor 2: a supporter response from a different dialogue with a different strategy

    Args:
        dialogues: Translated ESConv dialogues.
        n_instances: Number of MC instances to generate.
        min_context_turns: Minimum conversation turns before the target turn.
        use_spanish: Use Spanish translations (content_es) or English (content).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)

    def _text(turn: ESConvTurn) -> str:
        if use_spanish and turn.content_es:
            return turn.content_es
        return turn.content

    def _situation(d: ESConvDialogue) -> str:
        if use_spanish and d.situation_es:
            return d.situation_es
        return d.situation

    # Build pool of supporter turns with enough context
    pool: list[tuple[int, int]] = []  # (dialogue_idx, turn_idx)
    for d_idx, d in enumerate(dialogues):
        for t_idx in _find_supporter_turns(d):
            if t_idx >= min_context_turns:
                pool.append((d_idx, t_idx))

    if len(pool) < n_instances:
        logger.warning(
            "Only %d eligible turns available (requested %d)", len(pool), n_instances
        )
        n_instances = len(pool)

    # Build strategy-indexed supporter pool for diverse distractors
    strategy_pool: dict[str, list[tuple[int, int]]] = {}
    for d_idx, t_idx in pool:
        s = dialogues[d_idx].turns[t_idx].strategy
        strategy_pool.setdefault(s, []).append((d_idx, t_idx))

    selected = rng.sample(pool, n_instances)
    instances: list[MCInstance] = []

    for d_idx, t_idx in selected:
        d = dialogues[d_idx]
        correct_turn = d.turns[t_idx]
        patient_turn = d.turns[t_idx - 1]

        # Build context (all turns before the patient message)
        context = []
        for t in d.turns[:t_idx - 1]:
            role = "patient" if t.speaker == "seeker" else "therapist"
            context.append({"speaker": role, "text": _text(t)})

        # Distractor 1: similar position from different dialogue
        candidates_pos = [
            (di, ti) for di, ti in pool
            if di != d_idx and abs(ti - t_idx) <= 4
        ]
        if not candidates_pos:
            candidates_pos = [(di, ti) for di, ti in pool if di != d_idx]
        dist1_idx = rng.choice(candidates_pos)
        dist1_text = _text(dialogues[dist1_idx[0]].turns[dist1_idx[1]])

        # Distractor 2: different strategy from different dialogue
        other_strategies = [s for s in strategy_pool if s != correct_turn.strategy]
        if other_strategies:
            chosen_strategy = rng.choice(other_strategies)
            candidates_strat = [
                (di, ti) for di, ti in strategy_pool[chosen_strategy]
                if di != d_idx
            ]
        else:
            candidates_strat = [(di, ti) for di, ti in pool if di != d_idx]
        if not candidates_strat:
            candidates_strat = [(di, ti) for di, ti in pool if di != d_idx]
        dist2_idx = rng.choice(candidates_strat)
        dist2_text = _text(dialogues[dist2_idx[0]].turns[dist2_idx[1]])

        # Shuffle options
        options_list = [_text(correct_turn), dist1_text, dist2_text]
        correct_pos = 0  # Will be updated after shuffle
        order = list(range(3))
        rng.shuffle(order)
        shuffled = [options_list[i] for i in order]
        correct_pos = order.index(0) + 1  # 1-indexed

        options = {
            "option_1": shuffled[0],
            "option_2": shuffled[1],
            "option_3": shuffled[2],
        }

        instances.append(MCInstance(
            dialogue_id=d_idx,
            turn_index=t_idx,
            context=context,
            patient_message=_text(patient_turn),
            correct_response=_text(correct_turn),
            distractor_1=dist1_text,
            distractor_2=dist2_text,
            correct_option=correct_pos,
            options=options,
            metadata={
                "emotion_type": d.emotion_type,
                "problem_type": d.problem_type,
                "correct_strategy": correct_turn.strategy,
                "distractor_1_source": f"d{dist1_idx[0]}_t{dist1_idx[1]}",
                "distractor_2_source": f"d{dist2_idx[0]}_t{dist2_idx[1]}",
                "distractor_2_strategy": dialogues[dist2_idx[0]].turns[dist2_idx[1]].strategy,
            },
        ))

    logger.info("Generated %d MC instances", len(instances))
    return instances


def save_mc_dataset(
    instances: list[MCInstance],
    output_path: str | Path,
) -> None:
    """Save MC dataset as JSON (round-file compatible format)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for inst in instances:
        records.append({
            "dialogue_id": inst.dialogue_id,
            "turn_index": inst.turn_index,
            "context": inst.context,
            "patient_message": inst.patient_message,
            "options": inst.options,
            "correct_option": inst.correct_option,
            "metadata": inst.metadata,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d MC instances to %s", len(records), output_path)


def save_translated_dialogues(
    dialogues: list[ESConvDialogue],
    output_path: str | Path,
) -> None:
    """Save translated dialogues as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for d in dialogues:
        turns = []
        for t in d.turns:
            turns.append({
                "speaker": t.speaker,
                "content_en": t.content,
                "content_es": t.content_es,
                "strategy": t.strategy,
            })
        records.append({
            "dialogue_id": d.dialogue_id,
            "emotion_type": d.emotion_type,
            "problem_type": d.problem_type,
            "situation_en": d.situation,
            "situation_es": d.situation_es,
            "turns": turns,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d translated dialogues to %s", len(records), output_path)
