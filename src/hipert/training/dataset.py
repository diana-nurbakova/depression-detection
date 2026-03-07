"""PyTorch datasets for encoder training with curriculum scheduling.

Supports:
- BDI-Sen (graded 0-3, 21 BDI-II symptoms)
- eRisk 2025 T1 (binary, 21 BDI-II symptoms, full PRE/TEXT/POST context)
- eRisk 2023 T1 (binary, 21 BDI-II symptoms, text-only)
- ADHD silver labels (graded 0-3, 18 ASRS symptoms, confidence weights)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ScoringExample:
    """A single (sentence, symptom, label) training example."""
    text: str
    pre: str
    post: str
    symptom_id: int       # 1-indexed (ASRS 1-18 or BDI-II 1-21)
    label: int            # 0-3 for graded, 0-1 for binary
    weight: float = 1.0   # confidence weight (1.0 for gold data)
    difficulty: float = 0.5  # curriculum difficulty (0=easy, 1=hard)


class ScoringDataset(Dataset):
    """Dataset for training the symptom-conditioned encoder."""

    def __init__(
        self,
        examples: list[ScoringExample],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        include_context: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Tokenize main text
        text_enc = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "text_input_ids": text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "symptom_id": torch.tensor(ex.symptom_id, dtype=torch.long),
            "label": torch.tensor(ex.label, dtype=torch.long),
            "weight": torch.tensor(ex.weight, dtype=torch.float),
            "difficulty": torch.tensor(ex.difficulty, dtype=torch.float),
        }

        # Tokenize context if available and requested
        if self.include_context and ex.pre:
            pre_enc = self.tokenizer(
                ex.pre,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["pre_input_ids"] = pre_enc["input_ids"].squeeze(0)
            item["pre_attention_mask"] = pre_enc["attention_mask"].squeeze(0)

        if self.include_context and ex.post:
            post_enc = self.tokenizer(
                ex.post,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["post_input_ids"] = post_enc["input_ids"].squeeze(0)
            item["post_attention_mask"] = post_enc["attention_mask"].squeeze(0)

        return item

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Custom collation that handles optional context fields."""
        result = {}
        for key in ["text_input_ids", "text_attention_mask", "symptom_id",
                     "label", "weight", "difficulty"]:
            result[key] = torch.stack([b[key] for b in batch])

        # Optional context fields
        for key in ["pre_input_ids", "pre_attention_mask",
                     "post_input_ids", "post_attention_mask"]:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])

        return result


class CurriculumScheduler:
    """Curriculum learning scheduler for Stage B training.

    Root-p competence growth:
        c(t) = min(1, (0.01^p + t/T * (1 - 0.01^p)) ^ (1/p))

    Inclusion probability:
        P_include(s, t) = sigmoid(beta(t) * (c(t) - difficulty(s)))

    Phases:
        1. c < 0.35: Motor H/I, Verbal H/I (easy symptoms, pi=0.2)
        2. 0.35 <= c < 0.65: Organization, Memory (medium, pi=0.5)
        3. c >= 0.65: Sustained Attention (hard, pi=0.8)
    """

    def __init__(
        self,
        total_steps: int,
        p: float = 2.0,
        beta_scale: float = 5.0,
    ):
        self.total_steps = max(total_steps, 1)
        self.p = p
        self.beta_scale = beta_scale

    def competence(self, step: int) -> float:
        """Compute competence level at given step."""
        t_ratio = step / self.total_steps
        base = 0.01 ** self.p
        return min(1.0, (base + t_ratio * (1.0 - base)) ** (1.0 / self.p))

    def inclusion_prob(self, step: int, difficulty: float) -> float:
        """Compute inclusion probability for a sample."""
        c = self.competence(step)
        beta = self.beta_scale * c
        return 1.0 / (1.0 + math.exp(-beta * (c - difficulty)))

    def filter_batch(
        self,
        examples: list[ScoringExample],
        step: int,
    ) -> list[ScoringExample]:
        """Filter examples based on curriculum inclusion probability."""
        c = self.competence(step)
        filtered = []
        for ex in examples:
            prob = self.inclusion_prob(step, ex.difficulty)
            if torch.rand(1).item() < prob:
                filtered.append(ex)
        return filtered if filtered else examples[:1]  # never return empty


# Difficulty priors for ASRS symptoms (from spec)
SYMPTOM_DIFFICULTY = {
    # Motor H/I — easiest (most concrete behaviors)
    5: 0.2, 6: 0.2, 12: 0.2, 13: 0.25, 14: 0.25,
    # Verbal H/I — easy
    15: 0.3, 16: 0.3, 17: 0.3, 18: 0.3,
    # Organization/Memory — medium
    1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5,
    # Sustained Attention — hardest (most confounders)
    7: 0.8, 8: 0.8, 9: 0.8, 10: 0.8, 11: 0.8,
}


def load_silver_labels(
    silver_labels_dir: Path,
    symptom_ids: list[int] | None = None,
    min_confidence: float = 0.3,
) -> list[ScoringExample]:
    """Load ADHD silver labels from scored JSONL files.

    Args:
        silver_labels_dir: Directory with symptom_{id}.jsonl files.
        symptom_ids: Which symptoms to load (default: all 1-18).
        min_confidence: Minimum confidence_weight to include.

    Returns:
        List of ScoringExample with confidence-weighted labels.
    """
    if symptom_ids is None:
        symptom_ids = list(range(1, 19))

    examples = []
    for sid in symptom_ids:
        path = silver_labels_dir / f"symptom_{sid}.jsonl"
        if not path.exists():
            continue

        difficulty = SYMPTOM_DIFFICULTY.get(sid, 0.5)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                weight = data.get("confidence_weight", 0.5)
                if weight < min_confidence:
                    continue

                examples.append(ScoringExample(
                    text=data.get("text", ""),
                    pre=data.get("pre", ""),
                    post=data.get("post", ""),
                    symptom_id=sid,
                    label=data.get("final_label", 0),
                    weight=weight,
                    difficulty=difficulty,
                ))

    logger.info(
        "Loaded %d silver label examples from %d symptoms",
        len(examples), len(set(e.symptom_id for e in examples)),
    )
    return examples
