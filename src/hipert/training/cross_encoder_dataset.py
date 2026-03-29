"""Dataset and data utilities for cross-encoder v2 training.

Handles tokenization of [CLS] symptom [SEP] sentence [SEP] pairs,
leave-symptom-out cross-validation splits, and sublist sampling for ListMLE.

Spec reference: hipert_v2_spec.md Sections 2, 5.5, 6.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Leave-symptom-out CV folds (spec Section 6.1)
SYMPTOM_CV_FOLDS = {
    1: {"train": list(range(1, 15)), "val": [15, 16, 17, 18]},       # Verbal_HI
    2: {"train": [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "val": [5, 6, 7, 8]},  # Motor_HI part
    3: {"train": list(range(5, 19)), "val": [1, 2, 3, 4]},           # Organization
    4: {"train": [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18], "val": [9, 10, 11, 12]},  # Sustained Attention
    5: {"train": list(range(1, 13)), "val": [13, 14, 15, 16, 17, 18]},  # Internal Drive + Verbal
}


class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder training.

    Tokenizes symptom-sentence pairs as:
        [CLS] <symptom_text> [SEP] <sentence_text> [SEP]
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize as sentence pair: symptom_text + sentence_text
        encoding = self.tokenizer(
            item["symptom_text"],
            item["sentence_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["score"], dtype=torch.long),
            "confidence": torch.tensor(item["confidence"], dtype=torch.float),
            "symptom_id": torch.tensor(item["symptom_id"], dtype=torch.long),
        }

        # Token type IDs (if available)
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return result

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate batch items."""
        result = {}
        for key in ["input_ids", "attention_mask", "label", "confidence", "symptom_id"]:
            result[key] = torch.stack([b[key] for b in batch])

        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])

        return result


def create_cv_splits(
    data: list[dict],
    fold: int,
) -> tuple[list[dict], list[dict]]:
    """Split data using leave-symptom-out cross-validation.

    Args:
        data: Full training data list.
        fold: Fold number (1-5).

    Returns:
        (train_data, val_data) tuple.
    """
    if fold not in SYMPTOM_CV_FOLDS:
        raise ValueError(f"Fold must be 1-5, got {fold}")

    fold_spec = SYMPTOM_CV_FOLDS[fold]
    train_symptoms = set(fold_spec["train"])
    val_symptoms = set(fold_spec["val"])

    train_data = [d for d in data if d["symptom_id"] in train_symptoms]
    val_data = [d for d in data if d["symptom_id"] in val_symptoms]

    logger.info(
        "Fold %d: train=%d examples (%d symptoms), val=%d examples (%d symptoms)",
        fold, len(train_data), len(train_symptoms),
        len(val_data), len(val_symptoms),
    )

    return train_data, val_data


class SublistSampler:
    """Sample stratified sublists for efficient ListMLE training.

    Each batch contains M sublists of size L, each from a single symptom.
    Sublists are stratified to include representatives from each grade.

    Spec reference: hipert_v2_spec.md Section 5.5
    """

    def __init__(
        self,
        data: list[dict],
        sublist_size: int = 64,
        num_sublists_per_batch: int = 8,
    ):
        self.sublist_size = sublist_size
        self.num_sublists = num_sublists_per_batch

        # Group data by symptom
        self.by_symptom: dict[int, list[dict]] = {}
        for d in data:
            sid = d["symptom_id"]
            if sid not in self.by_symptom:
                self.by_symptom[sid] = []
            self.by_symptom[sid].append(d)

        self.symptom_ids = sorted(self.by_symptom.keys())

    def sample_sublist(self, symptom_id: int, seed: int) -> list[dict]:
        """Sample a stratified sublist for one symptom."""
        symptom_data = self.by_symptom.get(symptom_id, [])
        if not symptom_data:
            return []

        # Group by grade
        by_grade: dict[int, list[dict]] = {g: [] for g in range(4)}
        for d in symptom_data:
            by_grade[d["score"]].append(d)

        rng = np.random.RandomState(seed)
        selected = []
        remaining_budget = self.sublist_size

        # Minimum 2 per present grade
        for grade in range(4):
            if by_grade[grade]:
                n_take = min(2, len(by_grade[grade]), remaining_budget)
                indices = rng.choice(len(by_grade[grade]), n_take, replace=False)
                selected.extend([by_grade[grade][i] for i in indices])
                remaining_budget -= n_take

        # Fill remaining proportionally
        pool = [d for d in symptom_data if d not in selected]
        if pool and remaining_budget > 0:
            n_take = min(remaining_budget, len(pool))
            indices = rng.choice(len(pool), n_take, replace=False)
            selected.extend([pool[i] for i in indices])

        return selected

    def sample_batch(self, step: int) -> list[dict]:
        """Sample a full batch of sublists from random symptoms."""
        rng = np.random.RandomState(step)
        batch = []
        chosen = rng.choice(self.symptom_ids, self.num_sublists, replace=True)
        for i, sid in enumerate(chosen):
            sublist = self.sample_sublist(sid, step * 100 + i)
            batch.extend(sublist)
        return batch
