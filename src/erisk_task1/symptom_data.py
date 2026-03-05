"""Data loaders for DepreSym, ReDSM5, and BDI-Sen datasets.

Produces a unified multi-label training dataset for the sentence transformer:
each sample is (sentence_text, 21-dim label vector, weight).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import BDI_ITEMS

logger = logging.getLogger(__name__)

# --- Name mappings ---

# BDI-Sen symptom names → BDI-II item IDs (1-indexed)
BDISEN_SYMPTOM_TO_ITEM: dict[str, int] = {
    "Sadness": 1,
    "Pessimism": 2,
    "Sense_of_failure": 3,
    "Loss_of_Pleasure": 4,
    "Guilty_feelings": 5,
    "Sense_of_punishment": 6,
    "Self-dislike": 7,
    "Self-incrimination": 8,
    "Suicidal_ideas": 9,
    "Crying": 10,
    "Agitation": 11,
    "Social_withdrawal": 12,
    "Indecision": 13,
    "Feelings_of_worthlessness": 14,
    "Loss_of_energy": 15,
    "Change_of_sleep": 16,
    "Irritability": 17,
    "Changes_in_appetite": 18,
    "Concentration_difficulty": 19,
    "Tiredness_or_fatigue": 20,
    "Loss_of_interest_in_sex": 21,
}

# DSM-5 criterion → list of mapped BDI-II item IDs
DSM5_TO_BDI_ITEMS: dict[str, list[int]] = {
    "DEPRESSED_MOOD": [1, 10, 17],
    "ANHEDONIA": [4, 12],
    "APPETITE_CHANGE": [18],
    "SLEEP_ISSUES": [16],
    "PSYCHOMOTOR": [11],
    "FATIGUE": [15, 20],
    "WORTHLESSNESS": [5, 6, 7, 8, 14],
    "COGNITIVE_ISSUES": [13, 19],
    "SUICIDAL_THOUGHTS": [9],
    "SPECIAL_CASE": [],  # protective/positive signals — used as hard negatives
}

NUM_ITEMS = 21


@dataclass
class LabelledSentence:
    """A single training sample for the sentence transformer."""
    text: str
    labels: np.ndarray  # shape (21,), binary multi-label
    weight: float = 1.0
    source: str = ""  # "depresym", "bdisen", "redsm5"


def load_depresym(
    pools_path: str | Path,
    qrels_path: str | Path,
) -> list[LabelledSentence]:
    """Load DepreSym dataset by joining pools_docnos.json with a qrels file.

    Args:
        pools_path: Path to pools_docnos.json (contains sentence texts).
        qrels_path: Path to a qrels file (e.g. qrels-consensus.txt).

    Returns:
        List of LabelledSentence with 21-dim multi-label vectors.
    """
    # Load sentence texts from pools
    with open(pools_path, encoding="utf-8") as f:
        data = json.load(f)
    pools = data["pools"]

    # Build sentence_id → text mapping (sentences can appear in multiple pools)
    sentence_texts: dict[str, str] = {}
    for pool in pools:
        for sentence_id, text in pool["pool_list"]:
            sentence_texts[sentence_id] = text

    # Load qrels: SYMPTOM_ID 0 sentence-id RELEVANCE
    # sentence_id → {item_id: relevance}
    sentence_labels: dict[str, dict[int, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            symptom_id = int(parts[0])  # 1-indexed BDI-II item
            sentence_id = parts[2]
            relevance = int(parts[3])
            if sentence_id not in sentence_labels:
                sentence_labels[sentence_id] = {}
            sentence_labels[sentence_id][symptom_id] = relevance

    # Build labelled samples
    samples = []
    for sentence_id, item_relevances in sentence_labels.items():
        text = sentence_texts.get(sentence_id)
        if text is None:
            continue

        labels = np.zeros(NUM_ITEMS, dtype=np.float32)
        for item_id, rel in item_relevances.items():
            if 1 <= item_id <= NUM_ITEMS and rel == 1:
                labels[item_id - 1] = 1.0

        samples.append(LabelledSentence(
            text=text,
            labels=labels,
            weight=1.0,
            source="depresym",
        ))

    logger.info(
        "Loaded DepreSym: %d sentences, %d with ≥1 positive label",
        len(samples),
        sum(1 for s in samples if s.labels.sum() > 0),
    )
    return samples


def load_redsm5(
    annotations_path: str | Path,
    posts_path: str | Path,
) -> list[LabelledSentence]:
    """Load ReDSM5 dataset with DSM-5 → BDI-II label mapping.

    Positive annotations (status=1) are mapped to all corresponding BDI-II items
    with reduced weight (0.5) since the mapping is one-to-many.
    SPECIAL_CASE annotations are treated as hard negatives (all-zero labels).
    """
    # Load post texts
    post_texts: dict[str, str] = {}
    with open(posts_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_texts[row["post_id"]] = row["text"]

    # Load annotations: each row maps a sentence within a post to a DSM-5 criterion
    # sentence_id → {dsm5_symptom: status}
    sentence_annotations: dict[str, dict[str, int]] = {}
    sentence_explanations: dict[str, list[str]] = {}
    with open(annotations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sentence_id"]
            symptom = row["DSM5_symptom"]
            status = int(row["status"])
            if sid not in sentence_annotations:
                sentence_annotations[sid] = {}
                sentence_explanations[sid] = []
            sentence_annotations[sid][symptom] = status
            if row.get("explanation"):
                sentence_explanations[sid].append(row["explanation"])

    # Build labelled samples from sentence-level annotations
    samples = []
    # Get sentence texts from posts_path — sentences are identified by sentence_id
    # which is post_id + "_" + sentence_index
    sentence_texts: dict[str, str] = {}
    with open(annotations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sentence_id"]
            if sid not in sentence_texts and row.get("sentence_text"):
                sentence_texts[sid] = row["sentence_text"]

    for sid, annotations in sentence_annotations.items():
        text = sentence_texts.get(sid)
        if text is None:
            # Fall back to full post text
            post_id = row.get("post_id", "")
            text = post_texts.get(post_id)
        if text is None:
            continue

        labels = np.zeros(NUM_ITEMS, dtype=np.float32)
        is_positive = False

        for dsm5_symptom, status in annotations.items():
            if status != 1:
                continue
            mapped_items = DSM5_TO_BDI_ITEMS.get(dsm5_symptom, [])
            if not mapped_items:
                # SPECIAL_CASE: hard negative — keep all-zero labels
                continue
            for item_id in mapped_items:
                labels[item_id - 1] = 1.0
                is_positive = True

        # Weight: 0.5 for positives (one-to-many mapping uncertainty),
        # 1.0 for hard negatives (SPECIAL_CASE — all zeros, high confidence)
        weight = 0.5 if is_positive else 1.0

        samples.append(LabelledSentence(
            text=text,
            labels=labels,
            weight=weight,
            source="redsm5",
        ))

    logger.info(
        "Loaded ReDSM5: %d sentences, %d positive, %d hard negatives",
        len(samples),
        sum(1 for s in samples if s.labels.sum() > 0),
        sum(1 for s in samples if s.labels.sum() == 0),
    )
    return samples


def load_bdisen(data_path: str | Path) -> list[LabelledSentence]:
    """Load BDI-Sen dataset from JSONL format.

    Each line: {"sentence": "...", "annotations": [{"symptom": "...", "severity": N, "label": 0|1}]}
    """
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("sentence", "")
            if not text:
                continue

            labels = np.zeros(NUM_ITEMS, dtype=np.float32)
            for ann in obj.get("annotations", []):
                symptom_name = ann.get("symptom", "")
                label = ann.get("label", 0)
                item_id = BDISEN_SYMPTOM_TO_ITEM.get(symptom_name)
                if item_id is not None and label == 1:
                    labels[item_id - 1] = 1.0

            samples.append(LabelledSentence(
                text=text,
                labels=labels,
                weight=1.0,
                source="bdisen",
            ))

    logger.info(
        "Loaded BDI-Sen: %d sentences, %d with ≥1 positive label",
        len(samples),
        sum(1 for s in samples if s.labels.sum() > 0),
    )
    return samples


@dataclass
class SymptomDataConfig:
    """Paths to all symptom datasets."""
    depresym_pools: str = "data/DepreSym_dataset/pools_docnos.json"
    depresym_qrels: str = "data/DepreSym_dataset/qrels-consensus.txt"
    redsm5_annotations: str = "data/RedSM5/redsm5_annotations.csv"
    redsm5_posts: str = "data/RedSM5/redsm5_posts.csv"
    bdisen_data: str = "data/BDI-Sen/full_dataset/bdi_unified.jsonl"


def load_all_datasets(
    config: SymptomDataConfig | None = None,
    base_dir: str | Path = ".",
) -> list[LabelledSentence]:
    """Load and merge all available datasets.

    Returns combined list of LabelledSentence ready for training.
    """
    if config is None:
        config = SymptomDataConfig()

    base = Path(base_dir)
    all_samples: list[LabelledSentence] = []

    # DepreSym (primary)
    pools_path = base / config.depresym_pools
    qrels_path = base / config.depresym_qrels
    if pools_path.exists() and qrels_path.exists():
        all_samples.extend(load_depresym(pools_path, qrels_path))
    else:
        logger.warning("DepreSym data not found at %s", pools_path)

    # ReDSM5 (supplementary)
    ann_path = base / config.redsm5_annotations
    posts_path = base / config.redsm5_posts
    if ann_path.exists() and posts_path.exists():
        all_samples.extend(load_redsm5(ann_path, posts_path))
    else:
        logger.warning("ReDSM5 data not found at %s", ann_path)

    # BDI-Sen (supplementary)
    bdisen_path = base / config.bdisen_data
    if bdisen_path.exists():
        all_samples.extend(load_bdisen(bdisen_path))
    else:
        logger.warning("BDI-Sen data not found at %s", bdisen_path)

    logger.info(
        "Total combined: %d samples (DepreSym: %d, ReDSM5: %d, BDI-Sen: %d)",
        len(all_samples),
        sum(1 for s in all_samples if s.source == "depresym"),
        sum(1 for s in all_samples if s.source == "redsm5"),
        sum(1 for s in all_samples if s.source == "bdisen"),
    )
    return all_samples


def compute_class_weights(samples: list[LabelledSentence]) -> np.ndarray:
    """Compute per-symptom positive weights for weighted BCE loss.

    Returns array of shape (21,) with pos_weight per symptom.
    """
    total = len(samples)
    pos_counts = np.zeros(NUM_ITEMS, dtype=np.float64)
    for s in samples:
        pos_counts += s.labels

    # pos_weight = num_negatives / num_positives (per symptom)
    neg_counts = total - pos_counts
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    return weights.astype(np.float32)


def dataset_statistics(samples: list[LabelledSentence]) -> dict:
    """Compute per-symptom statistics for the combined dataset."""
    total = len(samples)
    pos_counts = np.zeros(NUM_ITEMS, dtype=np.int64)
    source_counts: dict[str, int] = {}

    for s in samples:
        pos_counts += s.labels.astype(np.int64)
        source_counts[s.source] = source_counts.get(s.source, 0) + 1

    stats = {
        "total_samples": total,
        "source_counts": source_counts,
        "per_symptom": {},
    }
    for i in range(NUM_ITEMS):
        item_name = BDI_ITEMS[i + 1]
        stats["per_symptom"][item_name] = {
            "positive": int(pos_counts[i]),
            "negative": int(total - pos_counts[i]),
            "ratio": float(pos_counts[i] / total) if total > 0 else 0.0,
        }

    return stats
