"""PRIMATE dataset extraction for PHQ-9 few-shot examples.

Wraps the logic from specs/MentalRiskES/extract_primate_examples.py
as an importable module for the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

PHQ9_ITEMS = {
    0: {
        "name": "anhedonia",
        "phq_text": "Little interest or pleasure in doing things",
        "keywords": [
            "interest", "pleasure", "enjoy", "hobby", "fun", "bored",
            "motivation", "care", "apathy", "nothing matters", "don't want to",
            "used to love", "stopped doing", "can't be bothered", "pointless",
        ],
    },
    1: {
        "name": "depressed_mood",
        "phq_text": "Feeling down, depressed, or hopeless",
        "keywords": [
            "depressed", "hopeless", "sad", "down", "empty", "crying",
            "miserable", "despair", "numb", "hollow", "darkness", "low",
            "unhappy", "grief", "sorrow", "bleak",
        ],
    },
    2: {
        "name": "sleep",
        "phq_text": "Trouble falling or staying asleep, or sleeping too much",
        "keywords": [
            "sleep", "insomnia", "bed", "wake", "nap", "oversleep",
            "can't sleep", "toss and turn", "nightmare", "3am", "4am",
            "all night", "sleeping all day",
        ],
    },
    3: {
        "name": "fatigue",
        "phq_text": "Feeling tired or having little energy",
        "keywords": [
            "energy", "tired", "exhausted", "fatigue", "drained", "lazy",
            "no energy", "wiped out", "sluggish", "lethargic", "worn out",
            "can barely", "effort", "drag myself",
        ],
    },
    4: {
        "name": "appetite",
        "phq_text": "Poor appetite or overeating",
        "keywords": [
            "appetite", "eat", "food", "hungry", "weight", "meal",
            "starving", "overeating", "binge", "not eating", "lost weight",
            "gained weight", "can't eat", "comfort food",
        ],
    },
    5: {
        "name": "self_worth",
        "phq_text": "Feeling bad about yourself - or that you are a failure",
        "keywords": [
            "failure", "worthless", "guilty", "blame", "hate myself",
            "disappointment", "useless", "burden", "shame", "loser",
            "not good enough", "let everyone down", "pathetic",
        ],
    },
    6: {
        "name": "concentration",
        "phq_text": "Trouble concentrating on things",
        "keywords": [
            "concentrate", "focus", "attention", "distract", "think",
            "remember", "brain fog", "can't think", "mind wanders",
            "zoning out", "spacing out", "forgetful",
        ],
    },
    7: {
        "name": "psychomotor",
        "phq_text": "Moving or speaking slowly, or being restless/fidgety",
        "keywords": [
            "slow", "restless", "fidget", "agitated", "moving", "pace",
            "can't sit still", "jittery", "nervous energy", "sluggish",
            "frozen", "paralyzed", "can't move",
        ],
    },
    8: {
        "name": "suicidality",
        "phq_text": "Thoughts that you would be better off dead or of hurting yourself",
        "keywords": [
            "dead", "die", "kill", "suicide", "hurt myself", "end it",
            "better off dead", "not alive", "want to disappear",
            "self-harm", "cutting", "no point living",
        ],
    },
}


def _get_text(post: dict) -> str:
    title = post.get("post_title", "")
    text = post.get("post_text", "")
    return f"{title} {text}".strip()


def _get_labels(post: dict) -> dict[int, str]:
    """Extract PHQ-9 labels as {symptom_index: 'yes'/'no'}."""
    annotations = post.get("annotations", [])
    label_map = {}
    kw_map = {
        0: "interest", 1: "down", 2: "sleep", 3: "tired",
        4: "appetite", 5: "bad-about", 6: "concentrating",
        7: "moving", 8: "dead",
    }
    for annotation in annotations:
        if len(annotation) >= 2:
            q = annotation[0].lower().replace("-", " ")
            label = annotation[1].lower()
            for idx, keyword in kw_map.items():
                if keyword.lower() in q:
                    label_map[idx] = label
                    break
    return label_map


def _keyword_score(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def extract_primate_examples(
    posts: list[dict],
    max_per_symptom: int = 3,
) -> dict:
    """
    Extract few-shot examples per PHQ-9 symptom from PRIMATE dataset.

    Returns dict with per-symptom positive and negative examples.
    """
    selected_ids: set[int] = set()
    examples = {}

    for symptom_idx in range(9):
        info = PHQ9_ITEMS[symptom_idx]
        positives = []
        negatives = []

        for i, post in enumerate(posts):
            labels = _get_labels(post)
            if symptom_idx in labels:
                text = _get_text(post)
                wc = len(text.split())
                if wc < 50 or wc > 250:
                    continue
                if labels[symptom_idx] == "yes":
                    # Score: keyword clarity + exclusivity
                    kw = _keyword_score(text, info["keywords"])
                    other_pos = sum(1 for idx, l in labels.items() if idx != symptom_idx and l == "yes")
                    score = kw * 2.0 + 1.0 / (1 + other_pos) * 3.0
                    positives.append((score, i, post))
                else:
                    other_pos = sum(1 for idx, l in labels.items() if idx != symptom_idx and l == "yes")
                    score = other_pos * 2.0
                    negatives.append((score, i, post))

        positives.sort(reverse=True)
        negatives.sort(reverse=True)

        pos_examples = []
        for score, idx, post in positives:
            if idx not in selected_ids and len(pos_examples) < max_per_symptom:
                text = _get_text(post)
                pos_examples.append({
                    "text": text,
                    "word_count": len(text.split()),
                    "score": round(score, 2),
                    "all_labels": {PHQ9_ITEMS[k]["name"]: v for k, v in _get_labels(post).items()},
                })
                selected_ids.add(idx)

        neg_examples = []
        for score, idx, post in negatives:
            if idx not in selected_ids and len(neg_examples) < max_per_symptom:
                text = _get_text(post)
                neg_examples.append({
                    "text": text,
                    "word_count": len(text.split()),
                    "score": round(score, 2),
                })
                selected_ids.add(idx)

        examples[f"item_{symptom_idx + 1}_{info['name']}"] = {
            "phq_text": info["phq_text"],
            "total_positive": len(positives),
            "total_negative": len(negatives),
            "positive_examples": pos_examples,
            "negative_examples": neg_examples,
        }

    return examples


def compute_prevalence(posts: list[dict]) -> dict[str, dict]:
    """Compute per-symptom prevalence rates."""
    counts: Counter[int] = Counter()
    total = len(posts)
    for post in posts:
        for idx, label in _get_labels(post).items():
            if label == "yes":
                counts[idx] += 1
    return {
        PHQ9_ITEMS[idx]["name"]: {
            "count": counts[idx],
            "prevalence": round(counts[idx] / total, 4) if total > 0 else 0,
        }
        for idx in range(9)
    }


def compute_cooccurrence(posts: list[dict]) -> list[list[int]]:
    """Compute symptom co-occurrence matrix."""
    matrix = [[0] * 9 for _ in range(9)]
    for post in posts:
        labels = _get_labels(post)
        pos = [idx for idx, l in labels.items() if l == "yes" and idx < 9]
        for i in pos:
            for j in pos:
                matrix[i][j] += 1
    return matrix
