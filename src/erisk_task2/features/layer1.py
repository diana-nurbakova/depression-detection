"""Layer 1: Target User Textual Features (Spec Section 4.1).

- Sentence transformer ensemble embeddings (1920d)
- BDI-II symptom intensity scores (21d × multiple aggregations)
- Lexical indicators (4d)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _patch_st_device(model, device: str):
    """Add .device attribute to SentenceTransformer if missing (v3+ compat)."""
    import torch
    if not hasattr(model, "device"):
        model.device = torch.device(device)

# Lexical word lists
_FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}

_NEGATIVE_EMOTION = {
    "sad", "sadness", "unhappy", "depressed", "miserable", "hopeless", "worthless",
    "lonely", "alone", "empty", "numb", "anxious", "worried", "afraid", "scared",
    "angry", "frustrated", "irritated", "guilty", "ashamed", "hurt", "pain",
    "suffering", "terrible", "awful", "horrible", "hate", "despair", "cry",
    "crying", "tears", "grief", "sorrow", "broken", "lost", "helpless",
}

_ABSOLUTIST = {
    "always", "never", "nothing", "everything", "everyone", "nobody",
    "completely", "totally", "absolutely", "entirely", "constantly",
    "definitely", "certainly", "impossible", "forever",
}

_COGNITIVE_PROCESS = {
    "think", "know", "believe", "understand", "realize", "remember",
    "forget", "wonder", "suppose", "consider", "decide", "feel",
    "guess", "imagine", "mean", "notice", "figure", "reason",
}

# Default BDI-II symptom descriptions for reference embeddings (Variant C)
SYMPTOM_DESCRIPTIONS = {
    1: "Persistent feelings of sorrow, unhappiness, or emotional pain.",
    2: "Discouragement and hopelessness about the future.",
    3: "Feeling like a failure, seeing many disappointments.",
    4: "Reduced enjoyment of activities, hobbies, social life.",
    5: "Excessive guilt, self-blame for bad things.",
    6: "Expectation of punishment, sense that bad things are deserved.",
    7: "Self-criticism, disappointment in oneself as a person.",
    8: "Harsh self-judgment for all faults and mistakes.",
    9: "Thoughts of ending one's life, death wishes.",
    10: "Increased tearfulness, uncontrollable emotional outbursts.",
    11: "Restlessness, irritability, inability to stay still or relax.",
    12: "Social withdrawal, apathy, not caring about things.",
    13: "Difficulty making decisions, putting off choices.",
    14: "Profound sense of having no value, being useless.",
    15: "Fatigue, everything takes extra effort.",
    16: "Insomnia, oversleeping, or disrupted sleep patterns.",
    17: "Short temper, easily frustrated or angered.",
    18: "Eating much more or less, weight gain or loss.",
    19: "Brain fog, difficulty focusing, forgetfulness.",
    20: "Constant exhaustion, lack of motivation due to tiredness.",
    21: "Reduced libido, no sexual desire.",
}


class EmbeddingEncoder:
    """Manages the 3-model sentence transformer ensemble."""

    def __init__(
        self,
        model_names: list[str] | None = None,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_names = model_names or [
            "all-mpnet-base-v2",
            "all-MiniLM-L12-v2",
            "all-distilroberta-v1",
        ]
        self.device = device
        self.batch_size = batch_size
        self.models = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from sentence_transformers import SentenceTransformer
        for name in self.model_names:
            logger.info("Loading sentence transformer: %s", name)
            model = SentenceTransformer(name, device=self.device)
            _patch_st_device(model, self.device)
            self.models.append(model)
        self._loaded = True

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with all 3 models, return concatenated embeddings.

        Returns shape (n_texts, total_dim) where total_dim = sum of model dims.
        """
        self.load()
        if not texts:
            total_dim = sum(m.get_sentence_embedding_dimension() for m in self.models)
            return np.zeros((0, total_dim))

        embeddings = []
        for model in self.models:
            emb = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embeddings.append(emb)

        return np.concatenate(embeddings, axis=1)

    @property
    def total_dim(self) -> int:
        if self._loaded:
            return sum(m.get_sentence_embedding_dimension() for m in self.models)
        return 1920  # default: 768 + 384 + 768


class SymptomScorer:
    """Compute BDI-II symptom activation scores via cosine similarity."""

    def __init__(self, encoder: EmbeddingEncoder):
        self.encoder = encoder
        self.reference_embeddings: Optional[np.ndarray] = None  # (21, dim)

    def build_references(self, descriptions: dict[int, str] | None = None):
        """Compute reference embeddings for all 21 symptoms."""
        descs = descriptions or SYMPTOM_DESCRIPTIONS
        texts = [descs[i] for i in range(1, 22)]
        self.reference_embeddings = self.encoder.encode(texts)
        # Normalize
        norms = np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.reference_embeddings = self.reference_embeddings / norms

    def score(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embedding and 21 symptom references.

        Args:
            embedding: (dim,) user embedding vector

        Returns:
            (21,) symptom activation scores in [-1, 1]
        """
        if self.reference_embeddings is None:
            self.build_references()
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return np.zeros(21)
        normed = embedding / norm
        return self.reference_embeddings @ normed


def compute_lexical_features(texts: list[str]) -> np.ndarray:
    """Compute 4 lexical indicator ratios from accumulated texts.

    Returns: [fps_ratio, neg_emo_ratio, absolutist_ratio, cognitive_ratio]
    """
    if not texts:
        return np.zeros(4)

    all_text = " ".join(texts).lower()
    words = re.findall(r'\b\w+\b', all_text)
    n_words = len(words)

    if n_words == 0:
        return np.zeros(4)

    word_set = set(words)
    fps_count = sum(1 for w in words if w in _FIRST_PERSON_SINGULAR)
    neg_count = sum(1 for w in words if w in _NEGATIVE_EMOTION)
    abs_count = sum(1 for w in words if w in _ABSOLUTIST)
    cog_count = sum(1 for w in words if w in _COGNITIVE_PROCESS)

    return np.array([
        fps_count / n_words,
        neg_count / n_words,
        abs_count / n_words,
        cog_count / n_words,
    ])


def compute_symptom_distribution_stats(activations: list[np.ndarray]) -> np.ndarray:
    """Compute 7 distributional statistics per symptom from activation history.

    Args:
        activations: list of (21,) arrays, one per round

    Returns:
        (147,) array: 21 symptoms × 7 stats (mean, var, skew, kurtosis, Q25, Q50, Q75)
    """
    if not activations:
        return np.zeros(147)

    arr = np.stack(activations)  # (n_rounds, 21)
    stats = []

    for s in range(21):
        col = arr[:, s]
        n = len(col)
        mean = np.mean(col)
        var = np.var(col)

        if n > 2 and var > 1e-10:
            std = np.sqrt(var)
            skew = float(np.mean(((col - mean) / std) ** 3))
            kurt = float(np.mean(((col - mean) / std) ** 4)) - 3.0
        else:
            skew = 0.0
            kurt = 0.0

        q25, q50, q75 = np.percentile(col, [25, 50, 75])
        stats.extend([mean, var, skew, kurt, q25, q50, q75])

    return np.array(stats)


def update_embedding_running_mean(
    current_sum: Optional[np.ndarray],
    current_weight: float,
    new_embedding: np.ndarray,
    decay_lambda: float = 0.95,
) -> tuple[np.ndarray, float]:
    """Update exponential-decay weighted running mean.

    Returns (new_sum, new_weight).
    """
    if current_sum is None:
        return new_embedding.copy(), 1.0
    decayed_sum = current_sum * decay_lambda
    decayed_weight = current_weight * decay_lambda
    return decayed_sum + new_embedding, decayed_weight + 1.0


def get_running_mean(sum_vec: Optional[np.ndarray], weight: float) -> np.ndarray:
    """Get current running mean from sum and weight."""
    if sum_vec is None or weight < 1e-8:
        return np.zeros(1920)
    return sum_vec / weight
