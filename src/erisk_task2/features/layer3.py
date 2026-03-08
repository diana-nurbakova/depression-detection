"""Layer 3: Emotion and Topic Distributions (Spec Sections 4.3, 7).

- Emotion classification (Plutchik's 8 emotions)
- BERTopic dynamic topic modeling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Plutchik's 8 primary emotions
PLUTCHIK_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]
NUM_EMOTIONS = 8


class EmotionClassifier:
    """Classify text into Plutchik's 8 primary emotions."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline
        logger.info("Loading emotion classifier: %s", self.model_name)
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            truncation=True,
            max_length=512,
            device=self.device if self.device != "cpu" else -1,
        )

    def classify(self, texts: list[str], min_words: int = 10) -> list[np.ndarray]:
        """Classify texts into 8-dim emotion distributions.

        Texts shorter than min_words get uniform distribution.
        Returns list of (8,) arrays.
        """
        self.load()
        results = []
        for text in texts:
            word_count = len(text.split())
            if word_count < min_words:
                results.append(np.ones(NUM_EMOTIONS) / NUM_EMOTIONS)
                continue

            preds = self._pipeline(text)[0]
            dist = np.zeros(NUM_EMOTIONS)
            for p in preds:
                label = p["label"].lower()
                # Map model labels to Plutchik
                for i, emotion in enumerate(PLUTCHIK_EMOTIONS):
                    if emotion in label:
                        dist[i] = p["score"]
                        break
            # Normalize
            total = dist.sum()
            if total > 0:
                dist = dist / total
            else:
                dist = np.ones(NUM_EMOTIONS) / NUM_EMOTIONS
            results.append(dist)

        return results

    def aggregate_window(self, distributions: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Aggregate emotion distributions over a window.

        Returns (mean_distribution (8d), entropy).
        """
        if not distributions:
            return np.ones(NUM_EMOTIONS) / NUM_EMOTIONS, np.log(NUM_EMOTIONS)

        arr = np.stack(distributions)
        mean_dist = arr.mean(axis=0)
        total = mean_dist.sum()
        if total > 0:
            mean_dist = mean_dist / total

        # Shannon entropy
        entropy = -np.sum(mean_dist * np.log(mean_dist + 1e-10))
        return mean_dist, float(entropy)


class TopicModeler:
    """BERTopic wrapper for depression topic modeling."""

    def __init__(
        self,
        n_topics: int = 40,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_cluster_size: int = 50,
        min_samples: int = 10,
    ):
        self.n_topics = n_topics
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = None
        self.depression_topic_ids: set[int] = set()
        self._n_topics_actual: int = 0

    def fit(self, documents: list[str], depression_labels: Optional[list[bool]] = None):
        """Fit BERTopic on training documents.

        Args:
            documents: list of concatenated text chunks
            depression_labels: optional, to identify depression-related topics post-hoc
        """
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

        logger.info("Fitting BERTopic on %d documents", len(documents))

        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            prediction_data=True,
        )
        embedding_model = SentenceTransformer("all-mpnet-base-v2")

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=self.n_topics,
            verbose=True,
        )
        topics, probs = self.model.fit_transform(documents)
        self._n_topics_actual = len(set(topics)) - (1 if -1 in topics else 0)
        logger.info("BERTopic found %d topics", self._n_topics_actual)

        # Label depression-relevant topics
        if depression_labels is not None:
            self._label_depression_topics(topics, depression_labels)

    def _label_depression_topics(self, topics: list[int], depression_labels: list[bool]):
        """Identify topics over-represented in depressed users' texts."""
        topic_dep_counts: dict[int, int] = {}
        topic_total_counts: dict[int, int] = {}

        for topic, is_dep in zip(topics, depression_labels):
            if topic == -1:
                continue
            topic_total_counts[topic] = topic_total_counts.get(topic, 0) + 1
            if is_dep:
                topic_dep_counts[topic] = topic_dep_counts.get(topic, 0) + 1

        base_rate = sum(depression_labels) / max(len(depression_labels), 1)
        self.depression_topic_ids = set()

        for topic_id, total in topic_total_counts.items():
            dep_count = topic_dep_counts.get(topic_id, 0)
            if total > 10 and (dep_count / total) > base_rate * 1.5:
                self.depression_topic_ids.add(topic_id)

        logger.info(
            "Identified %d depression-related topics out of %d",
            len(self.depression_topic_ids), len(topic_total_counts),
        )

    def transform(self, text: str) -> tuple[np.ndarray, float, float]:
        """Get topic distribution for a text.

        Returns:
            (topic_distribution (n_topics,), topic_entropy, depression_topic_proportion)
        """
        if self.model is None:
            n = max(self.n_topics, 1)
            return np.zeros(n), 0.0, 0.0

        topics, probs = self.model.transform([text])

        # Get topic distribution
        if probs is not None and len(probs) > 0:
            dist = np.array(probs[0]) if hasattr(probs[0], '__len__') else np.zeros(self._n_topics_actual)
        else:
            dist = np.zeros(self._n_topics_actual)

        # Ensure valid distribution
        if dist.sum() > 0:
            dist = dist / dist.sum()
        else:
            dist = np.ones_like(dist) / max(len(dist), 1)

        # Entropy
        entropy = -np.sum(dist * np.log(dist + 1e-10))

        # Depression topic proportion
        dep_proportion = 0.0
        if self.depression_topic_ids and topics[0] != -1:
            dep_proportion = float(topics[0] in self.depression_topic_ids)

        return dist, float(entropy), dep_proportion

    def save(self, path: str | Path):
        if self.model:
            self.model.save(str(path))

    def load(self, path: str | Path):
        from bertopic import BERTopic
        self.model = BERTopic.load(str(path))
