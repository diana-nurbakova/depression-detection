"""Bi-encoder retrieval using SentenceTransformer models.

Encodes the corpus once and caches embeddings to disk as .npy files.
Computes cosine similarity per query for candidate retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hipert.models import Sentence

logger = logging.getLogger(__name__)


class BiEncoderRetriever:
    """Bi-encoder retrieval using SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        batch_size: int = 256,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._docnos: list[str] = []

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_corpus(
        self,
        sentences: list[Sentence],
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Encode all sentences and cache embeddings to disk.

        Args:
            sentences: List of Sentence objects to encode.
            force_recompute: If True, ignore cached embeddings.

        Returns:
            Numpy array of shape (n_sentences, embedding_dim).
        """
        safe_name = self.model_name.replace("/", "_")
        embeddings_path = self.cache_dir / f"embeddings_{safe_name}.npy"
        docnos_path = self.cache_dir / f"docnos_{safe_name}.npy"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if (
            not force_recompute
            and embeddings_path.exists()
            and docnos_path.exists()
        ):
            logger.info("Loading cached embeddings from %s", embeddings_path)
            self._embeddings = np.load(str(embeddings_path))
            self._docnos = np.load(str(docnos_path), allow_pickle=True).tolist()
            logger.info(
                "Loaded %d embeddings (dim=%d)",
                self._embeddings.shape[0], self._embeddings.shape[1],
            )
            return self._embeddings

        logger.info(
            "Encoding %d sentences with %s (batch_size=%d)",
            len(sentences), self.model_name, self.batch_size,
        )

        # Extract text for encoding
        texts = [s.text for s in sentences]
        self._docnos = [s.docno for s in sentences]

        # Encode in batches with progress bar
        self._embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )

        # Cache to disk
        np.save(str(embeddings_path), self._embeddings)
        np.save(str(docnos_path), np.array(self._docnos, dtype=object))

        logger.info(
            "Encoded and cached: shape=%s, saved to %s",
            self._embeddings.shape, embeddings_path,
        )
        return self._embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 5000,
    ) -> list[tuple[int, float]]:
        """Retrieve top-K most similar sentences for a query.

        Args:
            query: Query string to search for.
            top_k: Number of results to return.

        Returns:
            List of (sentence_index, cosine_similarity_score) tuples,
            sorted by descending similarity.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Corpus not encoded yet. Call encode_corpus() first.",
            )

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        # Cosine similarity via dot product (embeddings are normalized)
        scores = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-K indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]

    def get_docno(self, index: int) -> str:
        """Get the DOCNO for a given sentence index."""
        return self._docnos[index]

    @property
    def is_loaded(self) -> bool:
        return self._embeddings is not None
