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
        checkpoint_every: int = 500,
    ) -> np.ndarray:
        """Encode all sentences and cache embeddings to disk.

        Supports resuming from intermediate checkpoints so that progress
        is not lost if the process is interrupted.

        Args:
            sentences: List of Sentence objects to encode.
            force_recompute: If True, ignore cached embeddings.
            checkpoint_every: Save intermediate checkpoint every N batches.

        Returns:
            Numpy array of shape (n_sentences, embedding_dim).
        """
        safe_name = self.model_name.replace("/", "_")
        embeddings_path = self.cache_dir / f"embeddings_{safe_name}.npy"
        docnos_path = self.cache_dir / f"docnos_{safe_name}.npy"
        chunks_dir = self.cache_dir / f"chunks_{safe_name}"
        progress_path = self.cache_dir / f"embeddings_{safe_name}.progress"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Check for completed cache first
        if (
            not force_recompute
            and embeddings_path.exists()
            and docnos_path.exists()
        ):
            logger.info("Loading cached embeddings from %s", embeddings_path)
            self._embeddings = np.load(str(embeddings_path), mmap_mode="r")
            self._docnos = np.load(str(docnos_path), allow_pickle=True).tolist()
            logger.info(
                "Loaded %d embeddings (dim=%d)",
                self._embeddings.shape[0], self._embeddings.shape[1],
            )
            return self._embeddings

        # Extract text and docnos
        texts = [s.text for s in sentences]
        self._docnos = [s.docno for s in sentences]
        total = len(texts)
        chunk_size = self.batch_size * checkpoint_every

        # Check for partial checkpoint to resume from
        completed_chunks = 0
        chunks_dir.mkdir(parents=True, exist_ok=True)

        if (
            not force_recompute
            and progress_path.exists()
        ):
            completed_chunks = int(progress_path.read_text().strip())
            # Verify all chunk files exist
            for i in range(completed_chunks):
                chunk_path = chunks_dir / f"chunk_{i:04d}.npy"
                if not chunk_path.exists():
                    logger.warning(
                        "Missing chunk file %s, restarting from scratch",
                        chunk_path,
                    )
                    completed_chunks = 0
                    break

        start_idx = completed_chunks * chunk_size

        if start_idx > 0 and start_idx < total:
            logger.info(
                "Resuming encoding from sentence %d/%d (%.1f%% done, %d chunks cached)",
                start_idx, total, 100 * start_idx / total, completed_chunks,
            )
        elif start_idx == 0:
            logger.info(
                "Encoding %d sentences with %s (batch_size=%d, checkpoint every %d batches)",
                total, self.model_name, self.batch_size, checkpoint_every,
            )

        # Encode remaining chunks
        remaining_texts = texts[start_idx:]
        n_remaining = len(remaining_texts)
        total_chunks = (total + chunk_size - 1) // chunk_size
        current_chunk = completed_chunks

        if n_remaining > 0:
            for chunk_start in tqdm(
                range(0, n_remaining, chunk_size),
                desc="Encoding chunks",
                unit="chunk",
                total=(n_remaining + chunk_size - 1) // chunk_size,
            ):
                chunk_end = min(chunk_start + chunk_size, n_remaining)
                chunk_texts = remaining_texts[chunk_start:chunk_end]

                chunk_embeddings = self.model.encode(
                    chunk_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )

                # Save this chunk to its own file
                chunk_path = chunks_dir / f"chunk_{current_chunk:04d}.npy"
                np.save(str(chunk_path), chunk_embeddings)
                current_chunk += 1

                # Update progress
                progress_path.write_text(str(current_chunk))
                encoded_so_far = min(current_chunk * chunk_size, total)
                logger.info(
                    "Chunk %d/%d saved (%d/%d sentences, %.1f%%)",
                    current_chunk, total_chunks,
                    encoded_so_far, total, 100 * encoded_so_far / total,
                )

        # Assemble chunks directly into a memory-mapped .npy file
        # (avoids Windows 2 GB single-write limit and keeps RAM low)
        logger.info("Assembling %d chunks into final embeddings...", current_chunk)
        first_chunk = np.load(str(chunks_dir / "chunk_0000.npy"))
        embed_dim = first_chunk.shape[1]

        mmap_out = np.lib.format.open_memmap(
            str(embeddings_path), mode="w+",
            dtype=np.float32, shape=(total, embed_dim),
        )
        mmap_out[: first_chunk.shape[0]] = first_chunk
        del first_chunk
        offset = chunk_size
        for i in range(1, current_chunk):
            chunk = np.load(str(chunks_dir / f"chunk_{i:04d}.npy"))
            end = offset + chunk.shape[0]
            mmap_out[offset:end] = chunk
            del chunk
            offset = end
        mmap_out.flush()
        del mmap_out

        # Reload as regular array for in-memory queries
        self._embeddings = np.load(str(embeddings_path), mmap_mode="r")
        np.save(str(docnos_path), np.array(self._docnos, dtype=object))

        # Clean up chunk files and progress
        import shutil
        shutil.rmtree(str(chunks_dir), ignore_errors=True)
        if progress_path.exists():
            progress_path.unlink()

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
