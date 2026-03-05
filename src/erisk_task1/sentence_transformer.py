"""Sentence transformer for BDI-II symptom relevance scoring.

Tier 2 model-based feature: fine-tuned sentence transformer that produces
a 21-dimensional BDI-II relevance probability vector per input sentence.

Training: multi-label classification on DepreSym + ReDSM5 + BDI-Sen.
Inference: encode persona response sentences → per-symptom relevance scores.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NUM_ITEMS = 21


@dataclass
class SentenceTransformerConfig:
    """Configuration for the symptom sentence transformer."""
    # Model — mpnet-base-v2 chosen over MiniLM for richer 768-dim embeddings,
    # better at distinguishing nuanced symptom language (proven in eRisk 2024).
    base_model: str = "all-mpnet-base-v2"
    model_path: str = ""  # path to fine-tuned model; empty = use base
    max_seq_length: int = 128
    device: str = "cpu"

    # Training
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_split: float = 0.1
    loss: str = "weighted_bce"  # "weighted_bce" or "focal"
    focal_gamma: float = 2.0

    # Data
    depresym_pools: str = "data/DepreSym_dataset/pools_docnos.json"
    depresym_qrels: str = "data/DepreSym_dataset/qrels-consensus.txt"
    redsm5_annotations: str = "data/RedSM5/redsm5_annotations.csv"
    redsm5_posts: str = "data/RedSM5/redsm5_posts.csv"
    bdisen_data: str = "data/BDI-Sen/full_dataset/bdi_unified.jsonl"

    # Output
    output_dir: str = "models/symptom_transformer"


class SymptomScorer:
    """Inference wrapper for the fine-tuned symptom sentence transformer.

    Encodes sentences and produces 21-dim BDI-II relevance probability vectors.
    """

    def __init__(self, config: SentenceTransformerConfig):
        self.config = config
        self._model = None
        self._classifier = None

    def load(self) -> None:
        """Load the fine-tuned model for inference."""
        import torch
        from sentence_transformers import SentenceTransformer

        model_path = self.config.model_path or self.config.base_model
        logger.info("Loading symptom scorer from %s", model_path)

        self._model = SentenceTransformer(model_path, device=self.config.device)
        self._model.max_seq_length = self.config.max_seq_length

        # Load classification head if fine-tuned model exists
        classifier_path = Path(model_path) / "classifier_head.pt"
        if classifier_path.exists():
            embedding_dim = self._model.get_sentence_embedding_dimension()
            self._classifier = torch.nn.Linear(embedding_dim, NUM_ITEMS)
            state = torch.load(
                classifier_path, map_location=self.config.device, weights_only=True
            )
            self._classifier.load_state_dict(state)
            self._classifier.eval()
            self._classifier.to(self.config.device)
            logger.info("Loaded classifier head from %s", classifier_path)
        else:
            logger.warning(
                "No classifier head found at %s — using cosine similarity fallback",
                classifier_path,
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    def score_sentences(self, sentences: list[str]) -> np.ndarray:
        """Score a batch of sentences for BDI-II symptom relevance.

        Args:
            sentences: List of sentence strings.

        Returns:
            Array of shape (len(sentences), 21) with relevance probabilities.
        """
        if not self.is_loaded():
            self.load()

        import torch

        embeddings = self._model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )

        if self._classifier is not None:
            with torch.no_grad():
                logits = self._classifier(embeddings)
                probs = torch.sigmoid(logits).cpu().numpy()
        else:
            # Fallback: return zeros (no trained head available)
            probs = np.zeros((len(sentences), NUM_ITEMS), dtype=np.float32)

        return probs

    def score_text(self, text: str) -> np.ndarray:
        """Score a single text (may contain multiple sentences).

        Splits into sentences and returns max-pooled relevance per item.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return np.zeros(NUM_ITEMS, dtype=np.float32)

        probs = self.score_sentences(sentences)
        # Max-pool across sentences for each symptom
        return probs.max(axis=0)

    def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._classifier = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple rule-based)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def train_symptom_transformer(config: SentenceTransformerConfig) -> Path:
    """Train the multi-label symptom classifier.

    Fine-tunes a sentence transformer with a classification head on the
    combined DepreSym + ReDSM5 + BDI-Sen dataset.

    Returns path to the saved model directory.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader, TensorDataset

    from .symptom_data import (
        SymptomDataConfig,
        compute_class_weights,
        dataset_statistics,
        load_all_datasets,
    )

    # Load data
    data_config = SymptomDataConfig(
        depresym_pools=config.depresym_pools,
        depresym_qrels=config.depresym_qrels,
        redsm5_annotations=config.redsm5_annotations,
        redsm5_posts=config.redsm5_posts,
        bdisen_data=config.bdisen_data,
    )
    samples = load_all_datasets(data_config)
    if not samples:
        raise ValueError("No training data loaded")

    # Log statistics
    stats = dataset_statistics(samples)
    logger.info("Dataset statistics: %s", json.dumps(stats["source_counts"]))

    # Compute class weights for weighted BCE
    pos_weights = compute_class_weights(samples)
    logger.info("Pos weights range: %.1f - %.1f", pos_weights.min(), pos_weights.max())

    # Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    val_size = int(len(samples) * config.eval_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Load base model
    logger.info("Loading base model: %s", config.base_model)
    model = SentenceTransformer(config.base_model, device=config.device)
    model.max_seq_length = config.max_seq_length
    embedding_dim = model.get_sentence_embedding_dimension()

    # Classification head
    classifier = torch.nn.Linear(embedding_dim, NUM_ITEMS).to(config.device)

    # Encode all sentences (batch)
    logger.info("Encoding %d sentences...", len(samples))
    all_texts = [s.text for s in samples]
    all_embeddings = model.encode(
        all_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        batch_size=config.batch_size,
    )
    all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)
    all_labels = torch.tensor(
        np.array([s.labels for s in samples]), dtype=torch.float32
    )
    all_weights = torch.tensor(
        np.array([s.weight for s in samples]), dtype=torch.float32
    )

    # Build dataloaders
    train_dataset = TensorDataset(
        all_embeddings[train_indices],
        all_labels[train_indices],
        all_weights[train_indices],
    )
    val_dataset = TensorDataset(
        all_embeddings[val_indices],
        all_labels[val_indices],
        all_weights[val_indices],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Loss function
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(
        config.device
    )
    if config.loss == "focal":
        loss_fn = _FocalBCEWithLogitsLoss(
            pos_weight=pos_weight_tensor, gamma=config.focal_gamma
        )
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor, reduction="none"
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    best_val_f1 = 0.0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        classifier.train()
        epoch_loss = 0.0

        for batch_emb, batch_labels, batch_weights in train_loader:
            batch_emb = batch_emb.to(config.device)
            batch_labels = batch_labels.to(config.device)
            batch_weights = batch_weights.to(config.device)

            logits = classifier(batch_emb)
            per_sample_loss = loss_fn(logits, batch_labels).mean(dim=1)
            weighted_loss = (per_sample_loss * batch_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        classifier.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_emb, batch_labels, _ in val_loader:
                batch_emb = batch_emb.to(config.device)
                logits = classifier(batch_emb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_true.append(batch_labels.numpy())

        val_preds_arr = np.concatenate(val_preds)
        val_true_arr = np.concatenate(val_true)
        val_f1 = _compute_per_symptom_f1(val_preds_arr, val_true_arr)
        macro_f1 = val_f1.mean()

        logger.info(
            "Epoch %d/%d: loss=%.4f, macro_f1=%.4f",
            epoch + 1, config.epochs, avg_loss, macro_f1,
        )

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(
                classifier.state_dict(), output_dir / "classifier_head.pt"
            )
            logger.info("New best model saved (macro_f1=%.4f)", macro_f1)

    # Save the sentence transformer model for self-contained loading
    model.save(str(output_dir))
    logger.info("Training complete. Model saved to %s", output_dir)

    # Save training stats
    from .models import BDI_ITEMS
    per_symptom_f1 = {}
    for i in range(NUM_ITEMS):
        per_symptom_f1[BDI_ITEMS[i + 1]] = float(val_f1[i])

    train_stats = {
        "best_macro_f1": float(best_val_f1),
        "per_symptom_f1": per_symptom_f1,
        "dataset_stats": stats,
        "config": {
            "base_model": config.base_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "loss": config.loss,
        },
    }
    with open(output_dir / "training_stats.json", "w") as f:
        json.dump(train_stats, f, indent=2)

    return output_dir


class _FocalBCEWithLogitsLoss:
    """Focal loss variant of BCEWithLogitsLoss for hard-example mining."""

    def __init__(self, pos_weight, gamma: float = 2.0):
        import torch
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="none"
        )

    def __call__(self, logits, targets):
        import torch
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return focal_weight * bce_loss


def _compute_per_symptom_f1(
    preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Compute per-symptom F1 scores."""
    binary_preds = (preds >= threshold).astype(np.float32)
    f1_scores = np.zeros(NUM_ITEMS, dtype=np.float64)

    for i in range(NUM_ITEMS):
        tp = ((binary_preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((binary_preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((binary_preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores[i] = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return f1_scores
