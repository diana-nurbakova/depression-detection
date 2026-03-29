"""Cross-encoder reranker for HiPerT v2.

Architecture (replaces bi-encoder from v1):
    Input: [CLS] <symptom_text> [SEP] <sentence_text> [SEP]
    Output: CORAL ordinal logits (3 thresholds) or ListMLE scalar score

Key advantage over bi-encoder: cross-attention between symptom and sentence
enables valence/negation detection that bi-encoders miss.

Spec reference: hipert_v2_spec.md Section 3
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Pre-configured backbones (same as v1)
BACKBONES = {
    "mental-roberta": "mental/mental-roberta-base",
    "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


class CrossEncoderReranker(nn.Module):
    """Cross-encoder reranker for symptom-sentence relevance scoring.

    Processes symptom and sentence jointly via cross-attention:
        φ(s,q) = head(enc([CLS] q [SEP] s [SEP]))

    Supports two head types:
        - 'coral': K=3 ordinal threshold logits → E[score] ∈ [0, 3]
        - 'listmle': K=1 scalar relevance score
    """

    def __init__(
        self,
        backbone_name: str = "mpnet",
        head_type: str = "coral",
        num_thresholds: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_unfrozen: int = 4,
    ):
        super().__init__()

        model_id = BACKBONES.get(backbone_name, backbone_name)
        self.encoder = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.head_type = head_type
        self.backbone_name = backbone_name

        enc_dim = self.encoder.config.hidden_size  # 768

        out_dim = num_thresholds if head_type == "coral" else 1

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

        # Freeze bottom layers
        self._freeze_bottom_layers(num_unfrozen)

    def _freeze_bottom_layers(self, num_unfrozen: int = 4) -> None:
        """Freeze all encoder layers except top `num_unfrozen`."""
        # Get encoder layers
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            all_layers = list(self.encoder.encoder.layer)
        elif hasattr(self.encoder, "layers"):
            all_layers = list(self.encoder.layers)
        else:
            logger.warning("Cannot identify backbone layers for freezing")
            return

        n_freeze = max(0, len(all_layers) - num_unfrozen)
        for layer in all_layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        # Always freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Froze %d/%d layers. Trainable: %d/%d params (%.1f%%)",
            n_freeze, len(all_layers), trainable, total,
            100 * trainable / total,
        )

    def unfreeze_layers(self, num_unfrozen: int) -> None:
        """Adjust the number of unfrozen layers."""
        # Unfreeze all first
        for param in self.parameters():
            param.requires_grad = True
        # Then re-freeze
        self._freeze_bottom_layers(num_unfrozen)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) — [CLS] symptom [SEP] sentence [SEP]
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) — 0 for symptom, 1 for sentence

        Returns:
            If CORAL: (batch, 3) logits for ordinal thresholds
            If ListMLE: (batch, 1) scalar relevance scores
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some models (e.g., mpnet) don't use token_type_ids
        if token_type_ids is not None and hasattr(self.encoder.config, "type_vocab_size"):
            if self.encoder.config.type_vocab_size > 1:
                kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.head(cls_hidden)
        return logits

    def predict_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce a scalar ranking score for inference.

        CORAL: E[score] = σ(f₁) + σ(f₂) + σ(f₃) ∈ [0, 3]
        ListMLE: raw scalar output
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        if self.head_type == "coral":
            probs = torch.sigmoid(logits)  # (batch, 3)
            scores = probs.sum(dim=-1)     # (batch,) in [0, 3]
        else:
            scores = logits.squeeze(-1)    # (batch,)
        return scores

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        """Save model checkpoint with metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "head_type": self.head_type,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        backbone_name: str | None = None,
        head_type: str | None = None,
        **kwargs,
    ) -> "CrossEncoderReranker":
        """Load model from checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        bb = backbone_name or state.get("backbone_name", "mpnet")
        ht = head_type or state.get("head_type", "coral")

        model = cls(backbone_name=bb, head_type=ht, **kwargs)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded cross-encoder checkpoint: %s", path)
        return model
