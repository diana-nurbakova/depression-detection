"""Context-aware encoder with symptom conditioning for relevance scoring.

Architecture:
    h_pre  = CLS(Encoder(PRE))
    h_text = CLS(Encoder(TEXT))
    h_post = CLS(Encoder(POST))
    h_fused = MultiHeadAttn(Q=h_text, K=[h_pre,h_text,h_post], V=[...])
    z = LayerNorm(W_proj · [h_fused * e_q ; h_fused + e_q])
    logits = W_rel · z  (4 classes: relevance 0-3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Pre-configured backbones
BACKBONES = {
    "mental-roberta": "mental/mental-roberta-base",
    "clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

NUM_CLASSES = 4  # relevance: 0, 1, 2, 3
NUM_SYMPTOMS = 18  # ASRS items 1-18


class ContextFusion(nn.Module):
    """Multi-head attention to fuse PRE/TEXT/POST context."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_text: torch.Tensor,   # (B, d)
        h_pre: torch.Tensor,    # (B, d)
        h_post: torch.Tensor,   # (B, d)
    ) -> torch.Tensor:
        # Stack context: (B, 3, d)
        context = torch.stack([h_pre, h_text, h_post], dim=1)
        # Query is h_text: (B, 1, d)
        query = h_text.unsqueeze(1)
        # Attend over context
        fused, _ = self.attn(query, context, context)
        # Residual + norm
        fused = self.norm(fused.squeeze(1) + h_text)
        return fused


class SymptomConditionedEncoder(nn.Module):
    """Full encoder: backbone + context fusion + symptom conditioning + classifier."""

    def __init__(
        self,
        backbone_name: str = "mpnet",
        num_symptoms: int = NUM_SYMPTOMS,
        num_classes: int = NUM_CLASSES,
        fusion_heads: int = 4,
        dropout: float = 0.1,
        freeze_backbone_layers: int = 0,
    ):
        super().__init__()

        # Resolve backbone
        model_id = BACKBONES.get(backbone_name, backbone_name)
        self.backbone = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        hidden_dim = self.backbone.config.hidden_size
        self.hidden_dim = hidden_dim

        # Context fusion
        self.context_fusion = ContextFusion(
            hidden_dim, num_heads=fusion_heads, dropout=dropout,
        )

        # Learnable symptom embeddings (initialized later from priors)
        self.symptom_embeddings = nn.Embedding(num_symptoms + 1, hidden_dim)
        # +1 because symptoms are 1-indexed; index 0 is unused

        # Projection: [h_fused * e_q ; h_fused + e_q] -> d
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Ordinal classifier: d -> 4 logits
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Optional: freeze backbone layers for gradual unfreezing
        if freeze_backbone_layers > 0:
            self._freeze_layers(freeze_backbone_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N transformer layers of the backbone."""
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.backbone, "encoder"):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
        else:
            logger.warning("Cannot identify backbone layers for freezing")
            return

        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        logger.info("Froze %d/%d backbone layers", num_layers, len(layers))

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (for later training stages)."""
        for param in self.parameters():
            param.requires_grad = True

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text through backbone, return CLS pooled representation."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token pooling
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        symptom_ids: torch.Tensor,
        pre_input_ids: torch.Tensor | None = None,
        pre_attention_mask: torch.Tensor | None = None,
        post_input_ids: torch.Tensor | None = None,
        post_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_input_ids: (B, seq_len) tokenized TARGET text
            text_attention_mask: (B, seq_len)
            symptom_ids: (B,) ASRS item numbers (1-18)
            pre_input_ids: (B, seq_len) tokenized PRE context (optional)
            pre_attention_mask: (B, seq_len)
            post_input_ids: (B, seq_len) tokenized POST context (optional)
            post_attention_mask: (B, seq_len)

        Returns:
            logits: (B, 4) ordinal relevance logits
        """
        # Encode text
        h_text = self._encode_text(text_input_ids, text_attention_mask)

        # Encode context if available, otherwise use text as context
        if pre_input_ids is not None and pre_attention_mask is not None:
            h_pre = self._encode_text(pre_input_ids, pre_attention_mask)
        else:
            h_pre = h_text

        if post_input_ids is not None and post_attention_mask is not None:
            h_post = self._encode_text(post_input_ids, post_attention_mask)
        else:
            h_post = h_text

        # Fuse context
        h_fused = self.context_fusion(h_text, h_pre, h_post)

        # Symptom conditioning
        e_q = self.symptom_embeddings(symptom_ids)  # (B, d)

        # Hadamard product + addition concatenation
        combined = torch.cat([h_fused * e_q, h_fused + e_q], dim=-1)  # (B, 2d)

        # Project and classify
        z = self.projection(combined)  # (B, d)
        logits = self.classifier(z)    # (B, 4)

        return logits

    def get_representation(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        symptom_ids: torch.Tensor,
        pre_input_ids: torch.Tensor | None = None,
        pre_attention_mask: torch.Tensor | None = None,
        post_input_ids: torch.Tensor | None = None,
        post_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get the projected representation z (for calibration and analysis)."""
        h_text = self._encode_text(text_input_ids, text_attention_mask)

        if pre_input_ids is not None and pre_attention_mask is not None:
            h_pre = self._encode_text(pre_input_ids, pre_attention_mask)
        else:
            h_pre = h_text

        if post_input_ids is not None and post_attention_mask is not None:
            h_post = self._encode_text(post_input_ids, post_attention_mask)
        else:
            h_post = h_text

        h_fused = self.context_fusion(h_text, h_pre, h_post)
        e_q = self.symptom_embeddings(symptom_ids)
        combined = torch.cat([h_fused * e_q, h_fused + e_q], dim=-1)
        return self.projection(combined)

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        """Save model checkpoint with metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.state_dict(),
            "hidden_dim": self.hidden_dim,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        logger.info("Checkpoint saved: %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        backbone_name: str = "mpnet",
        **kwargs,
    ) -> "SymptomConditionedEncoder":
        """Load model from checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(backbone_name=backbone_name, **kwargs)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Checkpoint loaded: %s", path)
        return model
