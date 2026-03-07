"""Post-hoc calibration for ordinal probability estimates.

Per-symptom temperature scaling + optional Dirichlet calibration.
Re-calibrated every E_recal=3 epochs during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Per-symptom temperature scaling.

    Learns T_j for each symptom j such that:
        p_cal(r | s, q_j) = softmax(logits / T_j)
    """

    def __init__(self, num_symptoms: int = 18):
        super().__init__()
        # Initialize temperatures to 1.0 (no effect)
        self.temperatures = nn.Parameter(
            torch.ones(num_symptoms + 1),  # 1-indexed
        )

    def forward(
        self,
        logits: torch.Tensor,
        symptom_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            logits: (B, C) raw logits from encoder
            symptom_ids: (B,) symptom IDs (1-indexed)

        Returns:
            Calibrated probabilities (B, C)
        """
        temps = self.temperatures[symptom_ids].unsqueeze(-1)  # (B, 1)
        # Clamp temperature to prevent division by zero
        temps = temps.clamp(min=0.1, max=10.0)
        scaled_logits = logits / temps
        return F.softmax(scaled_logits, dim=-1)

    def fit(
        self,
        logits_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        symptom_ids_list: list[torch.Tensor],
        max_iter: int = 50,
    ) -> None:
        """Fit temperatures on validation data using LBFGS."""
        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        all_symptom_ids = torch.cat(symptom_ids_list, dim=0)

        optimizer = LBFGS([self.temperatures], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temps = self.temperatures[all_symptom_ids].unsqueeze(-1).clamp(min=0.1, max=10.0)
            scaled = all_logits / temps
            loss = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        logger.info(
            "Temperature scaling fitted. Range: [%.3f, %.3f]",
            self.temperatures[1:19].min().item(),
            self.temperatures[1:19].max().item(),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


class DirichletCalibration(nn.Module):
    """Dirichlet calibration: learns a linear transform in log-probability space.

    p_cal = softmax(W_dir * log(p_uncal) + b_dir)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.eye(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply Dirichlet calibration.

        Args:
            probs: (B, C) probability estimates

        Returns:
            Calibrated probabilities (B, C)
        """
        log_probs = torch.log(probs.clamp(min=1e-7))
        transformed = log_probs @ self.W.T + self.b
        return F.softmax(transformed, dim=-1)

    def fit(
        self,
        probs_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Fit Dirichlet parameters on validation data."""
        all_probs = torch.cat(probs_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        optimizer = LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            cal_probs = self.forward(all_probs)
            loss = F.nll_loss(torch.log(cal_probs.clamp(min=1e-7)), all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        logger.info("Dirichlet calibration fitted.")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


class CalibrationPipeline:
    """Full calibration: temperature scaling -> Dirichlet."""

    def __init__(self, num_symptoms: int = 18, num_classes: int = 4):
        self.temp_scaling = TemperatureScaling(num_symptoms)
        self.dirichlet = DirichletCalibration(num_classes)

    def calibrate(
        self,
        logits: torch.Tensor,
        symptom_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply full calibration pipeline.

        Returns calibrated probabilities (B, C).
        """
        # Step 1: temperature-scaled probabilities
        temp_probs = self.temp_scaling(logits, symptom_ids)
        # Step 2: Dirichlet refinement
        cal_probs = self.dirichlet(temp_probs)
        return cal_probs

    def expected_score(
        self,
        logits: torch.Tensor,
        symptom_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected relevance score: sum(r * p_cal(r)).

        Returns (B,) expected scores in [0, 3].
        """
        cal_probs = self.calibrate(logits, symptom_ids)
        levels = torch.arange(
            cal_probs.size(-1), dtype=torch.float, device=cal_probs.device,
        )
        return (cal_probs * levels).sum(dim=-1)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = "cpu",
    ) -> None:
        """Fit calibration on validation data."""
        model.eval()
        all_logits = []
        all_labels = []
        all_symptom_ids = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    symptom_ids=batch["symptom_id"],
                    pre_input_ids=batch.get("pre_input_ids"),
                    pre_attention_mask=batch.get("pre_attention_mask"),
                    post_input_ids=batch.get("post_input_ids"),
                    post_attention_mask=batch.get("post_attention_mask"),
                )
                all_logits.append(logits.cpu())
                all_labels.append(batch["label"].cpu())
                all_symptom_ids.append(batch["symptom_id"].cpu())

        # Fit temperature scaling
        self.temp_scaling.fit(all_logits, all_labels, all_symptom_ids)

        # Fit Dirichlet on temperature-scaled probs
        temp_probs = [
            self.temp_scaling(l, s) for l, s in zip(all_logits, all_symptom_ids)
        ]
        self.dirichlet.fit(temp_probs, all_labels)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.temp_scaling.save(directory / "temp_scaling.pt")
        self.dirichlet.save(directory / "dirichlet.pt")

    def load(self, directory: Path) -> None:
        self.temp_scaling.load(directory / "temp_scaling.pt")
        self.dirichlet.load(directory / "dirichlet.pt")
