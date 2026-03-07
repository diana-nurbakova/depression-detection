"""Loss functions for ordinal relevance scoring.

L_composite = lambda_1 * L_ord + lambda_2 * L_rank + lambda_3 * L_hier + L_sil
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropy(nn.Module):
    """Ordinal-aware cross-entropy with label smoothing toward adjacent levels.

    Instead of hard one-hot targets, smooths probability mass toward
    adjacent ordinal levels: epsilon/2 to each neighbor.
    """

    def __init__(self, num_classes: int = 4, epsilon: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def _smooth_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Create ordinal-smoothed soft targets."""
        batch_size = labels.size(0)
        targets = torch.zeros(batch_size, self.num_classes, device=labels.device)

        for i in range(batch_size):
            y = labels[i].item()
            targets[i, y] = 1.0 - self.epsilon

            # Distribute epsilon to adjacent levels
            neighbors = []
            if y > 0:
                neighbors.append(y - 1)
            if y < self.num_classes - 1:
                neighbors.append(y + 1)

            if neighbors:
                per_neighbor = self.epsilon / len(neighbors)
                for n in neighbors:
                    targets[i, n] = per_neighbor

        return targets

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ordinal CE loss.

        Args:
            logits: (B, C) raw logits
            labels: (B,) integer labels 0..C-1
            weights: (B,) optional per-sample confidence weights
        """
        soft_targets = self._smooth_targets(labels)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)  # (B,)

        if weights is not None:
            loss = loss * weights

        return loss.mean()


class SymmetricCrossEntropy(nn.Module):
    """Symmetric cross-entropy for label noise robustness.

    L_sce = alpha * CE(p, q) + beta * CE(q, p)
    where CE(q, p) = -sum(p * log(q+eps)) acts as a regularizer.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, num_classes: int = 4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, self.num_classes).float()

        # Standard CE: -sum(q * log(p))
        ce_forward = F.cross_entropy(logits, labels, reduction="none")

        # Reverse CE: -sum(p * log(q + eps))
        ce_reverse = -(probs * torch.log(one_hot + 1e-7)).sum(dim=-1)

        loss = self.alpha * ce_forward + self.beta * ce_reverse

        if weights is not None:
            loss = loss * weights

        return loss.mean()


class MarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss for ordinal consistency.

    For pairs (i, j) where label_i > label_j:
        L_rank = max(0, margin - (score_i - score_j))
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise ranking loss within the batch.

        Args:
            logits: (B, C) raw logits
            labels: (B,) integer labels
        """
        # Compute expected score: sum(r * p(r)) for r in {0,1,2,3}
        probs = F.softmax(logits, dim=-1)
        scores = torch.arange(
            logits.size(1), dtype=torch.float, device=logits.device,
        )
        expected = (probs * scores).sum(dim=-1)  # (B,)

        # Build pairwise comparisons
        n = labels.size(0)
        if n < 2:
            return torch.tensor(0.0, device=logits.device)

        # All pairs where label_i > label_j
        labels_i = labels.unsqueeze(1).expand(n, n)
        labels_j = labels.unsqueeze(0).expand(n, n)
        mask = (labels_i > labels_j).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        scores_i = expected.unsqueeze(1).expand(n, n)
        scores_j = expected.unsqueeze(0).expand(n, n)

        # Margin loss: max(0, margin - (score_i - score_j))
        pair_loss = F.relu(self.margin - (scores_i - scores_j))
        loss = (pair_loss * mask).sum() / mask.sum()

        return loss


class HierarchyRegularization(nn.Module):
    """Hierarchy-aware embedding regularization.

    Encourages symptom embeddings within the same factor/subcluster
    to be closer than those in different clusters.

    L_hier = sum_{same cluster} ||e_i - e_j||^2
             - lambda * sum_{diff cluster} ||e_i - e_j||^2
    """

    def __init__(
        self,
        factor_groups: dict[str, list[int]] | None = None,
        lambda_repel: float = 0.01,
    ):
        super().__init__()
        self.lambda_repel = lambda_repel

        # Default ASRS factor groups
        self.factor_groups = factor_groups or {
            "inattention": [7, 8, 9, 10, 11],
            "organization": [1, 2, 3, 4],
            "motor_hi": [5, 6, 12, 13, 14],
            "verbal_hi": [15, 16, 17, 18],
        }

        # Build membership lookup
        self._same_group: set[tuple[int, int]] = set()
        for items in self.factor_groups.values():
            for i in items:
                for j in items:
                    if i != j:
                        self._same_group.add((i, j))

    def forward(self, symptom_embeddings: nn.Embedding) -> torch.Tensor:
        """Compute hierarchy regularization on symptom embeddings."""
        all_ids = []
        for items in self.factor_groups.values():
            all_ids.extend(items)
        all_ids = sorted(set(all_ids))

        if len(all_ids) < 2:
            return torch.tensor(0.0, device=symptom_embeddings.weight.device)

        ids_tensor = torch.tensor(all_ids, device=symptom_embeddings.weight.device)
        embs = symptom_embeddings(ids_tensor)  # (N, d)

        attract_loss = torch.tensor(0.0, device=embs.device)
        repel_loss = torch.tensor(0.0, device=embs.device)
        attract_count = 0
        repel_count = 0

        for i_idx, i_id in enumerate(all_ids):
            for j_idx, j_id in enumerate(all_ids):
                if i_id >= j_id:
                    continue
                dist = (embs[i_idx] - embs[j_idx]).pow(2).sum()
                if (i_id, j_id) in self._same_group:
                    attract_loss = attract_loss + dist
                    attract_count += 1
                else:
                    repel_loss = repel_loss + dist
                    repel_count += 1

        loss = torch.tensor(0.0, device=embs.device)
        if attract_count > 0:
            loss = loss + attract_loss / attract_count
        if repel_count > 0:
            loss = loss - self.lambda_repel * repel_loss / repel_count

        return loss


class CompositeLoss(nn.Module):
    """Combined loss: L = lambda_1*L_ord + lambda_2*L_rank + lambda_3*L_hier.

    For Stage B (noisy silver labels), uses SymmetricCE instead of OrdinalCE.
    """

    def __init__(
        self,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
        lambda_3: float = 0.05,
        epsilon: float = 0.2,
        margin: float = 0.5,
        use_symmetric_ce: bool = False,
        sce_alpha: float = 1.0,
        sce_beta: float = 0.5,
    ):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        if use_symmetric_ce:
            self.ordinal_loss = SymmetricCrossEntropy(
                alpha=sce_alpha, beta=sce_beta,
            )
        else:
            self.ordinal_loss = OrdinalCrossEntropy(epsilon=epsilon)

        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.hierarchy_loss = HierarchyRegularization()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        symptom_embeddings: nn.Embedding,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite loss.

        Returns:
            (total_loss, component_dict) for logging.
        """
        l_ord = self.ordinal_loss(logits, labels, weights)
        l_rank = self.ranking_loss(logits, labels)
        l_hier = self.hierarchy_loss(symptom_embeddings)

        total = (
            self.lambda_1 * l_ord
            + self.lambda_2 * l_rank
            + self.lambda_3 * l_hier
        )

        components = {
            "loss_total": total.item(),
            "loss_ordinal": l_ord.item(),
            "loss_ranking": l_rank.item(),
            "loss_hierarchy": l_hier.item(),
        }

        return total, components
