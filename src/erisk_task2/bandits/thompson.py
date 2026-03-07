"""Thompson Sampling for dynamic symptom weighting (Spec Section 8).

21-armed bandit with Beta posteriors per symptom per user.
"""

from __future__ import annotations

import numpy as np

from erisk_task2.models import NUM_SYMPTOMS


class ThompsonSampler:
    """Per-user Thompson Sampling over 21 BDI-II symptoms."""

    def __init__(self, n_symptoms: int = NUM_SYMPTOMS, tau_active: float = 0.3):
        self.n_symptoms = n_symptoms
        self.tau_active = tau_active

    def init_posteriors(self) -> tuple[np.ndarray, np.ndarray]:
        """Initialize uniform Beta(1,1) priors.

        Returns (alphas, betas) each of shape (21,).
        """
        return np.ones(self.n_symptoms), np.ones(self.n_symptoms)

    def update(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        activations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update Beta posteriors based on symptom activations.

        Args:
            alphas: (21,) current alpha parameters
            betas: (21,) current beta parameters
            activations: (21,) symptom activation scores for this round

        Returns updated (alphas, betas).
        """
        active = activations > self.tau_active
        alphas = alphas + active.astype(float)
        betas = betas + (~active).astype(float)
        return alphas, betas

    def sample_weights(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample normalized weights from Beta posteriors.

        Returns (21,) weight vector summing to 1.
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = rng.beta(alphas, betas)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.n_symptoms) / self.n_symptoms
        return weights

    def compute_features(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        activations: np.ndarray,
    ) -> np.ndarray:
        """Compute all Thompson Sampling features.

        Returns (25,) array:
            - weighted_symptom_score (1d)
            - symptom_entropy (1d)
            - active_symptom_count (1d)
            - weight_vector (21d)
            - posterior_uncertainty (1d)
        """
        # Expected weights (mean of Beta)
        expected = alphas / (alphas + betas)
        total = expected.sum()
        weights = expected / total if total > 0 else np.ones(self.n_symptoms) / self.n_symptoms

        # Weighted symptom score
        weighted_score = float(np.dot(weights, activations))

        # Symptom entropy (of the weight distribution)
        entropy = -float(np.sum(weights * np.log(weights + 1e-10)))

        # Active symptom count
        active_count = float(np.sum(activations > self.tau_active))

        # Posterior uncertainty (mean variance of Beta distributions)
        variance = (alphas * betas) / ((alphas + betas) ** 2 * (alphas + betas + 1))
        uncertainty = float(variance.mean())

        return np.concatenate([
            [weighted_score, entropy, active_count],
            weights,
            [uncertainty],
        ])
