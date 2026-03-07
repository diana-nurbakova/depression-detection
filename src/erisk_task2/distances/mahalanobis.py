"""Mahalanobis Distance for cross-user anomaly scoring (Spec Section 5.2).

Fit on control users, compute distance for each user against control norms.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MahalanobisScorer:
    """Mahalanobis distance scorer fit on control user feature distributions."""

    def __init__(self, n_pca_components: int = 50):
        self.n_pca_components = n_pca_components
        self.pca: Optional[PCA] = None
        self.control_mean: Optional[np.ndarray] = None
        self.control_cov_inv: Optional[np.ndarray] = None
        self.depressed_mean: Optional[np.ndarray] = None
        self.depressed_cov_inv: Optional[np.ndarray] = None

    def fit(
        self,
        control_features: np.ndarray,
        depressed_features: Optional[np.ndarray] = None,
    ):
        """Fit reference distributions from training data.

        Args:
            control_features: (n_control, n_features) array
            depressed_features: (n_depressed, n_features) optional, for relative MD
        """
        # PCA reduction for numerical stability
        n_components = min(self.n_pca_components, control_features.shape[1], control_features.shape[0] - 1)
        self.pca = PCA(n_components=n_components)

        # Fit PCA on control data
        control_reduced = self.pca.fit_transform(control_features)

        # Control distribution
        self.control_mean = control_reduced.mean(axis=0)
        cov = LedoitWolf().fit(control_reduced)
        self.control_cov_inv = np.linalg.pinv(cov.covariance_)

        # Depressed distribution (for relative Mahalanobis)
        if depressed_features is not None and len(depressed_features) > 5:
            dep_reduced = self.pca.transform(depressed_features)
            self.depressed_mean = dep_reduced.mean(axis=0)
            dep_cov = LedoitWolf().fit(dep_reduced)
            self.depressed_cov_inv = np.linalg.pinv(dep_cov.covariance_)

        logger.info(
            "Mahalanobis fitted: PCA %d -> %d components, control=%d, depressed=%d",
            control_features.shape[1], n_components,
            len(control_features),
            len(depressed_features) if depressed_features is not None else 0,
        )

    def score(self, feature_vector: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for a single user.

        Returns (3,) array: [D_M_control, D_M_relative, D_M_depressed]
        """
        if self.pca is None or self.control_mean is None:
            return np.zeros(3)

        x = self.pca.transform(feature_vector.reshape(1, -1))[0]

        # Distance from control distribution
        diff_c = x - self.control_mean
        d_control = float(np.sqrt(np.maximum(diff_c @ self.control_cov_inv @ diff_c, 0)))

        # Distance from depressed distribution
        d_depressed = 0.0
        if self.depressed_mean is not None and self.depressed_cov_inv is not None:
            diff_d = x - self.depressed_mean
            d_depressed = float(np.sqrt(np.maximum(diff_d @ self.depressed_cov_inv @ diff_d, 0)))

        # Relative Mahalanobis: RMD = D_control - D_depressed
        d_relative = d_control - d_depressed

        return np.array([d_control, d_relative, d_depressed])

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump({
                "pca": self.pca,
                "control_mean": self.control_mean,
                "control_cov_inv": self.control_cov_inv,
                "depressed_mean": self.depressed_mean,
                "depressed_cov_inv": self.depressed_cov_inv,
                "n_pca_components": self.n_pca_components,
            }, f)

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pca = data["pca"]
        self.control_mean = data["control_mean"]
        self.control_cov_inv = data["control_cov_inv"]
        self.depressed_mean = data.get("depressed_mean")
        self.depressed_cov_inv = data.get("depressed_cov_inv")
        self.n_pca_components = data.get("n_pca_components", 50)
