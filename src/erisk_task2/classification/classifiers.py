"""Classification layer (Spec Section 9.2).

XGBoost, Neural Network, SVM, and Ensemble classifiers.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Abstract classifier interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(depressed) for each sample."""
        ...

    @abstractmethod
    def save(self, path: Path):
        ...

    @abstractmethod
    def load(self, path: Path):
        ...


class XGBoostClassifier(BaseClassifier):
    """XGBoost with class imbalance handling."""

    def __init__(
        self,
        max_depth: int = 6,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
    ):
        self.params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        }
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        import xgboost as xgb

        X_scaled = self.scaler.fit_transform(X)

        if self.scale_pos_weight is None:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            spw = n_neg / max(n_pos, 1)
        else:
            spw = self.scale_pos_weight

        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        self.model.fit(X_scaled, y)
        logger.info("XGBoost fitted on %d samples (pos_weight=%.1f)", len(y), spw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "params": self.params}, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.params = data["params"]


class NeuralNetClassifier(BaseClassifier):
    """2-layer MLP classifier."""

    def __init__(self, hidden_sizes: tuple = (256, 64), dropout: float = 0.3, epochs: int = 100):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        # Class weights
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        # Build model
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_sizes[0]),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.BatchNorm1d(self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_sizes[1], 1),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.model.train()
        best_loss = float("inf")
        patience = 10
        wait = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.model.eval()
        logger.info("MLP fitted on %d samples, %d epochs", len(y), epoch + 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        import torch
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t).squeeze(1)
            probs = torch.sigmoid(logits).numpy()
        return probs

    def save(self, path: Path):
        import torch
        data = {
            "state_dict": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "hidden_sizes": self.hidden_sizes,
        }
        torch.save(data, path)

    def load(self, path: Path):
        import torch
        import torch.nn as nn
        data = torch.load(path, weights_only=False)
        self.scaler = data["scaler"]
        self.hidden_sizes = data["hidden_sizes"]
        # Rebuild model (need input_dim from scaler)
        input_dim = self.scaler.n_features_in_
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_sizes[0]),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.BatchNorm1d(self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_sizes[1], 1),
        )
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()


class SVMClassifier(BaseClassifier):
    """RBF SVM classifier."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.svm import SVC
        X_scaled = self.scaler.fit_transform(X)
        self.model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        self.model.fit(X_scaled, y)
        logger.info("SVM fitted on %d samples", len(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]


class EnsembleClassifier(BaseClassifier):
    """Stacking ensemble: XGBoost + NN + SVM with logistic regression meta-learner."""

    def __init__(self):
        self.base_classifiers = [
            XGBoostClassifier(),
            NeuralNetClassifier(),
            SVMClassifier(),
        ]
        self.meta_model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression

        # Generate out-of-fold predictions for meta-learner
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_classifiers)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            for i, clf in enumerate(self.base_classifiers):
                clf_copy = type(clf)()
                clf_copy.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = clf_copy.predict_proba(X[val_idx])

        # Fit meta-learner
        self.meta_model = LogisticRegression(class_weight="balanced", random_state=42)
        self.meta_model.fit(meta_features, y)

        # Refit base classifiers on full data
        for clf in self.base_classifiers:
            clf.fit(X, y)

        logger.info("Ensemble fitted with %d base classifiers", len(self.base_classifiers))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.meta_model is None:
            return np.full(len(X), 0.5)
        meta = np.column_stack([clf.predict_proba(X) for clf in self.base_classifiers])
        return self.meta_model.predict_proba(meta)[:, 1]

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, clf in enumerate(self.base_classifiers):
            clf.save(path / f"base_{i}.pkl")
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(self.meta_model, f)

    def load(self, path: Path):
        path = Path(path)
        for i, clf in enumerate(self.base_classifiers):
            clf.load(path / f"base_{i}.pkl")
        with open(path / "meta.pkl", "rb") as f:
            self.meta_model = pickle.load(f)


def create_classifier(classifier_type: str) -> BaseClassifier:
    """Factory function for creating classifiers."""
    classifiers = {
        "xgboost": XGBoostClassifier,
        "neural_net": NeuralNetClassifier,
        "svm": SVMClassifier,
        "ensemble": EnsembleClassifier,
    }
    cls = classifiers.get(classifier_type)
    if cls is None:
        raise ValueError(f"Unknown classifier: {classifier_type}. Options: {list(classifiers)}")
    return cls()
