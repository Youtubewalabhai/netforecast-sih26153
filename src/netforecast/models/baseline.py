"""Logistic Regression Baseline for K-step Ahead Infiltration Forecasting."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaselineKStepClassifier:
    """Standard Machine Learning Baseline (Logistic Regression)

    Predicts infiltration K windows ahead given the current window or flattened sequence.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit baseline on features.

        Args:
            X: Features. If 3D (N, seq_len, D), it will be flattened or take the last window.
            y: Target binary label (infiltration at t+K or t+1).
        """
        if X.ndim == 3:
            # Use flattened sequence or last window state
            N, S, D = X.shape
            X_flat = X.reshape(N, S * D)
        else:
            X_flat = X

        self.model.fit(X_flat, y.ravel())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict infiltration probability."""
        if X.ndim == 3:
            N, S, D = X.shape
            X_flat = X.reshape(N, S * D)
        else:
            X_flat = X
        # Return probability of positive class (infiltration)
        return self.model.predict_proba(X_flat)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary infiltration class."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(
        self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute precision, recall, F1, FPR, ROC-AUC."""
        y_true_flat = y_true.ravel()
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        prec = precision_score(y_true_flat, preds, zero_division=0)
        rec = recall_score(y_true_flat, preds, zero_division=0)
        f1 = f1_score(y_true_flat, preds, zero_division=0)
        acc = accuracy_score(y_true_flat, preds)

        cm = confusion_matrix(y_true_flat, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        try:
            auc = roc_auc_score(y_true_flat, probs) if len(np.unique(y_true_flat)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
            "roc_auc": float(auc),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    def save(self, path: Union[str, Path]):
        """Save baseline model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Union[str, Path]):
        """Load baseline model."""
        self.model = joblib.load(path)
        return self
