"""
Phase 5 — Model Training and Evaluation
-----------------------------------------
Supports three classifiers from the paper:
  1. Support Vector Machine (SVM)       — F1 ≈ 0.85
  2. Random Forest                      — F1 ≈ 0.83
  3. Feedforward Neural Network (MLP)   — F1 ≈ 0.89  ← best performer

Evaluation: accuracy, precision, recall, F1 (macro), k-fold cross-validation,
and confusion matrix analysis.

Sentiment classes
-----------------
  0 → negative
  1 → neutral
  2 → positive
  3 → suspicious
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

LABEL_NAMES = ["negative", "neutral", "positive", "suspicious"]


class SentimentClassifier:
    """
    Unified wrapper around SVM / RandomForest / MLP classifiers.
    """

    MODELS = {
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }

    def __init__(self, model_type: str = "neural_network"):
        if model_type not in self.MODELS:
            raise ValueError(f"model_type must be one of {list(self.MODELS.keys())}")
        self.model_type = model_type
        self.model = self.MODELS[model_type]
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.model_type} does not support predict_proba")

    def predict_label(self, X: np.ndarray) -> list:
        """Return human-readable sentiment labels."""
        indices = self.predict(X)
        return [LABEL_NAMES[i] if i < len(LABEL_NAMES) else "unknown" for i in indices]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        self._check_fitted()
        y_pred = self.predict(X_test)
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
        """K-fold cross-validation to reduce overfitting risk."""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.model, X, y, cv=kf,
            scoring=["accuracy", "f1_macro"],
            return_train_score=True,
        )
        return {
            "cv_accuracy_mean": round(cv_results["test_accuracy"].mean(), 4),
            "cv_accuracy_std": round(cv_results["test_accuracy"].std(), 4),
            "cv_f1_mean": round(cv_results["test_f1_macro"].mean(), 4),
            "cv_f1_std": round(cv_results["test_f1_macro"].std(), 4),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        joblib.dump(self.model, path)
        print(f"[SentimentClassifier] Model saved to {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"[SentimentClassifier] Model loaded from {path}")
        return self

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")


# ---------------------------------------------------------------------------
# Convenience: train and compare all three classifiers
# ---------------------------------------------------------------------------

def compare_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Train SVM, Random Forest, and Neural Network on the same split.
    Returns a dict of evaluation metrics per model.
    """
    results = {}
    for name in ["svm", "random_forest", "neural_network"]:
        clf = SentimentClassifier(model_type=name)
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        results[name] = {
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        print(f"[{name}] F1={metrics['f1_macro']}  Acc={metrics['accuracy']}")
    return results
