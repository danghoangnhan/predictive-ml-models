"""Ensemble model combining multiple predictors."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Ensemble classifier combining multiple models."""

    def __init__(self, models: Optional[List] = None, weights: Optional[List[float]] = None):
        """Initialize ensemble model."""
        self.models = models or []
        self.weights = weights or [1.0 / len(models)] if models else []

    def add_model(self, model, weight: float = 1.0) -> None:
        """Add model to ensemble."""
        self.models.append(model)
        self.weights.append(weight)
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        logger.info(f"Added model to ensemble. Total models: {len(self.models)}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        """Fit all models in ensemble."""
        for i, model in enumerate(self.models):
            if hasattr(model, "fit"):
                model.fit(X, y)
                logger.info(f"Trained model {i+1}/{len(self.models)} in ensemble")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble voting."""
        if not self.models:
            raise ValueError("Ensemble has no models")

        predictions = []

        for model in self.models:
            if hasattr(model, "predict"):
                pred = model.predict(X)
                predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted majority voting for classification
        if predictions.dtype == object or len(predictions.shape) == 2:
            # Handle multi-class or string predictions
            unique_classes = np.unique(predictions)
            weighted_votes = np.zeros(len(X))

            for i, pred in enumerate(predictions):
                for j, sample in enumerate(pred):
                    if sample == unique_classes[-1]:  # Assume last class is positive
                        weighted_votes[j] += self.weights[i]

            return (weighted_votes > 0.5).astype(int)
        else:
            return np.mean(predictions * np.array(self.weights).reshape(-1, 1), axis=0).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability using ensemble averaging."""
        if not self.models:
            raise ValueError("Ensemble has no models")

        probas = []

        for model in self.models:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                probas.append(proba)

        if not probas:
            raise ValueError("No models support predict_proba")

        probas = np.array(probas)
        weighted_proba = np.average(probas, axis=0, weights=self.weights)

        return weighted_proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance from all models."""
        all_importances = {}

        for i, model in enumerate(self.models):
            if hasattr(model, "get_feature_importance"):
                importances = model.get_feature_importance()
                for feature, importance in importances.items():
                    if feature not in all_importances:
                        all_importances[feature] = 0
                    all_importances[feature] += importance * self.weights[i]
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if hasattr(model, "feature_names"):
                    for feature, importance in zip(model.feature_names, importances):
                        if feature not in all_importances:
                            all_importances[feature] = 0
                        all_importances[feature] += importance * self.weights[i]

        return dict(sorted(all_importances.items(), key=lambda x: x[1], reverse=True))

    def get_model_count(self) -> int:
        """Get number of models in ensemble."""
        return len(self.models)

    def get_model_weights(self) -> Dict[int, float]:
        """Get model weights."""
        return {i: w for i, w in enumerate(self.weights)}
