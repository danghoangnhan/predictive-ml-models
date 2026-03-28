"""Cross-validation utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation wrapper."""

    def __init__(self, cv_folds: int = 5, stratified: bool = True):
        """Initialize cross-validator."""
        self.cv_folds = cv_folds
        self.stratified = stratified

    def validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "accuracy",
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation."""

        cv = self.cv_folds if self.cv_folds > 1 else 5

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        logger.info(
            f"Cross-validation (n_folds={cv}): "
            f"mean={scores.mean():.4f}, std={scores.std():.4f}"
        )

        return {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
            "folds": len(scores),
        }

    def validate_multiple_metrics(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation with multiple metrics."""

        cv = self.cv_folds if self.cv_folds > 1 else 5

        cv_results = cross_validate(model, X, y, cv=cv, scoring=metrics)

        results = {}
        for metric in metrics:
            key = f"test_{metric}"
            if key in cv_results:
                scores = cv_results[key]
                results[metric] = {
                    "scores": scores,
                    "mean": scores.mean(),
                    "std": scores.std(),
                }

        logger.info(f"Multi-metric cross-validation completed")
        return results
