"""Model training pipeline."""

import logging
from typing import Any

import pandas as pd

from src.data import DataSplitter
from src.models import HealthPredictor, PatternDetector

logger = logging.getLogger(__name__)


class Trainer:
    """Train and evaluate ML models."""

    def __init__(self, model_type: str = "health"):
        """Initialize trainer."""
        self.model_type = model_type
        self.model = None
        self.training_history = {}

    def train_health_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str = "xgboost",
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> dict[str, Any]:
        """Train health prediction model."""

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_train_val_test(
            X, y, test_size=test_size, val_size=val_size, stratify=True
        )

        # Initialize and train model
        self.model = HealthPredictor(model_type=algorithm)
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.model.score(X_train, y_train)
        val_score = self.model.model.score(X_val, y_val)
        test_score = self.model.model.score(X_test, y_test)

        self.training_history = {
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "test_accuracy": test_score,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }

        logger.info(
            f"Health model trained. "
            f"Train: {train_score:.4f}, Val: {val_score:.4f}, Test: {test_score:.4f}"
        )

        return self.training_history

    def train_stock_pattern_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train stock pattern detection model."""

        # Split data
        X_train, X_test = self._simple_split(X, test_size)
        y_train, y_test = self._simple_split(y, test_size)

        # Train model
        self.model = PatternDetector()
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.model.score(X_train, y_train)
        test_score = self.model.model.score(X_test, y_test)

        self.training_history = {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        logger.info(
            f"Stock pattern model trained. Train: {train_score:.4f}, Test: {test_score:.4f}"
        )

        return self.training_history

    def get_model(self):
        """Get trained model."""
        return self.model

    def get_training_history(self) -> dict[str, Any]:
        """Get training history."""
        return self.training_history

    @staticmethod
    def _simple_split(data, test_size: float = 0.2):
        """Simple train-test split."""
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]
