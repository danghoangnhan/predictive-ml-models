"""Mental health prediction model (GAD-7 based)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Tuple, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthPredictor:
    """Predict patient mental health deterioration from GAD-7 scores."""

    def __init__(self, model_type: str = "xgboost"):
        """Initialize health predictor."""
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.threshold = 0.65

        if model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ) -> "HealthPredictor":
        """Train health predictor."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y, sample_weight=sample_weight)
        logger.info(f"Trained {self.model_type} health predictor on {len(X)} samples")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict clinical deterioration (0/1)."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict deterioration probability."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)

    def predict_with_risk_level(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with clinical risk levels: safe, warning, critical."""
        proba = self.predict_proba(X)[:, 1]

        risk_levels = np.zeros(len(proba), dtype=object)
        risk_levels[proba < 0.35] = "safe"
        risk_levels[(proba >= 0.35) & (proba < 0.65)] = "warning"
        risk_levels[proba >= 0.65] = "critical"

        return risk_levels, proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            raise ValueError(f"Model {self.model_type} does not support feature_importances_")

        importance_dict = dict(zip(self.feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved model to {path}")

    @staticmethod
    def load(path: Path) -> "HealthPredictor":
        """Load model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return model
