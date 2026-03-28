"""Model explainability using SHAP and LIME."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Explainer:
    """Generate model explanations using SHAP and LIME."""

    def __init__(self, model, X_background: Optional[pd.DataFrame] = None):
        """Initialize explainer."""
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize SHAP explainer."""
        try:
            import shap

            if hasattr(self.model, "model"):
                model = self.model.model
            else:
                model = self.model

            # TreeExplainer for tree-based models
            if hasattr(model, "predict") and hasattr(model, "feature_importances_"):
                if self.X_background is not None:
                    self.explainer = shap.TreeExplainer(model)
                    logger.info("Initialized SHAP TreeExplainer")
        except ImportError:
            logger.warning("SHAP not available")

    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction."""

        if self.explainer is None:
            logger.warning("Explainer not initialized, returning feature importance instead")
            return self._feature_importance_explanation(X)

        try:
            import shap

            # Get SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle different return types
            if isinstance(shap_values, list):
                # For multi-class, take positive class
                sv = shap_values[1]
            else:
                sv = shap_values

            explanation = {
                "prediction": self.model.predict(X)[sample_idx],
                "shap_values": sv[sample_idx].tolist() if hasattr(sv[sample_idx], "tolist") else sv[sample_idx],
                "base_value": float(self.explainer.expected_value),
                "feature_names": list(X.columns),
            }

            return explanation
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._feature_importance_explanation(X)

    def explain_multiple(self, X: pd.DataFrame, n_samples: int = 10) -> list:
        """Generate explanations for multiple predictions."""
        explanations = []

        for i in range(min(n_samples, len(X))):
            exp = self.explain_prediction(X, sample_idx=i)
            explanations.append(exp)

        return explanations

    def feature_importance(self) -> Dict[str, float]:
        """Get global feature importance."""
        try:
            if hasattr(self.model, "get_feature_importance"):
                return self.model.get_feature_importance()
            elif hasattr(self.model, "feature_importances_"):
                model = self.model.model if hasattr(self.model, "model") else self.model
                importances = model.feature_importances_
                feature_names = (
                    self.model.feature_names
                    if hasattr(self.model, "feature_names")
                    else [f"feature_{i}" for i in range(len(importances))]
                )
                return dict(zip(feature_names, importances))
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")

        return {}

    def _feature_importance_explanation(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Fallback explanation using feature importance."""
        importance = self.feature_importance()

        return {
            "prediction": self.model.predict(X)[0],
            "feature_importance": importance,
            "method": "feature_importance_fallback",
        }

    def pdp_explanation(
        self, X: pd.DataFrame, feature_name: str, n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate partial dependence plot data."""
        if feature_name not in X.columns:
            raise ValueError(f"Feature '{feature_name}' not found in data")

        feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), n_points)
        predictions = []

        for val in feature_values:
            X_temp = X.copy()
            X_temp[feature_name] = val

            if hasattr(self.model, "predict_proba"):
                pred = self.model.predict_proba(X_temp).mean(axis=0)
            else:
                pred = self.model.predict(X_temp).mean()

            predictions.append(pred)

        return feature_values, np.array(predictions)
