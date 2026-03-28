import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP-based model explainability."""

    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.explainer = None

    def create_explainer(self):
        """Create SHAP explainer."""
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer created")
        except ImportError:
            logger.error("SHAP not installed")
            raise

    def explain_prediction(self, x: np.ndarray) -> dict:
        """Explain a single prediction using SHAP."""
        if self.explainer is None:
            self.create_explainer()

        shap_values = self.explainer.shap_values(x)
        base_value = self.explainer.expected_value

        return {"shap_values": shap_values, "base_value": base_value}

    def feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> dict:
        """Get feature importance from SHAP values."""
        if self.explainer is None:
            self.create_explainer()

        shap_values = self.explainer.shap_values(X)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        feature_importance = {X.columns[i]: mean_abs_shap[i] for i in range(len(X.columns))}

        # Sort and return top N
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        return {k: v for k, v in list(sorted_importance.items())[:top_n]}


class LimeExplainer:
    """LIME-based model explainability."""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: list = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.explainer = None

    def create_explainer(self):
        """Create LIME explainer."""
        try:
            import lime
            import lime.lime_tabular

            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values, feature_names=self.feature_names, mode="classification"
            )
            logger.info("LIME explainer created")
        except ImportError:
            logger.error("LIME not installed")
            raise

    def explain_prediction(self, x: np.ndarray, top_features: int = 10) -> dict:
        """Explain a prediction using LIME."""
        if self.explainer is None:
            self.create_explainer()

        explanation = self.explainer.explain_instance(x, self.model.predict_proba)

        return {
            "prediction": explanation.predicted_label,
            "top_features": explanation.as_list(label=1)[:top_features],
        }


class FeatureImportance:
    """Model-agnostic feature importance."""

    @staticmethod
    def get_importance(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        """Get feature importance from model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_dict = {X_test.columns[i]: importances[i] for i in range(len(X_test.columns))}
            return dict(sorted(feature_dict.items(), key=lambda x: x[1], reverse=True))

        logger.warning("Model does not support feature_importances_")
        return None
