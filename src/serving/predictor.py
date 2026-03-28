"""Real-time prediction service."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Predictor:
    """Real-time prediction service for single or batch samples."""

    def __init__(self, model, preprocessor=None, explainer=None):
        """Initialize predictor."""
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = explainer
        self.prediction_cache = {}

    def predict_health(
        self, patient_data: Dict[str, Any], explain: bool = False
    ) -> Dict[str, Any]:
        """Predict health deterioration for a patient."""

        # Convert to DataFrame
        X = pd.DataFrame([patient_data])

        # Preprocess if available
        if self.preprocessor:
            X = self.preprocessor.preprocess_health_data(X, fit=False)

        # Predict
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        result = {
            "prediction": int(prediction),
            "probability": float(proba[1]),
            "risk_level": self._get_risk_level(proba[1]),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add explanation if requested
        if explain and self.explainer:
            try:
                explanation = self.explainer.explain_prediction(X)
                result["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")

        return result

    def predict_stock_pattern(
        self, stock_data: pd.DataFrame, explain: bool = False
    ) -> Dict[str, Any]:
        """Predict stock chart pattern."""

        if len(stock_data) < 50:
            return {"error": "Insufficient data (need at least 50 records)"}

        # Extract features
        if hasattr(self.model, "extract_pattern_features"):
            features = self.model.extract_pattern_features(stock_data)
            X = features.iloc[[-1]]
        else:
            X = stock_data.iloc[[-1]]

        # Predict
        pattern, confidence = self.model.predict_proba(X)

        result = {
            "pattern": pattern,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    def predict_stock_forecast(
        self, stock_data: pd.DataFrame, horizon: int = 30
    ) -> Dict[str, Any]:
        """Forecast stock price."""

        if len(stock_data) == 0:
            return {"error": "Empty data"}

        # Extract close prices
        if "close" in stock_data.columns:
            prices = stock_data["close"]
        else:
            prices = stock_data.iloc[:, -1]

        # Forecast
        forecasts, confidence = self.model.forecast(
            pd.DataFrame(prices), horizon=horizon
        )

        result = {
            "forecast": forecasts.tolist(),
            "confidence_lower": confidence[0].tolist(),
            "confidence_upper": confidence[1].tolist(),
            "horizon": horizon,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return result

    def _get_risk_level(self, probability: float) -> str:
        """Determine clinical risk level."""
        if probability < 0.35:
            return "safe"
        elif probability < 0.65:
            return "warning"
        else:
            return "critical"

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": type(self.model).__name__,
            "has_explainer": self.explainer is not None,
            "has_preprocessor": self.preprocessor is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }
