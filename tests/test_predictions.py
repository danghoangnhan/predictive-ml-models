"""Integration tests for predictions."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import ModelMetrics
from src.models import HealthPredictor
from src.serving import Predictor


class TestHealthMetrics:
    """Test health evaluation metrics."""

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3]])

        metrics = ModelMetrics.classification_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        result = ModelMetrics.confusion_matrix_report(y_true, y_pred)

        assert "confusion_matrix" in result
        assert result["confusion_matrix"].shape == (2, 2)


class TestPredictionService:
    """Test prediction service."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        X = pd.DataFrame(
            {
                "gad7_score": np.random.randint(0, 22, 50),
                "age": np.random.randint(18, 80, 50),
                "gender": np.random.choice([0, 1], 50),
                "bmi": np.random.uniform(18, 35, 50),
                "sleep_hours": np.random.uniform(4, 10, 50),
            }
        )
        y = np.random.randint(0, 2, 50)

        model = HealthPredictor(model_type="xgboost")
        model.fit(X, y)
        return model

    def test_single_prediction(self, trained_model):
        """Test single prediction through service."""
        predictor = Predictor(trained_model)

        patient_data = {
            "gad7_score": 15,
            "age": 35,
            "gender": 0,
            "bmi": 24.5,
            "sleep_hours": 6.5,
        }

        result = predictor.predict_health(patient_data)

        assert "prediction" in result
        assert "probability" in result
        assert "risk_level" in result
        assert result["risk_level"] in ["safe", "warning", "critical"]

    def test_risk_level_determination(self, trained_model):
        """Test risk level determination."""
        predictor = Predictor(trained_model)

        # Test low probability
        assert predictor._get_risk_level(0.2) == "safe"

        # Test medium probability
        assert predictor._get_risk_level(0.5) == "warning"

        # Test high probability
        assert predictor._get_risk_level(0.8) == "critical"
