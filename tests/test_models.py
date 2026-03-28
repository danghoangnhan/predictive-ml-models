"""Tests for ML models."""

import numpy as np
import pandas as pd
import pytest

from src.models import HealthPredictor, PatternDetector


class TestHealthPredictor:
    """Test health prediction model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample health data."""
        df = pd.DataFrame(
            {
                "gad7_score": np.random.randint(0, 22, 50),
                "age": np.random.randint(18, 80, 50),
                "gender": np.random.choice([0, 1], 50),
                "bmi": np.random.uniform(18, 35, 50),
                "sleep_hours": np.random.uniform(4, 10, 50),
            }
        )
        y = np.random.randint(0, 2, 50)
        return df, y

    def test_initialization(self):
        """Test model initialization."""
        model = HealthPredictor(model_type="xgboost")
        assert model.model is not None
        assert model.model_type == "xgboost"

    def test_fit_predict(self, sample_data):
        """Test training and prediction."""
        X, y = sample_data
        model = HealthPredictor(model_type="xgboost")
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        model = HealthPredictor(model_type="xgboost")
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        model = HealthPredictor(model_type="xgboost")
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert all(isinstance(k, str) for k in importance.keys())


class TestPatternDetector:
    """Test stock pattern detector."""

    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data."""
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(100, 110, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.randint(900000, 1100000, 100),
            }
        )
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def test_initialization(self):
        """Test pattern detector initialization."""
        detector = PatternDetector()
        assert detector.model is not None
        assert len(detector.PATTERNS) > 0

    def test_feature_extraction(self, sample_stock_data):
        """Test pattern feature extraction."""
        detector = PatternDetector()
        features = detector.extract_pattern_features(sample_stock_data)

        assert len(features) > 0
        assert "volatility" in features.columns
        assert "support" in features.columns
        assert "resistance" in features.columns

    def test_fit_predict(self, sample_stock_data):
        """Test pattern detection training and prediction."""
        detector = PatternDetector()
        features = detector.extract_pattern_features(sample_stock_data)

        # Create dummy labels
        y = np.random.choice(detector.PATTERNS, len(features))

        detector.fit(features, y)
        predictions = detector.predict(features)

        assert len(predictions) == len(features)
        assert all(p in detector.PATTERNS for p in predictions)
