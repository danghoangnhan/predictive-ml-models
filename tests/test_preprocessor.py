"""Tests for data preprocessing."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import Preprocessor


class TestPreprocessor:
    """Test preprocessing utilities."""

    def test_health_data_preprocessing(self):
        """Test health data preprocessing."""
        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "gad7_score": [10, 15],
            "age": [35, 45],
            "gender": ["M", "F"],
            "clinical_deterioration": [0, 1],
        })

        preprocessor = Preprocessor()
        result = preprocessor.preprocess_health_data(df, fit=True)

        assert len(result) == 2
        assert "gad7_severity" in result.columns
        assert "age_group" in result.columns

    def test_stock_data_preprocessing(self):
        """Test stock data preprocessing."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000000] * 10,
            "pattern": ["breakout"] * 10,
        })

        preprocessor = Preprocessor()
        result = preprocessor.preprocess_stock_data(df, fit=True)

        assert len(result) > 0
        assert "returns" in result.columns
        assert "volatility" in result.columns

    def test_feature_scaling(self):
        """Test feature scaling."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        })

        preprocessor = Preprocessor()
        scaled = preprocessor.scale_features(df, fit=True)

        assert scaled.shape == (5, 2)
        assert np.allclose(scaled.mean(axis=0), [0, 0], atol=1e-10)

    def test_missing_value_handling(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "gad7_score": [10, np.nan, 15],
            "age": [35, 45, np.nan],
            "gender": ["M", "F", "M"],
            "clinical_deterioration": [0, 1, 0],
        })

        preprocessor = Preprocessor()
        result = preprocessor.preprocess_health_data(df, fit=True)

        # Check no NaN values in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0
