import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessor import FinanceFeatureEngineering, HealthcareFeatureEngineering


@pytest.fixture
def sample_health_df():
    """Create sample health data."""
    dates = [datetime.now() - timedelta(days=i) for i in range(10)]
    return pd.DataFrame(
        {
            "patient_id": ["P001"] * 10,
            "gad7_score": [10, 12, 15, 14, 16, 18, 17, 19, 20, 18],
            "journal_text": ["text"] * 10,
            "timestamp": dates,
            "label": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def sample_finance_df():
    """Create sample finance data."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 10,
            "date": pd.date_range(start="2024-01-01", periods=10),
            "open": np.random.uniform(100, 110, 10),
            "high": np.random.uniform(110, 120, 10),
            "low": np.random.uniform(90, 100, 10),
            "close": np.random.uniform(100, 110, 10),
            "volume": np.random.uniform(1000000, 10000000, 10),
        }
    )


def test_healthcare_temporal_features(sample_health_df):
    """Test temporal feature extraction."""
    df = HealthcareFeatureEngineering.extract_temporal_features(sample_health_df)

    assert "day_of_week" in df.columns
    assert "week_of_year" in df.columns
    assert "day_of_month" in df.columns


def test_healthcare_trend_features(sample_health_df):
    """Test trend feature extraction."""
    df = HealthcareFeatureEngineering.extract_temporal_features(sample_health_df)
    df = HealthcareFeatureEngineering.extract_trend_features(df)

    assert "gad7_7day_ma" in df.columns
    assert "gad7_trend" in df.columns
    assert "gad7_volatility" in df.columns


def test_healthcare_nlp_features(sample_health_df):
    """Test NLP feature extraction."""
    df = HealthcareFeatureEngineering.extract_nlp_features(sample_health_df)

    assert "journal_sentiment" in df.columns
    assert "journal_anxiety_keywords_count" in df.columns
    assert "journal_length" in df.columns


def test_finance_candlestick_features(sample_finance_df):
    """Test candlestick feature extraction."""
    df = FinanceFeatureEngineering.extract_candlestick_features(sample_finance_df)

    assert "body_ratio" in df.columns
    assert "upper_wick_ratio" in df.columns
    assert "lower_wick_ratio" in df.columns


def test_finance_technical_features(sample_finance_df):
    """Test technical feature extraction."""
    df = FinanceFeatureEngineering.extract_technical_features(sample_finance_df)

    assert "atr" in df.columns
    assert "bollinger_upper" in df.columns
    assert "bollinger_lower" in df.columns
    assert "volume_sma_20" in df.columns
