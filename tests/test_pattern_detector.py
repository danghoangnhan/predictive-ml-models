import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pattern_detector import PatternDetector
from data.splitter import DataSplitter


@pytest.fixture
def sample_pattern_data():
    """Create sample finance data."""
    X = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000000, 10000000, 100),
        'atr': np.random.uniform(1, 5, 100),
        'bollinger_width': np.random.uniform(5, 15, 100),
    })
    y = pd.Series(np.random.randint(0, 4, 100))
    return X, y


def test_pattern_detector_initialization():
    """Test detector initialization."""
    model = PatternDetector()
    assert model.name == "PatternDetector"
    assert not model.is_trained


def test_pattern_detector_train(sample_pattern_data):
    """Test model training."""
    X, y = sample_pattern_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)
    
    model = PatternDetector()
    model.train(X_train, y_train)
    
    assert model.is_trained
    assert model.feature_names is not None


def test_pattern_detector_predict(sample_pattern_data):
    """Test prediction."""
    X, y = sample_pattern_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)
    
    model = PatternDetector()
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1, 2, 3] for pred in predictions)


def test_pattern_detector_predict_proba(sample_pattern_data):
    """Test probability predictions."""
    X, y = sample_pattern_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)
    
    model = PatternDetector()
    model.train(X_train, y_train)
    
    proba = model.predict_proba(X_test)
    
    assert proba.shape == (len(X_test), 4)


def test_pattern_detector_evaluate(sample_pattern_data):
    """Test model evaluation."""
    X, y = sample_pattern_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)
    
    model = PatternDetector()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
