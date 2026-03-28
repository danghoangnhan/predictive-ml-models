import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.health_predictor import HealthcarePredictor
from data.splitter import DataSplitter


@pytest.fixture
def sample_health_data():
    """Create sample health data."""
    X = pd.DataFrame({
        'gad7_score': np.random.randint(5, 21, 100),
        'days_since_last_assessment': np.random.randint(1, 30, 100),
        'journal_length': np.random.randint(10, 500, 100),
        'gad7_7day_ma': np.random.uniform(5, 20, 100),
        'gad7_trend': np.random.uniform(-5, 5, 100),
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


def test_healthcare_predictor_initialization():
    """Test predictor initialization."""
    model = HealthcarePredictor()
    assert model.name == "HealthcarePredictor"
    assert model.model_type == "ensemble"
    assert not model.is_trained


def test_healthcare_predictor_train(sample_health_data):
    """Test model training."""
    X, y = sample_health_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)
    
    model = HealthcarePredictor()
    model.train(X_train, y_train)
    
    assert model.is_trained
    assert model.feature_names is not None


def test_healthcare_predictor_predict(sample_health_data):
    """Test prediction."""
    X, y = sample_health_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)
    
    model = HealthcarePredictor()
    model.train(X_train, y_train)
    
    predictions, probabilities = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert len(probabilities) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)


def test_healthcare_predictor_evaluate(sample_health_data):
    """Test model evaluation."""
    X, y = sample_health_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)
    
    model = HealthcarePredictor()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'auc_roc' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics


def test_healthcare_predictor_feature_importance(sample_health_data):
    """Test feature importance extraction."""
    X, y = sample_health_data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)
    
    model = HealthcarePredictor()
    model.train(X_train, y_train)
    
    importance = model.get_feature_importance()
    
    assert importance is not None
    assert len(importance) > 0
