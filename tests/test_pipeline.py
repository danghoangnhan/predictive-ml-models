import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import DataLoader
from data.preprocessor import HealthcareFeatureEngineering
from data.splitter import DataSplitter
from models.health_predictor import HealthcarePredictor


def test_full_healthcare_pipeline(tmp_path):
    """Test end-to-end healthcare pipeline."""
    
    # Create sample data
    data = {
        'patient_id': [f'P{i:03d}' for i in range(1, 51)],
        'gad7_score': np.random.randint(5, 21, 50),
        'journal_text': ['test'] * 50,
        'timestamp': pd.date_range(start='2024-01-01', periods=50),
        'label': np.random.randint(0, 2, 50)
    }
    df = pd.DataFrame(data)
    
    # Save to temp file
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Load data
    loaded_df = DataLoader.load_csv(str(csv_path))
    assert len(loaded_df) == 50
    
    # Feature engineering
    feature_df = HealthcareFeatureEngineering.engineer_all_features(loaded_df)
    assert feature_df.shape[0] == 50
    
    # Prepare features
    feature_cols = [col for col in feature_df.columns if col not in ['patient_id', 'journal_text', 'timestamp', 'label']]
    X = feature_df[feature_cols]
    y = feature_df['label']
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    
    # Train model
    model = HealthcarePredictor()
    model.train(X_train, y_train)
    
    # Make predictions
    predictions, proba = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert model.is_trained
