"""End-to-end finance pattern detection pipeline."""

import logging
from pathlib import Path
from data.loader import load_and_validate_finance_data
from data.preprocessor import FinanceFeatureEngineering
from data.splitter import DataSplitter
from models.pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


class FinancePipeline:
    """End-to-end finance pattern detection pipeline."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
    
    def load_data(self, filepath: str):
        """Load and validate finance data."""
        return load_and_validate_finance_data(filepath)
    
    def preprocess(self, df):
        """Apply feature engineering."""
        return FinanceFeatureEngineering.engineer_all_features(df)
    
    def train(self, data_path: str):
        """Train the pipeline."""
        logger.info("Starting finance pipeline training...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess
        df = self.preprocess(df)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'label', 'target']]
        X = df[feature_cols]
        y = df.get('label', df.get('target', None))
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)
        
        # Train model
        self.model = PatternDetector(model_type='xgboost')
        self.model.train(X_train, y_train)
        
        # Evaluate
        metrics = self.model.evaluate(X_test, y_test)
        logger.info(f"Pipeline training complete. Metrics: {metrics}")
        
        return metrics
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
