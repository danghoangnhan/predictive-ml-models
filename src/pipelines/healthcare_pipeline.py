"""End-to-end healthcare prediction pipeline."""

import logging

from data.loader import load_and_validate_health_data
from data.preprocessor import HealthcareFeatureEngineering
from data.splitter import DataSplitter
from models.health_predictor import HealthcarePredictor

logger = logging.getLogger(__name__)


class HealthcarePipeline:
    """End-to-end healthcare prediction pipeline."""

    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load_data(self, filepath: str):
        """Load and validate health data."""
        return load_and_validate_health_data(filepath)

    def preprocess(self, df):
        """Apply feature engineering."""
        return HealthcareFeatureEngineering.engineer_all_features(df)

    def train(self, data_path: str, test_ratio: float = 0.15):
        """Train the pipeline."""
        logger.info("Starting healthcare pipeline training...")

        # Load data
        df = self.load_data(data_path)

        # Preprocess
        df = self.preprocess(df)

        # Prepare features
        feature_cols = [
            col
            for col in df.columns
            if col not in ["patient_id", "journal_text", "timestamp", "label", "target"]
        ]
        X = df[feature_cols]
        y = df.get("label", df.get("target", None))

        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)

        # Train model
        self.model = HealthcarePredictor()
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
