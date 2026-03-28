#!/usr/bin/env python
"""Training script for ML models."""

import argparse
import logging
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import settings
from data.loader import load_and_validate_finance_data, load_and_validate_health_data
from data.preprocessor import FinanceFeatureEngineering, HealthcareFeatureEngineering
from data.splitter import DataSplitter
from models.health_predictor import HealthcarePredictor
from models.pattern_detector import PatternDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_healthcare_model(data_path: str, output_path: str):
    """Train healthcare predictor."""
    logger.info("Training healthcare model...")

    # Load data
    df = load_and_validate_health_data(data_path)
    if df is None:
        logger.error("Failed to load health data")
        return

    # Feature engineering
    df = HealthcareFeatureEngineering.engineer_all_features(df)

    # Prepare features and target
    feature_cols = [
        col
        for col in df.columns
        if col not in ["patient_id", "journal_text", "timestamp", "label", "target"]
    ]
    X = df[feature_cols]
    y = df.get("label", df.get("target", None))

    if y is None:
        logger.warning("No target column found. Using random labels for demo.")
        import numpy as np

        y = np.random.randint(0, 2, len(X))

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.stratified_split(X, y)

    # Train model
    model = HealthcarePredictor(ensemble_models=["logistic_regression", "random_forest"])
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Healthcare model metrics: {metrics}")

    # Save model
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")


def train_finance_model(data_path: str, output_path: str):
    """Train pattern detector model."""
    logger.info("Training finance pattern detection model...")

    # Load data
    df = load_and_validate_finance_data(data_path)
    if df is None:
        logger.error("Failed to load finance data")
        return

    # Feature engineering
    df = FinanceFeatureEngineering.engineer_all_features(df)

    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ["symbol", "date", "label", "target"]]
    X = df[feature_cols]
    y = df.get("label", df.get("target", None))

    if y is None:
        logger.warning("No target column found. Using random labels for demo.")
        import numpy as np

        y = np.random.randint(0, 4, len(X))

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.time_series_split(X, y)

    # Train model
    model = PatternDetector(model_type="xgboost")
    model.train(X_train, y_train)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Finance model metrics: {metrics}")

    # Save model
    model.save(output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--domain", choices=["healthcare", "finance"], required=True, help="Domain to train"
    )
    parser.add_argument("--data-path", help="Path to training data")
    parser.add_argument("--output-path", help="Path to save model")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    if args.domain == "healthcare":
        data_path = args.data_path or str(settings.SAMPLE_DATA_PATH / "health_scores.csv")
        output_path = args.output_path or str(settings.MODELS_DIR / "healthcare_ensemble.pkl")
        train_healthcare_model(data_path, output_path)

    elif args.domain == "finance":
        data_path = args.data_path or str(settings.SAMPLE_DATA_PATH / "stock_patterns.csv")
        output_path = args.output_path or str(settings.MODELS_DIR / "finance_xgboost.pkl")
        train_finance_model(data_path, output_path)


if __name__ == "__main__":
    main()
