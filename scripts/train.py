"""Model training script."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.training.trainer import Trainer
from src.models import HealthPredictor, PatternDetector

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_health_model(args):
    """Train health prediction model."""
    logger.info("Starting health model training...")

    # Load data
    loader = DataLoader(config.SAMPLE_DATA_PATH)
    df = loader.load_health_data()

    if len(df) == 0:
        logger.error("No health data found")
        return False

    if not loader.validate_health_data(df):
        logger.error("Health data validation failed")
        return False

    # Prepare features
    feature_cols = [col for col in df.columns if col not in ["patient_id", "clinical_deterioration"]]
    X = df[feature_cols]
    y = df["clinical_deterioration"]

    # Preprocess
    preprocessor = Preprocessor()
    X_processed = preprocessor.preprocess_health_data(X, fit=True)

    # Train
    trainer = Trainer(model_type="health")
    history = trainer.train_health_model(
        X_processed,
        y,
        algorithm=args.algorithm,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    logger.info(f"Training completed: {history}")

    # Save model
    model = trainer.get_model()
    model.save(config.HEALTH_MODEL_PATH)
    logger.info(f"Model saved to {config.HEALTH_MODEL_PATH}")

    return True


def train_stock_model(args):
    """Train stock pattern model."""
    logger.info("Starting stock pattern model training...")

    # Load data
    loader = DataLoader(config.SAMPLE_DATA_PATH)
    df = loader.load_stock_data()

    if len(df) == 0:
        logger.error("No stock data found")
        return False

    if not loader.validate_stock_data(df):
        logger.error("Stock data validation failed")
        return False

    # Preprocess
    preprocessor = Preprocessor()
    df_processed = preprocessor.preprocess_stock_data(df, fit=True)

    # Extract features
    detector = PatternDetector()
    features = detector.extract_pattern_features(df_processed)

    if len(features) == 0:
        logger.error("No features extracted")
        return False

    # Get target
    if "pattern" in df_processed.columns:
        y = df_processed.loc[features.index, "pattern"]
    else:
        logger.error("Pattern column not found")
        return False

    # Train
    trainer = Trainer(model_type="stock")
    history = trainer.train_stock_pattern_model(features, y, test_size=args.test_size)

    logger.info(f"Training completed: {history}")

    # Save model
    model = trainer.get_model()
    model.save(config.STOCK_MODEL_PATH)
    logger.info(f"Model saved to {config.STOCK_MODEL_PATH}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["health", "stock", "all"],
        default="health",
        help="Model to train",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["xgboost", "random_forest"],
        default="xgboost",
        help="Algorithm for health model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size",
    )

    args = parser.parse_args()

    success = True

    if args.model in ["health", "all"]:
        if not train_health_model(args):
            success = False

    if args.model in ["stock", "all"]:
        if not train_stock_model(args):
            success = False

    if success:
        logger.info("All training completed successfully")
        sys.exit(0)
    else:
        logger.error("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
