"""Model evaluation script."""

import argparse
import logging
import sys

from src.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.evaluation import DriftDetector, ModelMetrics
from src.models import HealthPredictor

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_health_model(args):
    """Evaluate health model."""
    logger.info("Starting health model evaluation...")

    # Load data
    loader = DataLoader(config.SAMPLE_DATA_PATH)
    df = loader.load_health_data()

    if len(df) == 0:
        logger.error("No health data found")
        return False

    # Prepare features
    feature_cols = [
        col for col in df.columns if col not in ["patient_id", "clinical_deterioration"]
    ]
    X = df[feature_cols]
    y = df["clinical_deterioration"].values

    # Preprocess
    preprocessor = Preprocessor()
    X_processed = preprocessor.preprocess_health_data(X, fit=True)

    # Try to load trained model
    try:
        model = HealthPredictor.load(config.HEALTH_MODEL_PATH)
    except FileNotFoundError:
        logger.warning(f"Model not found at {config.HEALTH_MODEL_PATH}, training new model")
        model = HealthPredictor(model_type="xgboost")
        model.fit(X_processed, y)

    # Make predictions
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)

    # Calculate metrics
    metrics = ModelMetrics.classification_metrics(y, y_pred, y_proba)
    logger.info(f"Classification metrics: {metrics}")

    # Confusion matrix
    cm_report = ModelMetrics.confusion_matrix_report(y, y_pred)
    logger.info(f"Confusion matrix:\n{cm_report['confusion_matrix']}")

    # Feature importance
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
        logger.info(f"Top 5 features: {dict(list(importance.items())[:5])}")

    # Drift detection
    drift_detector = DriftDetector(threshold=config.DRIFT_ALERT_THRESHOLD)
    drift_detector.fit_baseline(X_processed)
    drift_report = drift_detector.detect_drift(X_processed)
    logger.info(f"Drift detection: {drift_report['drift_detected']}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ML models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["health", "stock"],
        default="health",
        help="Model to evaluate",
    )

    args = parser.parse_args()

    if args.model == "health":
        if not evaluate_health_model(args):
            sys.exit(1)
    else:
        logger.warning(f"Evaluation not implemented for {args.model}")

    logger.info("Evaluation completed")
    sys.exit(0)


if __name__ == "__main__":
    main()
