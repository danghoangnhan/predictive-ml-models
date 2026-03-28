"""Generate evaluation and drift detection report."""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

from src.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models import HealthPredictor
from src.evaluation import ModelMetrics, DriftDetector

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_report(args):
    """Generate comprehensive evaluation report."""
    logger.info("Generating evaluation report...")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "app_version": config.APP_VERSION,
        "models": {},
        "drift_detection": {},
    }

    # Load health data
    loader = DataLoader(config.SAMPLE_DATA_PATH)
    df_health = loader.load_health_data()

    if len(df_health) > 0:
        # Prepare features
        feature_cols = [
            col
            for col in df_health.columns
            if col not in ["patient_id", "clinical_deterioration"]
        ]
        X = df_health[feature_cols]
        y = df_health["clinical_deterioration"].values

        # Preprocess
        preprocessor = Preprocessor()
        X_processed = preprocessor.preprocess_health_data(X, fit=True)

        # Load or train model
        try:
            model = HealthPredictor.load(config.HEALTH_MODEL_PATH)
        except FileNotFoundError:
            logger.warning("Model not found, training new model")
            model = HealthPredictor(model_type="xgboost")
            model.fit(X_processed, y)

        # Evaluate
        y_pred = model.predict(X_processed)
        y_proba = model.predict_proba(X_processed)
        metrics = ModelMetrics.classification_metrics(y, y_pred, y_proba)

        report["models"]["health"] = {
            "status": "trained",
            "algorithm": "xgboost",
            "metrics": metrics,
            "model_path": str(config.HEALTH_MODEL_PATH),
        }

        # Drift detection
        drift_detector = DriftDetector(threshold=config.DRIFT_ALERT_THRESHOLD)
        drift_detector.fit_baseline(X_processed)
        drift_report = drift_detector.detect_drift(X_processed)

        report["drift_detection"]["health"] = {
            "drift_detected": drift_report["drift_detected"],
            "psi_scores": {k: float(v) for k, v in drift_report["psi_scores"].items()},
            "drifted_features": [(f, float(s)) for f, s in drift_report["drifted_features"]],
        }

    logger.info("Report generation completed")

    # Save report
    output_path = Path(args.output or "report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 60)

    for model_name, model_info in report["models"].items():
        print(f"\n{model_name.upper()} MODEL")
        print(f"  Status: {model_info.get('status', 'unknown')}")
        if "metrics" in model_info:
            for metric, value in model_info["metrics"].items():
                print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 60)

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--drift-check", action="store_true", help="Include drift detection")

    args = parser.parse_args()

    if not generate_report(args):
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
