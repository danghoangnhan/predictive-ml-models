"""Prediction script."""

import argparse
import json
import logging
from pathlib import Path

from src.config import config
from src.models import HealthPredictor
from src.serving import BatchPredictor, Predictor

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def predict_single(args):
    """Make single prediction."""
    logger.info("Loading model...")

    try:
        model = HealthPredictor.load(config.HEALTH_MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model not found at {config.HEALTH_MODEL_PATH}")
        return False

    predictor = Predictor(model)

    # Create sample patient data
    patient_data = {
        "gad7_score": args.gad7_score,
        "age": args.age,
        "gender": args.gender,
        "bmi": args.bmi or 24.5,
        "sleep_hours": args.sleep_hours or 7.0,
    }

    logger.info(f"Input data: {patient_data}")

    result = predictor.predict_health(patient_data, explain=args.explain)

    logger.info("Prediction result:")
    logger.info(json.dumps(result, indent=2, default=str))

    return True


def predict_batch(args):
    """Make batch predictions."""
    logger.info("Loading model...")

    try:
        model = HealthPredictor.load(config.HEALTH_MODEL_PATH)
    except FileNotFoundError:
        logger.error(f"Model not found at {config.HEALTH_MODEL_PATH}")
        return False

    batch_predictor = BatchPredictor(model)

    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return False

    logger.info(f"Loading data from {csv_path}")

    results = batch_predictor.predict_from_csv(csv_path)

    logger.info(
        f"Batch prediction completed: {results['successful_predictions']}/{results['total_samples']}"
    )

    # Save results
    output_path = Path(args.output or "predictions_output.csv")
    batch_predictor.save_predictions(results, output_path)
    logger.info(f"Results saved to {output_path}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Make predictions")
    subparsers = parser.add_subparsers(dest="mode", help="Prediction mode")

    # Single prediction
    single_parser = subparsers.add_parser("single", help="Single prediction")
    single_parser.add_argument("--gad7-score", type=int, required=True, help="GAD-7 score")
    single_parser.add_argument("--age", type=int, required=True, help="Age")
    single_parser.add_argument("--gender", type=str, required=True, help="Gender (M/F)")
    single_parser.add_argument("--bmi", type=float, help="BMI")
    single_parser.add_argument("--sleep-hours", type=float, help="Sleep hours")
    single_parser.add_argument("--explain", action="store_true", help="Show explanation")

    # Batch prediction
    batch_parser = subparsers.add_parser("batch", help="Batch prediction")
    batch_parser.add_argument("--csv-file", type=str, required=True, help="CSV file path")
    batch_parser.add_argument("--output", type=str, help="Output CSV file")

    args = parser.parse_args()

    if args.mode == "single":
        if not predict_single(args):
            return 1
    elif args.mode == "batch":
        if not predict_batch(args):
            return 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
