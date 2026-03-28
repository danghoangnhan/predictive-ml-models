"""Batch prediction service."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction service for multiple samples."""

    def __init__(self, model, preprocessor=None):
        """Initialize batch predictor."""
        self.model = model
        self.preprocessor = preprocessor

    def predict_batch(
        self, X: pd.DataFrame, batch_size: int = 32, return_proba: bool = True
    ) -> Dict[str, Any]:
        """Run batch predictions."""

        predictions = []
        probabilities = []
        errors = []

        for i in range(0, len(X), batch_size):
            batch = X.iloc[i : i + batch_size]

            try:
                # Preprocess if available
                if self.preprocessor:
                    batch_processed = self.preprocessor.preprocess_health_data(
                        batch, fit=False
                    )
                else:
                    batch_processed = batch

                # Predict
                batch_preds = self.model.predict(batch_processed)
                predictions.extend(batch_preds.tolist())

                if return_proba:
                    batch_proba = self.model.predict_proba(batch_processed)
                    probabilities.extend(batch_proba.tolist())

            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                errors.append(
                    {
                        "batch_idx": i // batch_size + 1,
                        "error": str(e),
                    }
                )

        result = {
            "total_samples": len(X),
            "successful_predictions": len(predictions),
            "failed_samples": len(errors),
            "predictions": predictions,
        }

        if probabilities:
            result["probabilities"] = probabilities

        if errors:
            result["errors"] = errors

        logger.info(
            f"Batch prediction complete: {len(predictions)}/{len(X)} successful"
        )

        return result

    def predict_from_csv(
        self, csv_path: Path, batch_size: int = 32
    ) -> Dict[str, Any]:
        """Load CSV and run batch predictions."""

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")

        # Run predictions
        return self.predict_batch(df, batch_size=batch_size)

    def save_predictions(
        self, predictions: Dict[str, Any], output_path: Path
    ) -> None:
        """Save predictions to CSV."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_results = pd.DataFrame({
            "prediction": predictions["predictions"],
        })

        if "probabilities" in predictions:
            df_results["probability"] = [p[1] for p in predictions["probabilities"]]

        df_results.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")

    def get_batch_statistics(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from batch predictions."""

        preds = np.array(predictions["predictions"])
        stats = {
            "total_predictions": len(preds),
            "positive_class_count": int((preds == 1).sum()),
            "negative_class_count": int((preds == 0).sum()),
            "positive_class_pct": float((preds == 1).mean() * 100),
        }

        if "probabilities" in predictions:
            probas = np.array(predictions["probabilities"])
            stats["mean_probability"] = float(probas.mean())
            stats["std_probability"] = float(probas.std())

        return stats
