import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect model and data drift."""

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions."""
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def kolmogorov_smirnov_test(
        data_train: np.ndarray, data_test: np.ndarray, threshold: float = 0.05
    ) -> bool:
        """
        Kolmogorov-Smirnov test for distribution shift.
        Returns True if drift detected (p-value < threshold).
        """
        stat, p_value = stats.ks_2samp(data_train, data_test)
        logger.info(f"KS test: stat={stat:.4f}, p-value={p_value:.4f}")

        if p_value < threshold:
            logger.warning(f"Drift detected: p-value {p_value:.4f} < {threshold}")
            return True

        return False

    @staticmethod
    def detect_prediction_drift(
        y_train: np.ndarray, y_pred: np.ndarray, threshold: float = 0.05
    ) -> bool:
        """Detect drift in predictions."""
        return DriftDetector.kolmogorov_smirnov_test(y_train, y_pred, threshold)

    @staticmethod
    def detect_feature_drift(
        X_train: pd.DataFrame, X_current: pd.DataFrame, threshold: float = 0.05
    ) -> dict:
        """Detect drift in input features."""
        drift_detected = {}

        for col in X_train.columns:
            ks_stat, p_value = stats.ks_2samp(X_train[col].dropna(), X_current[col].dropna())
            drift_detected[col] = {
                "stat": ks_stat,
                "p_value": p_value,
                "drifted": p_value < threshold,
            }

        return drift_detected


class ModelMonitor:
    """Monitor model performance over time."""

    def __init__(self):
        self.predictions_log = []
        self.metrics_log = []

    def log_prediction(self, y_true: np.ndarray, y_pred: np.ndarray, timestamp: str = None):
        """Log prediction results."""
        self.predictions_log.append({"timestamp": timestamp, "y_true": y_true, "y_pred": y_pred})

    def log_metrics(self, metrics: dict, timestamp: str = None):
        """Log metrics."""
        metrics["timestamp"] = timestamp
        self.metrics_log.append(metrics)

    def get_metrics_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        return pd.DataFrame(self.metrics_log)

    def check_performance_degradation(
        self, current_metrics: dict, baseline_metrics: dict, threshold: float = 0.05
    ) -> bool:
        """Check if performance degraded significantly."""
        key_metrics = ["accuracy", "f1", "auc_roc"]

        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                degradation = (
                    baseline_metrics[metric] - current_metrics[metric]
                ) / baseline_metrics[metric]

                if degradation > threshold:
                    logger.warning(f"Performance degradation in {metric}: {degradation:.2%}")
                    return True

        return False


class RetrainingTrigger:
    """Determine when models should be retrained."""

    def __init__(self, drift_threshold: float = 0.05, perf_threshold: float = 0.05):
        self.drift_threshold = drift_threshold
        self.perf_threshold = perf_threshold

    def should_retrain(
        self,
        drift_detected: bool,
        perf_degraded: bool,
        time_since_training: int = None,
        days_threshold: int = 30,
    ) -> bool:
        """Determine if model should be retrained."""

        if drift_detected:
            logger.info("Retraining triggered: drift detected")
            return True

        if perf_degraded:
            logger.info("Retraining triggered: performance degradation")
            return True

        if time_since_training and time_since_training > days_threshold:
            logger.info(
                f"Retraining triggered: {time_since_training} days since training (threshold: {days_threshold})"
            )
            return True

        return False
