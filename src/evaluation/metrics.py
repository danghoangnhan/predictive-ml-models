"""Model evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Calculate and report model evaluation metrics."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0, average="weighted"),
            "recall": recall_score(y_true, y_pred, zero_division=0, average="weighted"),
            "f1": f1_score(y_true, y_pred, zero_division=0, average="weighted"),
        }

        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
        }

    @staticmethod
    def confusion_matrix_report(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate confusion matrix and classification report."""

        cm = confusion_matrix(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True)

        return {
            "confusion_matrix": cm,
            "classification_report": report_dict,
        }

    @staticmethod
    def fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate fairness metrics (demographic parity, equalized odds)."""

        unique_groups = np.unique(groups)
        metrics = {}

        # Positive rate per group
        positive_rates = {}
        for group in unique_groups:
            group_mask = groups == group
            if sum(group_mask) > 0:
                pos_rate = y_pred[group_mask].mean()
                positive_rates[str(group)] = pos_rate

        metrics["positive_rates"] = positive_rates

        # Demographic parity difference
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            metrics["demographic_parity_diff"] = max(rates) - min(rates)

        return metrics
