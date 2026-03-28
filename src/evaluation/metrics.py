import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, auc
)
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Compute classification metrics."""
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
        """Compute comprehensive classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        
        return metrics


class RegressionMetrics:
    """Compute regression metrics."""
    
    @staticmethod
    def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error."""
        return np.sqrt(ClassificationMetrics.compute_mse(y_true, y_pred))
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute percentage error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute comprehensive regression metrics."""
        from sklearn.metrics import r2_score
        
        return {
            'mse': RegressionMetrics.compute_mse(y_true, y_pred),
            'rmse': RegressionMetrics.compute_rmse(y_true, y_pred),
            'mae': RegressionMetrics.compute_mae(y_true, y_pred),
            'mape': RegressionMetrics.compute_mape(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
