"""Model serving and prediction services."""

from .batch_predictor import BatchPredictor
from .predictor import Predictor

__all__ = ["Predictor", "BatchPredictor"]
