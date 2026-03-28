"""Model serving and prediction services."""

from .predictor import Predictor
from .batch_predictor import BatchPredictor

__all__ = ["Predictor", "BatchPredictor"]
