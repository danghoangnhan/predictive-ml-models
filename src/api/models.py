"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class HealthPredictionRequest(BaseModel):
    """Health prediction request."""
    gad7_score: int = Field(..., ge=0, le=21, description="GAD-7 score (0-21)")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    gender: str = Field(..., description="M or F")
    bmi: Optional[float] = None
    sleep_hours: Optional[float] = None


class HealthPredictionResponse(BaseModel):
    """Health prediction response."""
    prediction: int
    probability: float
    risk_level: str
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None


class StockPatternRequest(BaseModel):
    """Stock pattern detection request."""
    symbol: str = Field(..., description="Stock symbol")
    lookback_days: int = Field(default=60, ge=20)


class StockPatternResponse(BaseModel):
    """Stock pattern response."""
    pattern: str
    confidence: Dict[str, float]
    timestamp: datetime


class StockForecastRequest(BaseModel):
    """Stock forecast request."""
    symbol: str = Field(..., description="Stock symbol")
    horizon: int = Field(default=30, ge=1, le=365)


class StockForecastResponse(BaseModel):
    """Stock forecast response."""
    forecast: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    horizon: int
    timestamp: datetime


class HealthBatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_samples: int
    successful_predictions: int
    failed_samples: int
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    has_explainer: bool
    has_preprocessor: bool
    timestamp: datetime


class DriftDetectionResponse(BaseModel):
    """Drift detection response."""
    drift_detected: bool
    threshold: float
    drifted_features: List[tuple]
    psi_scores: Dict[str, float]
    num_samples: int
