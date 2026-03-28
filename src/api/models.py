from typing import Any

from pydantic import BaseModel, Field


class HealthPredictionRequest(BaseModel):
    """Request model for health prediction."""

    patient_id: str
    gad7_score: int = Field(..., ge=0, le=21)
    journal_text: str
    days_since_last_assessment: int = 7
    history: list[int] | None = None


class HealthPredictionResponse(BaseModel):
    """Response model for health prediction."""

    prediction: int
    risk_score: float
    confidence: float
    explanation: dict[str, Any] | None = None


class PatternPredictionRequest(BaseModel):
    """Request model for pattern prediction."""

    symbol: str
    ohlcv: list[list[float]]
    pattern_lookback_days: int = 20


class PatternPredictionResponse(BaseModel):
    """Response model for pattern prediction."""

    pattern: str
    confidence: float
    breakout_probability: float | None = None
    support_level: float | None = None
    resistance_level: float | None = None


class TrainingRequest(BaseModel):
    """Request model for model training."""

    domain: str  # healthcare or finance
    data_path: str
    model_type: str = "ensemble"


class TrainingResponse(BaseModel):
    """Response model for training."""

    status: str
    job_id: str
    estimated_duration_seconds: int


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: list[str]
    uptime_seconds: int
    timestamp: str
