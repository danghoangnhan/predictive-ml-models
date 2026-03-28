from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class HealthPredictionRequest(BaseModel):
    """Request model for health prediction."""
    patient_id: str
    gad7_score: int = Field(..., ge=0, le=21)
    journal_text: str
    days_since_last_assessment: int = 7
    history: Optional[List[int]] = None


class HealthPredictionResponse(BaseModel):
    """Response model for health prediction."""
    prediction: int
    risk_score: float
    confidence: float
    explanation: Optional[Dict[str, Any]] = None


class PatternPredictionRequest(BaseModel):
    """Request model for pattern prediction."""
    symbol: str
    ohlcv: List[List[float]]
    pattern_lookback_days: int = 20


class PatternPredictionResponse(BaseModel):
    """Response model for pattern prediction."""
    pattern: str
    confidence: float
    breakout_probability: Optional[float] = None
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


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
    models_loaded: List[str]
    uptime_seconds: int
    timestamp: str
