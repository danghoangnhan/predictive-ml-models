from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
from .models import (
    HealthPredictionRequest, HealthPredictionResponse,
    PatternPredictionRequest, PatternPredictionResponse,
    TrainingRequest, TrainingResponse,
    HealthCheckResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global model instances
health_model = None
pattern_model = None
start_time = datetime.now()


def set_models(health, pattern):
    """Set the model instances."""
    global health_model, pattern_model
    health_model = health
    pattern_model = pattern


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - start_time).total_seconds()
    models_loaded = []
    
    if health_model:
        models_loaded.append("healthcare_ensemble")
    if pattern_model:
        models_loaded.append("finance_pattern_detector")
    
    return HealthCheckResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        uptime_seconds=int(uptime),
        timestamp=datetime.now().isoformat()
    )


@router.post("/predict/health", response_model=HealthPredictionResponse)
async def predict_health(request: HealthPredictionRequest):
    """Predict health deterioration risk."""
    try:
        if health_model is None:
            raise HTTPException(status_code=503, detail="Health model not loaded")
        
        # Make prediction
        import pandas as pd
        import numpy as np
        
        # Prepare features (simplified - would use full feature engineering)
        features = pd.DataFrame([{
            'gad7_score': request.gad7_score,
            'days_since_last_assessment': request.days_since_last_assessment,
            'journal_length': len(request.journal_text.split())
        }])
        
        prediction, confidence = health_model.predict(features)
        
        return HealthPredictionResponse(
            prediction=int(prediction[0]),
            risk_score=float(confidence[0]) if confidence is not None else 0.0,
            confidence=float(confidence[0]) if confidence is not None else 0.0,
            explanation={"message": "Prediction generated"}
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/pattern", response_model=PatternPredictionResponse)
async def predict_pattern(request: PatternPredictionRequest):
    """Predict stock chart patterns."""
    try:
        if pattern_model is None:
            raise HTTPException(status_code=503, detail="Pattern model not loaded")
        
        # Make prediction
        import pandas as pd
        import numpy as np
        
        # Prepare features from OHLCV
        ohlcv_array = np.array(request.ohlcv)
        features = pd.DataFrame({
            'open': ohlcv_array[:, 0],
            'high': ohlcv_array[:, 1],
            'low': ohlcv_array[:, 2],
            'close': ohlcv_array[:, 3],
            'volume': ohlcv_array[:, 4]
        })
        
        prediction, confidence = pattern_model.predict_with_confidence(features)
        
        pattern_name = pattern_model.get_pattern_name(prediction[0])
        
        return PatternPredictionResponse(
            pattern=pattern_name,
            confidence=float(confidence[0]),
            breakout_probability=float(confidence[0]) * 0.9,
            support_level=float(ohlcv_array[:, 2].min()),
            resistance_level=float(ohlcv_array[:, 1].max())
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Trigger model retraining."""
    try:
        logger.info(f"Training triggered for {request.domain}")
        
        # This would trigger an async training job
        return TrainingResponse(
            status="training_started",
            job_id="job_12345",
            estimated_duration_seconds=120
        )
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
