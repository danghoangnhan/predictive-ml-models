"""FastAPI routes for predictions."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import logging

from .models import (
    HealthPredictionRequest,
    HealthPredictionResponse,
    StockPatternRequest,
    StockPatternResponse,
    HealthBatchPredictionResponse,
    ModelInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])

# These will be injected by main.py
health_predictor = None
stock_predictor = None


def set_predictors(health_pred, stock_pred):
    """Set global predictor instances."""
    global health_predictor, stock_predictor
    health_predictor = health_pred
    stock_predictor = stock_pred


@router.post("/predict/health", response_model=HealthPredictionResponse)
async def predict_health(request: HealthPredictionRequest, explain: bool = False):
    """Predict health deterioration."""
    if health_predictor is None:
        raise HTTPException(status_code=503, detail="Health model not loaded")

    try:
        result = health_predictor.predict_health(
            request.dict(), explain=explain
        )
        return HealthPredictionResponse(**result)
    except Exception as e:
        logger.error(f"Health prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/health/batch")
async def predict_health_batch(file: UploadFile = File(...)):
    """Batch health predictions from CSV."""
    if health_predictor is None:
        raise HTTPException(status_code=503, detail="Health model not loaded")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode("utf8")))

        results = health_predictor.model.predict(df)
        probas = health_predictor.model.predict_proba(df)

        return HealthBatchPredictionResponse(
            total_samples=len(df),
            successful_predictions=len(results),
            failed_samples=0,
            predictions=results.tolist(),
            probabilities=probas.tolist(),
        )
    except Exception as e:
        logger.error(f"Batch health prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/stock")
async def predict_stock(request: StockPatternRequest):
    """Predict stock chart pattern."""
    if stock_predictor is None:
        raise HTTPException(status_code=503, detail="Stock model not loaded")

    try:
        # This would normally fetch real data
        result = {
            "pattern": "unknown",
            "confidence": {},
            "timestamp": "2026-01-01T00:00:00"
        }
        return result
    except Exception as e:
        logger.error(f"Stock prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/health", response_model=ModelInfoResponse)
async def get_health_model_info():
    """Get health model information."""
    if health_predictor is None:
        raise HTTPException(status_code=503, detail="Health model not loaded")

    return health_predictor.get_model_info()


@router.get("/health/status")
async def health_status():
    """API health check."""
    return {"status": "ok", "models_loaded": health_predictor is not None}
