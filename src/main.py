"""FastAPI application for predictive ML models."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from src.config import config
from src.api.routes import router, set_predictors
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models import HealthPredictor
from src.serving import Predictor

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    description="Production ML platform for healthcare and finance predictions",
    version=config.APP_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictors
health_predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global health_predictor

    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")

    try:
        # Load sample data and train basic model
        loader = DataLoader(config.SAMPLE_DATA_PATH)
        df = loader.load_health_data()

        if len(df) > 0:
            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in ["patient_id", "clinical_deterioration"]]
            X = df[feature_cols]
            y = df["clinical_deterioration"]

            # Initialize preprocessor
            preprocessor = Preprocessor()
            X_processed = preprocessor.preprocess_health_data(X, fit=True)

            # Train model
            model = HealthPredictor(model_type="xgboost")
            model.fit(X_processed, y)

            # Initialize predictor service
            health_predictor = Predictor(model, preprocessor=preprocessor)

            logger.info("Health model loaded successfully")
        else:
            logger.warning("No training data found, models will be unavailable")

        # Set global predictors in routes
        set_predictors(health_predictor, None)

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")


# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "health_model_ready": health_predictor is not None,
    }


@app.get("/config")
async def get_config_info():
    """Get configuration info."""
    return {
        "app_env": config.APP_ENV,
        "debug": config.DEBUG,
        "log_level": config.LOG_LEVEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.DEBUG,
    )
