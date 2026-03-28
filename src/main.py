import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from api import routes
from models.health_predictor import HealthcarePredictor
from models.pattern_detector import PatternDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Predictive ML Models API",
    description="Healthcare and Finance prediction models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting up Predictive ML Models API")
    
    try:
        # Load models
        health_model = HealthcarePredictor()
        pattern_model = PatternDetector()
        
        # Set models in routes
        routes.set_models(health_model, pattern_model)
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # Continue startup even if models fail to load
        pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Predictive ML Models API")


def main():
    """Run the application."""
    logger.info(f"Starting API on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
        log_level="info"
    )


if __name__ == "__main__":
    main()
