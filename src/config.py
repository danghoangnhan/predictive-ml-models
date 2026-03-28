import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIGS_DIR = BASE_DIR / "configs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Settings:
    """Application settings from environment variables."""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Model Configuration
    HEALTHCARE_MODEL_PATH: str = os.getenv("HEALTHCARE_MODEL_PATH", str(MODELS_DIR / "healthcare_ensemble.pkl"))
    FINANCE_MODEL_PATH: str = os.getenv("FINANCE_MODEL_PATH", str(MODELS_DIR / "finance_xgboost.pkl"))
    
    # Data Configuration
    DATA_PATH: str = os.getenv("DATA_PATH", str(DATA_DIR))
    SAMPLE_DATA_PATH: str = os.getenv("SAMPLE_DATA_PATH", str(SAMPLE_DATA_DIR))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))
    
    # Monitoring
    DRIFT_DETECTION_ENABLED: bool = os.getenv("DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    RETRAINING_TRIGGER_ENABLED: bool = os.getenv("RETRAINING_TRIGGER_ENABLED", "true").lower() == "true"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
