"""Configuration management for predictive ML models."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration."""

    # Application
    APP_NAME = os.getenv("APP_NAME", "predictive-ml-models")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data"))
    SAMPLE_DATA_PATH = Path(os.getenv("SAMPLE_DATA_PATH", DATA_PATH / "sample"))
    LOGS_PATH = Path(os.getenv("LOGS_PATH", BASE_DIR / "logs"))
    MODELS_PATH = Path(os.getenv("MODELS_PATH", BASE_DIR / "models"))

    # Create directories if they don't exist
    for path in [DATA_PATH, SAMPLE_DATA_PATH, LOGS_PATH, MODELS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_WORKERS = int(os.getenv("API_WORKERS", 4))
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

    # Model Configuration
    HEALTH_MODEL_PATH = MODELS_PATH / os.getenv("HEALTH_MODEL_PATH", "health_model.pkl")
    STOCK_MODEL_PATH = MODELS_PATH / os.getenv("STOCK_MODEL_PATH", "stock_model.pkl")
    EXPLAINER_CACHE = os.getenv("EXPLAINER_CACHE", "true").lower() == "true"
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

    # Monitoring & Drift
    DRIFT_DETECTION_ENABLED = os.getenv("DRIFT_DETECTION_ENABLED", "true").lower() == "true"
    DRIFT_ALERT_THRESHOLD = float(os.getenv("DRIFT_ALERT_THRESHOLD", 0.15))
    DATA_DRIFT_CHECK_INTERVAL = int(os.getenv("DATA_DRIFT_CHECK_INTERVAL", 3600))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
    AUDIT_LOGGING = os.getenv("AUDIT_LOGGING", "true").lower() == "true"

    # Security & Privacy
    PII_MASKING = os.getenv("PII_MASKING", "false").lower() == "true"
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 3600))

    # Explainability
    SHAP_ENABLED = os.getenv("SHAP_ENABLED", "true").lower() == "true"
    SHAP_BACKGROUND_SAMPLES = int(os.getenv("SHAP_BACKGROUND_SAMPLES", 100))
    LIME_ENABLED = os.getenv("LIME_ENABLED", "true").lower() == "true"
    LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", 50))

    # Hyperparameter Optimization
    OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", 100))
    OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", 3600))
    CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", 5))

    # Stock Market Data
    STOCK_DATA_SOURCE = os.getenv("STOCK_DATA_SOURCE", "yfinance")
    STOCK_LOOKBACK_WINDOW = int(os.getenv("STOCK_LOOKBACK_WINDOW", 60))
    STOCK_FORECAST_HORIZON = int(os.getenv("STOCK_FORECAST_HORIZON", 30))

    # Healthcare
    HIPAA_COMPLIANCE = os.getenv("HIPAA_COMPLIANCE", "true").lower() == "true"
    PATIENT_DATA_RETENTION_DAYS = int(os.getenv("PATIENT_DATA_RETENTION_DAYS", 2555))
    AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", 2555))


class DevelopmentConfig(Config):
    """Development configuration."""
    APP_ENV = "development"
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    APP_ENV = "production"
    DEBUG = False
    LOG_LEVEL = "INFO"
    RATE_LIMIT_ENABLED = True


class TestingConfig(Config):
    """Testing configuration."""
    APP_ENV = "testing"
    DEBUG = True
    TESTING = True
    LOG_LEVEL = "DEBUG"


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("APP_ENV", "development")

    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Export current configuration
config = get_config()
