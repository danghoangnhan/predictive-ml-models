import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from CSV files."""

    @staticmethod
    def load_health_data(filepath: str) -> pd.DataFrame:
        """Load healthcare data (GAD-7 scores and journal entries)."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded health data from {filepath}: shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading health data: {e}")
            raise

    @staticmethod
    def load_finance_data(filepath: str) -> pd.DataFrame:
        """Load finance data (OHLCV and patterns)."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded finance data from {filepath}: shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading finance data: {e}")
            raise

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Generic CSV loader."""
        return pd.read_csv(filepath)


class DataValidator:
    """Validate data quality and format."""

    @staticmethod
    def validate_health_columns(df: pd.DataFrame) -> bool:
        """Validate required columns for health data."""
        required_cols = ["patient_id", "gad7_score", "journal_text", "timestamp"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True

    @staticmethod
    def validate_finance_columns(df: pd.DataFrame) -> bool:
        """Validate required columns for finance data."""
        required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True

    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """Check if missing values exceed threshold."""
        missing_ratio = df.isnull().sum() / len(df)
        if (missing_ratio > threshold).any():
            logger.warning(
                f"High missing values detected:\n{missing_ratio[missing_ratio > threshold]}"
            )
            return False
        return True


def load_and_validate_health_data(filepath: str) -> pd.DataFrame:
    """Load and validate health data."""
    df = DataLoader.load_health_data(filepath)
    validator = DataValidator()

    if not validator.validate_health_columns(df):
        logger.error("Health data validation failed")
        return None

    if not validator.check_missing_values(df):
        logger.warning("Health data has missing values")

    return df


def load_and_validate_finance_data(filepath: str) -> pd.DataFrame:
    """Load and validate finance data."""
    df = DataLoader.load_finance_data(filepath)
    validator = DataValidator()

    if not validator.validate_finance_columns(df):
        logger.error("Finance data validation failed")
        return None

    if not validator.check_missing_values(df):
        logger.warning("Finance data has missing values")

    return df
