"""Data loading utilities."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate data from various sources."""

    def __init__(self, data_path: Union[str, Path] = "data/sample"):
        """Initialize data loader."""
        self.data_path = Path(data_path)

    def load_health_data(self) -> pd.DataFrame:
        """Load health/GAD-7 data."""
        csv_path = self.data_path / "health_scores.csv"

        if not csv_path.exists():
            logger.warning(f"Health data file not found at {csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded health data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_stock_data(self) -> pd.DataFrame:
        """Load stock pattern data."""
        csv_path = self.data_path / "stock_patterns.csv"

        if not csv_path.exists():
            logger.warning(f"Stock data file not found at {csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded stock data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load arbitrary CSV file."""
        csv_path = self.data_path / filename

        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def validate_health_data(self, df: pd.DataFrame) -> bool:
        """Validate health data structure."""
        required_cols = {"patient_id", "gad7_score", "age", "gender", "clinical_deterioration"}

        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            logger.error(f"Missing required columns: {missing}")
            return False

        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} null values")

        return True

    def validate_stock_data(self, df: pd.DataFrame) -> bool:
        """Validate stock data structure."""
        required_cols = {"date", "open", "high", "low", "close", "volume", "pattern"}

        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            logger.error(f"Missing required columns: {missing}")
            return False

        return True
