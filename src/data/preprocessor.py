"""Feature engineering and preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Data preprocessing and feature engineering."""

    def __init__(self, scaler_type: str = "standard"):
        """Initialize preprocessor."""
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[list] = None

    def preprocess_health_data(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Preprocess health/GAD-7 data."""
        df_processed = df.copy()

        # Handle missing values
        df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)

        # Remove duplicates
        df_processed.drop_duplicates(subset=["patient_id"], keep="first", inplace=True)

        # Feature engineering
        df_processed["gad7_severity"] = pd.cut(
            df_processed["gad7_score"],
            bins=[0, 5, 10, 15, 21],
            labels=["minimal", "mild", "moderate", "severe"],
        )

        df_processed["age_group"] = pd.cut(
            df_processed["age"],
            bins=[0, 18, 35, 50, 65, 100],
            labels=["child", "young_adult", "adult", "senior", "elderly"],
        )

        # Encode categorical variables
        categorical_cols = ["gender", "gad7_severity", "age_group"]
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].dtype == "object":
                if fit:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df_processed[col] = self.label_encoders[col].transform(
                            df_processed[col].astype(str)
                        )

        # Outlier detection
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        Q1 = df_processed[numeric_cols].quantile(0.25)
        Q3 = df_processed[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ((df_processed[numeric_cols] < (Q1 - 1.5 * IQR)) |
                (df_processed[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        n_outliers = mask.sum()
        if n_outliers > 0:
            logger.info(f"Found {n_outliers} outliers in health data")

        logger.info(f"Preprocessed health data: {df_processed.shape}")
        return df_processed

    def preprocess_stock_data(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Preprocess stock/OHLCV data."""
        df_processed = df.copy()

        # Convert date to datetime
        if "date" in df_processed.columns:
            df_processed["date"] = pd.to_datetime(df_processed["date"])
            df_processed = df_processed.sort_values("date").reset_index(drop=True)

        # Handle missing values
        df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)

        # Feature engineering
        if "close" in df_processed.columns and "open" in df_processed.columns:
            df_processed["returns"] = df_processed["close"].pct_change()
            df_processed["volatility"] = df_processed["returns"].rolling(5).std()
            df_processed["momentum"] = df_processed["close"].diff(5)

        if "high" in df_processed.columns and "low" in df_processed.columns:
            df_processed["range"] = df_processed["high"] - df_processed["low"]
            df_processed["high_low_ratio"] = df_processed["high"] / (df_processed["low"] + 1e-8)

        if "volume" in df_processed.columns:
            df_processed["volume_sma"] = df_processed["volume"].rolling(5).mean()

        # Encode pattern labels
        if "pattern" in df_processed.columns and df_processed["pattern"].dtype == "object":
            if fit:
                le = LabelEncoder()
                df_processed["pattern_encoded"] = le.fit_transform(df_processed["pattern"])
                self.label_encoders["pattern"] = le
            else:
                if "pattern" in self.label_encoders:
                    df_processed["pattern_encoded"] = self.label_encoders["pattern"].transform(
                        df_processed["pattern"]
                    )

        # Drop NaN rows created by rolling calculations
        df_processed.dropna(inplace=True)

        logger.info(f"Preprocessed stock data: {df_processed.shape}")
        return df_processed

    def scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """Scale numeric features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if fit:
            X_scaled = self.scaler.fit_transform(X[numeric_cols])
        else:
            X_scaled = self.scaler.transform(X[numeric_cols])

        self.feature_names = list(numeric_cols)
        return X_scaled

    def get_feature_names(self) -> Optional[list]:
        """Get feature names."""
        return self.feature_names
