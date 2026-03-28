import logging

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


class HealthcareFeatureEngineering:
    """Feature engineering for healthcare data."""

    @staticmethod
    def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["week_of_year"] = df["timestamp"].dt.isocalendar().week
        df["day_of_month"] = df["timestamp"].dt.day
        return df

    @staticmethod
    def extract_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract trend features from GAD-7 scores."""
        df = df.copy()

        # Group by patient and calculate moving averages
        df["gad7_7day_ma"] = df.groupby("patient_id")["gad7_score"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df["gad7_14day_ma"] = df.groupby("patient_id")["gad7_score"].transform(
            lambda x: x.rolling(window=14, min_periods=1).mean()
        )

        # Trend: current vs 7-day MA
        df["gad7_trend"] = df["gad7_score"] - df["gad7_7day_ma"]

        # Rate of change
        df["gad7_rate_of_change"] = df.groupby("patient_id")["gad7_score"].diff()

        # Volatility
        df["gad7_volatility"] = df.groupby("patient_id")["gad7_score"].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )

        # Consecutive high scores (>15)
        df["consecutive_high_scores"] = (
            df.groupby("patient_id")
            .apply(lambda x: (x["gad7_score"] > 15).astype(int).cumsum())
            .reset_index(level=0, drop=True)
        )

        # Days since last assessment
        df["days_since_last_assessment"] = df.groupby("patient_id")["timestamp"].diff().dt.days

        return df

    @staticmethod
    def extract_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract NLP features from journal text."""
        df = df.copy()

        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Sentiment analysis
        df["journal_sentiment"] = df["journal_text"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"] if pd.notna(x) else 0
        )

        # Anxiety-related keywords
        anxiety_keywords = ["anxious", "worried", "nervous", "stressed", "panic", "fear"]
        df["journal_anxiety_keywords_count"] = df["journal_text"].apply(
            lambda x: sum(x.lower().count(kw) for kw in anxiety_keywords) if pd.notna(x) else 0
        )

        # Positive keywords
        positive_keywords = ["good", "happy", "better", "great", "wonderful", "calm"]
        df["journal_positive_keywords_count"] = df["journal_text"].apply(
            lambda x: sum(x.lower().count(kw) for kw in positive_keywords) if pd.notna(x) else 0
        )

        # Journal length
        df["journal_length"] = df["journal_text"].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )

        return df

    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = HealthcareFeatureEngineering.extract_temporal_features(df)
        df = HealthcareFeatureEngineering.extract_trend_features(df)
        df = HealthcareFeatureEngineering.extract_nlp_features(df)

        # Fill NaN values
        df.fillna(0, inplace=True)

        logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df


class FinanceFeatureEngineering:
    """Feature engineering for finance data."""

    @staticmethod
    def extract_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract candlestick pattern features."""
        df = df.copy()

        df["body_ratio"] = np.where(
            df["close"] > df["open"],
            (df["close"] - df["open"]) / (df["high"] - df["low"]),
            (df["open"] - df["close"]) / (df["high"] - df["low"]),
        )

        df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
            df["high"] - df["low"]
        )
        df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
            df["high"] - df["low"]
        )

        return df

    @staticmethod
    def extract_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical analysis features."""
        df = df.copy()

        # Average True Range (ATR)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["close"].shift()), np.abs(df["low"] - df["close"].shift())
            ),
        )
        df["atr"] = df["tr"].rolling(window=14).mean()

        # Bollinger Bands
        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bollinger_upper"] = sma + (std * 2)
        df["bollinger_lower"] = sma - (std * 2)
        df["bollinger_width"] = df["bollinger_upper"] - df["bollinger_lower"]

        # Volume SMA
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_surge"] = df["volume"] / df["volume_sma_20"]

        return df

    @staticmethod
    def extract_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract pattern-specific features."""
        df = df.copy()

        # Support and Resistance
        df["support_level"] = df["low"].rolling(window=20).min()
        df["resistance_level"] = df["high"].rolling(window=20).max()

        # Pattern geometry
        df["pattern_slope"] = (df["close"] - df["close"].shift(5)) / 5
        df["pattern_width"] = df["high"] - df["low"]
        df["pattern_height"] = df["resistance_level"] - df["support_level"]

        # Breakout probability (simplified)
        df["break_probability"] = (df["close"] - df["support_level"]) / (
            df["resistance_level"] - df["support_level"]
        )

        return df

    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = FinanceFeatureEngineering.extract_candlestick_features(df)
        df = FinanceFeatureEngineering.extract_technical_features(df)
        df = FinanceFeatureEngineering.extract_pattern_features(df)

        # Fill NaN values
        df.fillna(0, inplace=True)

        logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df


class DataPreprocessor:
    """Main data preprocessing pipeline."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit scaler to data."""
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data."""
        return self.scaler.fit_transform(X)
