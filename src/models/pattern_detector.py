"""Stock chart pattern detection model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Optional, List
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detect technical chart patterns in stock data."""

    PATTERNS = ["head_shoulders", "double_bottom", "triangle", "breakout", "consolidation"]

    def __init__(self):
        """Initialize pattern detector."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names = None
        self.label_to_pattern = {i: p for i, p in enumerate(self.PATTERNS)}

    def extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical features for pattern detection."""
        features = pd.DataFrame(index=df.index)

        # Volatility
        features["volatility"] = df["close"].pct_change().rolling(20).std()

        # Support and resistance (simple: min/max of lookback)
        features["support"] = df["low"].rolling(20).min()
        features["resistance"] = df["high"].rolling(20).max()

        # Range
        features["range"] = (df["high"] - df["low"]) / df["close"]

        # Volume
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Moving averages
        features["sma_20"] = df["close"].rolling(20).mean()
        features["sma_50"] = df["close"].rolling(50).mean()
        features["ma_ratio"] = features["sma_20"] / (features["sma_50"] + 1e-8)

        # Momentum
        features["momentum"] = df["close"].diff(5)
        features["roc"] = df["close"].pct_change(10)

        # Price position relative to bands
        features["bb_position"] = (
            (df["close"] - features["sma_20"]) /
            (2 * df["close"].rolling(20).std() + 1e-8)
        )

        features.dropna(inplace=True)
        return features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PatternDetector":
        """Train pattern detector."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        logger.info(f"Trained pattern detector on {len(X)} samples")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict chart pattern."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        predictions = self.model.predict(X)
        return np.array([self.label_to_pattern.get(p, "unknown") for p in predictions])

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
        """Predict pattern with confidence scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        proba = self.model.predict_proba(X)
        probabilities = {}

        for i, pattern in enumerate(self.PATTERNS):
            probabilities[pattern] = float(proba[0, i]) if len(proba) > 0 else 0.0

        top_pattern = self.PATTERNS[np.argmax(proba[0])] if len(proba) > 0 else "unknown"
        return top_pattern, probabilities

    def detect_patterns_in_window(self, df: pd.DataFrame) -> List[Dict]:
        """Detect multiple patterns in sliding window."""
        patterns_detected = []

        if len(df) < 50:
            logger.warning("Insufficient data for pattern detection")
            return patterns_detected

        features = self.extract_pattern_features(df)

        if len(features) == 0:
            return patterns_detected

        X = features.iloc[[-1]]  # Last row
        pattern, confidence = self.predict_proba(X)

        patterns_detected.append({
            "pattern": pattern,
            "confidence": confidence,
            "date": df.index[-1] if hasattr(df.index, '__getitem__') else None,
        })

        return patterns_detected

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved pattern detector to {path}")

    @staticmethod
    def load(path: Path) -> "PatternDetector":
        """Load model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded pattern detector from {path}")
        return model
