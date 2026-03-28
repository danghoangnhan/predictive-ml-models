"""Time series forecasting models (LSTM and Prophet)."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Time series forecasting with LSTM and statistical methods."""

    def __init__(self, model_type: str = "lstm", lookback: int = 60, forecast_horizon: int = 30):
        """Initialize time series forecaster."""
        self.model_type = model_type
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = []

    def create_sequences(
        self, data: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        X, y = [], []

        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])

        return np.array(X), np.array(y)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TimeSeriesForecaster":
        """Fit time series model."""

        if self.model_type == "lstm":
            self._fit_lstm(X)
        elif self.model_type == "arima":
            self._fit_statistical(X)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using statistical model")
            self._fit_statistical(X)

        return self

    def _fit_lstm(self, X: pd.DataFrame) -> None:
        """Fit LSTM model (simplified without TensorFlow)."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
        except ImportError:
            logger.warning("TensorFlow not available, using simple statistical model")
            self._fit_statistical(X)
            return

        # Normalize data
        data = X.values.flatten()
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        # Create sequences
        X_seq, y_seq = self.create_sequences(scaled_data, self.lookback)

        if len(X_seq) == 0:
            logger.warning("Insufficient data for sequence creation")
            return

        # Build LSTM
        self.model = Sequential([
            LSTM(128, activation="relu", input_shape=(self.lookback, 1), return_sequences=True),
            LSTM(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ])

        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train (simplified, no callbacks)
        self.model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
        logger.info(f"Trained LSTM model on {len(X_seq)} sequences")

    def _fit_statistical(self, X: pd.DataFrame) -> None:
        """Fit statistical time series model."""
        # Store recent values for simple AR-like prediction
        data = X.values.flatten()
        self.history = list(data[-self.lookback :])
        logger.info(f"Initialized statistical model with {len(self.history)} historical values")

    def forecast(self, X: pd.DataFrame, horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time series forecast."""
        if horizon is None:
            horizon = self.forecast_horizon

        if self.model_type == "lstm" and self.model is not None:
            return self._forecast_lstm(X, horizon)
        else:
            return self._forecast_statistical(X, horizon)

    def _forecast_lstm(self, X: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using LSTM."""
        data = X.values.flatten()
        scaled_data = self.scaler.transform(data.reshape(-1, 1))

        predictions = []
        current_seq = scaled_data[-self.lookback :]

        for _ in range(horizon):
            try:
                next_pred = self.model.predict(current_seq.reshape(1, -1, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                current_seq = np.append(current_seq[1:], next_pred)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                break

        # Inverse scale
        if predictions:
            forecasts = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        else:
            forecasts = np.array([]).reshape(-1, 1)

        # Simple confidence intervals
        confidence_upper = forecasts.flatten() * 1.1
        confidence_lower = forecasts.flatten() * 0.9

        return forecasts.flatten(), np.array([confidence_lower, confidence_upper])

    def _forecast_statistical(self, X: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast using simple statistical method."""
        data = X.values.flatten()

        # Simple exponential smoothing
        alpha = 0.3
        last_value = data[-1]
        predictions = []

        for _ in range(horizon):
            pred = alpha * last_value + (1 - alpha) * np.mean(data)
            predictions.append(pred)
            last_value = pred

        forecasts = np.array(predictions)
        forecast_std = np.std(data) * 0.5

        confidence_lower = forecasts - 1.96 * forecast_std
        confidence_upper = forecasts + 1.96 * forecast_std

        return forecasts, np.array([confidence_lower, confidence_upper])

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "type": self.model_type,
            "lookback": self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "scaler": self.scaler,
            "history": self.history,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved time series model to {path}")

    @staticmethod
    def load(path: Path) -> "TimeSeriesForecaster":
        """Load model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        forecaster = TimeSeriesForecaster(
            model_type=model_data["type"],
            lookback=model_data["lookback"],
            forecast_horizon=model_data["forecast_horizon"],
        )
        forecaster.scaler = model_data["scaler"]
        forecaster.history = model_data["history"]

        logger.info(f"Loaded time series model from {path}")
        return forecaster
