import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMTimeSeriesPredictor(BaseModel):
    """LSTM-based time series forecasting model."""
    
    def __init__(self, lookback_window: int = 7):
        super().__init__(name="LSTMTimeSeriesPredictor", model_type="lstm")
        self.lookback_window = lookback_window
        self.model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Train LSTM model."""
        logger.info(f"Training {self.name} with lookback window {self.lookback_window}")
        
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=(self.lookback_window, X_train.shape[2])),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
            
            self.is_trained = True
            logger.info("LSTM training complete")
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> dict:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics


class ProphetTimeSeriesPredictor(BaseModel):
    """Facebook Prophet-based time series forecasting."""
    
    def __init__(self):
        super().__init__(name="ProphetTimeSeriesPredictor", model_type="prophet")
        self.model = None
    
    def train(self, df: pd.DataFrame, **kwargs) -> None:
        """Train Prophet model."""
        logger.info(f"Training {self.name}")
        
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            if 'ds' not in df.columns or 'y' not in df.columns:
                logger.error("DataFrame must contain 'ds' (timestamp) and 'y' (value) columns")
                raise ValueError("Invalid DataFrame format for Prophet")
            
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            self.model.fit(df)
            
            self.is_trained = True
            logger.info("Prophet training complete")
        except ImportError:
            logger.error("Prophet not installed. Install with: pip install prophet")
            raise
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Make future predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def evaluate(self, df_test: pd.DataFrame, **kwargs) -> dict:
        """Evaluate model performance on test data."""
        forecast = self.predict(periods=len(df_test))
        
        # Compare with actual values
        actual_values = df_test['y'].values
        predicted_values = forecast['yhat'].values[:len(df_test)]
        
        metrics = {
            'mape': np.mean(np.abs((actual_values - predicted_values) / actual_values)),
            'rmse': np.sqrt(mean_squared_error(actual_values, predicted_values)),
            'mae': mean_absolute_error(actual_values, predicted_values)
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
