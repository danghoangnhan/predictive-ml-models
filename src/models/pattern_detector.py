import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PatternDetector(BaseModel):
    """Finance pattern detector for candlestick chart patterns."""
    
    PATTERN_CLASSES = {
        0: "triangle",
        1: "wedge",
        2: "flag",
        3: "other"
    }
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(name="PatternDetector", model_type=model_type)
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=4,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train pattern detector model."""
        logger.info(f"Training {self.name} with {self.model_type}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        logger.info("Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict chart patterns."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates for each pattern class."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> tuple:
        """Predict patterns with confidence scores."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> dict:
        """Evaluate model performance on all classes."""
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        # Per-class metrics
        for class_idx, class_name in self.PATTERN_CLASSES.items():
            class_mask = y_test == class_idx
            if class_mask.sum() > 0:
                metrics[f'{class_name}_precision'] = precision_score(
                    y_test, predictions, labels=[class_idx], average='weighted', zero_division=0
                )
                metrics[f'{class_name}_recall'] = recall_score(
                    y_test, predictions, labels=[class_idx], average='weighted', zero_division=0
                )
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> dict:
        """Get feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Model {self.model_type} does not support feature importance")
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            importance_dict = {name: imp for name, imp in zip(self.feature_names, importances)}
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        
        return importances
    
    def get_pattern_name(self, class_idx: int) -> str:
        """Get pattern name from class index."""
        return self.PATTERN_CLASSES.get(class_idx, "unknown")
