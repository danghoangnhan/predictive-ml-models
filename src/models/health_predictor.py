import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class HealthcarePredictor(BaseModel):
    """Healthcare deterioration risk predictor based on GAD-7 trends."""
    
    def __init__(self, ensemble_models: list = None):
        super().__init__(name="HealthcarePredictor", model_type="ensemble")
        
        self.ensemble_models = {}
        if ensemble_models is None:
            ensemble_models = ['logistic_regression', 'random_forest']
        
        for model_name in ensemble_models:
            if model_name == 'logistic_regression':
                self.ensemble_models['logistic_regression'] = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42
                )
            elif model_name == 'random_forest':
                self.ensemble_models['random_forest'] = RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
                )
            elif model_name == 'neural_network':
                self.ensemble_models['neural_network'] = MLPClassifier(
                    hidden_layer_sizes=(64, 32, 16),
                    learning_rate_init=0.001,
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train all ensemble models."""
        logger.info(f"Training {self.name} with {len(self.ensemble_models)} models")
        
        for name, model in self.ensemble_models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
        
        self.is_trained = True
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        logger.info("Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions (majority vote)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        probabilities = []
        
        for name, model in self.ensemble_models.items():
            pred = model.predict(X)
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
                probabilities.append(proba)
        
        # Majority vote
        ensemble_pred = np.array(predictions).T
        final_predictions = (ensemble_pred.sum(axis=1) > len(self.ensemble_models) / 2).astype(int)
        
        # Average probability
        final_proba = np.mean(probabilities, axis=0) if probabilities else None
        
        return final_predictions, final_proba
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        probabilities = []
        
        for name, model in self.ensemble_models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
                probabilities.append(proba)
        
        return np.mean(probabilities, axis=0) if probabilities else None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> dict:
        """Evaluate model performance."""
        predictions, probabilities = self.predict(X_test)
        
        metrics = {
            'accuracy': (predictions == y_test).mean(),
            'auc_roc': roc_auc_score(y_test, probabilities),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from Random Forest."""
        if 'random_forest' not in self.ensemble_models:
            logger.warning("Random Forest not in ensemble")
            return None
        
        rf_model = self.ensemble_models['random_forest']
        importances = rf_model.feature_importances_
        
        if self.feature_names:
            importance_dict = {name: imp for name, imp in zip(self.feature_names, importances)}
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        
        return importances
