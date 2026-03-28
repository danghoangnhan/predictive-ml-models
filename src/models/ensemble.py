import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleStackingModel(BaseModel):
    """Stacking ensemble combining multiple base learners."""
    
    def __init__(self, use_xgboost: bool = True, use_lightgbm: bool = True):
        super().__init__(name="EnsembleStacking", model_type="stacking")
        
        # Define base learners
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        if use_xgboost:
            base_learners.append(('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=8, random_state=42)))
        
        if use_lightgbm:
            base_learners.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=8, random_state=42)))
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train stacking ensemble."""
        logger.info("Training EnsembleStacking model")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        logger.info("Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> dict:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, probabilities),
            'f1': f1_score(y_test, predictions),
            'accuracy': (predictions == y_test).mean()
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> dict:
        """Get average feature importance from base learners."""
        importances_list = []
        
        for name, estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances_list.append(estimator.feature_importances_)
        
        if importances_list and self.feature_names:
            avg_importances = np.mean(importances_list, axis=0)
            importance_dict = {name: imp for name, imp in zip(self.feature_names, avg_importances)}
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        
        return None


class XGBoostLGBStackingModel(BaseModel):
    """Stacking with XGBoost and LightGBM as base learners."""
    
    def __init__(self):
        super().__init__(name="XGBoostLGBStacking", model_type="xgb_lgb_stacking")
        
        base_learners = [
            ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42))
        ]
        
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train stacking ensemble."""
        logger.info("Training XGBoost + LightGBM Stacking model")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        logger.info("Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> dict:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, probabilities),
            'f1': f1_score(y_test, predictions),
            'accuracy': (predictions == y_test).mean()
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
