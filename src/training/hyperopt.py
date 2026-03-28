"""Hyperparameter optimization using Optuna."""

import logging
from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optimize hyperparameters using Optuna."""

    def __init__(self, n_trials: int = 50, timeout: int | None = None, cv_folds: int = 5):
        """Initialize optimizer."""
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.study = None
        self.best_params = {}

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable,
        param_space: dict[str, Any],
        direction: str = "maximize",
    ) -> dict[str, Any]:
        """Run hyperparameter optimization."""

        def objective(trial):
            """Objective function for Optuna."""
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config.get("type", "int")
                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config.get("low", 1),
                        param_config.get("high", 100),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config.get("low", 0.0),
                        param_config.get("high", 1.0),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config.get("choices", [])
                    )

            # Create and evaluate model
            try:
                model = model_factory(**params)
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring="roc_auc")
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")

        # Create study and optimize
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            sampler=sampler,
            direction=direction,
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        logger.info(f"Optimization completed. Best score: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def get_best_params(self) -> dict[str, Any]:
        """Get best hyperparameters."""
        return self.best_params

    def get_study_trials(self) -> list:
        """Get all trials."""
        if self.study is None:
            return []
        return self.study.trials

    def get_optimization_history(self) -> tuple:
        """Get optimization history."""
        if self.study is None:
            return [], []

        trials = self.study.trials
        scores = [t.value if t.value is not None else float("-inf") for t in trials]
        return list(range(len(trials))), scores
