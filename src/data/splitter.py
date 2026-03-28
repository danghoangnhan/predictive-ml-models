"""Train/validation/test data splitting utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train, validation, and test sets."""

    @staticmethod
    def split_train_val_test(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets."""

        stratify_y = y if stratify else None

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_y[X_temp.index] if stratify else None,
        )

        logger.info(
            f"Split data: train={X_train.shape[0]}, "
            f"val={X_val.shape[0]}, test={X_test.shape[0]}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def kfold_split(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_splits: int = 5,
        stratified: bool = False,
        random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k-fold cross-validation splits."""

        if stratified and y is not None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))

        logger.info(f"Generated {len(splits)} k-fold splits (n_splits={n_splits})")
        return splits

    @staticmethod
    def time_series_split(
        X: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data (no shuffling, temporal order preserved)."""

        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        logger.info(
            f"Time series split: train={len(X_train)}, "
            f"val={len(X_val)}, test={len(X_test)}"
        )

        return X_train, X_val, X_test
