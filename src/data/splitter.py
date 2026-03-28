import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train/val/test sets."""

    @staticmethod
    def stratified_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data with stratification for classification tasks."""

        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

        # First split: train + temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=y
        )

        # Second split: val and test
        test_size_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size_ratio, random_state=random_state, stratify=y_temp
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def time_series_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data in chronological order for time series tasks."""

        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

        logger.info(
            f"Time series split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def kfold_split(
        X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
    ) -> list:
        """Create k-fold splits for cross-validation."""
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []

        for train_idx, test_idx in skf.split(X, y):
            folds.append((X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]))

        logger.info(f"Created {n_splits} k-fold splits")
        return folds
