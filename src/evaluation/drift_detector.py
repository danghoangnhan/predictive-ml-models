"""Data and concept drift detection."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and concept drift."""

    def __init__(self, threshold: float = 0.15):
        """Initialize drift detector."""
        self.threshold = threshold
        self.baseline_stats = None
        self.drift_history = []

    def fit_baseline(self, X: pd.DataFrame) -> None:
        """Fit baseline statistics from training data."""
        self.baseline_stats = {
            "mean": X.mean(),
            "std": X.std(),
            "min": X.min(),
            "max": X.max(),
            "q25": X.quantile(0.25),
            "q75": X.quantile(0.75),
        }
        logger.info("Baseline statistics fitted")

    def kolmogorov_smirnov_test(
        self, X_new: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        """Perform Kolmogorov-Smirnov test for each feature."""
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        results = {}

        for col in X_new.select_dtypes(include=[np.number]).columns:
            if col not in self.baseline_stats["mean"].index:
                continue

            # Generate baseline distribution (approximate)
            baseline_dist = np.random.normal(
                self.baseline_stats["mean"][col],
                self.baseline_stats["std"][col],
                100,
            )

            # KS test
            statistic, p_value = stats.ks_2samp(baseline_dist, X_new[col].dropna())
            results[col] = (statistic, p_value)

        return results

    def population_stability_index(self, X_new: pd.DataFrame) -> Dict[str, float]:
        """Calculate Population Stability Index for each feature."""
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        psi_scores = {}

        for col in X_new.select_dtypes(include=[np.number]).columns:
            if col not in self.baseline_stats["mean"].index:
                continue

            # Bin both distributions
            baseline_mean = self.baseline_stats["mean"][col]
            baseline_std = self.baseline_stats["std"][col]

            n_bins = 10
            bins = np.linspace(
                min(baseline_mean - 3 * baseline_std, X_new[col].min()),
                max(baseline_mean + 3 * baseline_std, X_new[col].max()),
                n_bins,
            )

            baseline_counts = np.histogram(
                np.random.normal(baseline_mean, baseline_std, 1000), bins=bins
            )[0]
            new_counts = np.histogram(X_new[col].dropna(), bins=bins)[0]

            # PSI calculation
            baseline_pct = baseline_counts / baseline_counts.sum()
            new_pct = new_counts / (new_counts.sum() + 1e-8)

            psi = np.sum((new_pct - baseline_pct) * np.log((new_pct + 1e-8) / (baseline_pct + 1e-8)))
            psi_scores[col] = psi

        return psi_scores

    def detect_drift(self, X_new: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive drift detection."""
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        # KS test
        ks_results = self.kolmogorov_smirnov_test(X_new)

        # PSI
        psi_results = self.population_stability_index(X_new)

        # Detect drift
        drift_detected = False
        drifted_features = []

        for feature, psi in psi_results.items():
            if psi > self.threshold:
                drift_detected = True
                drifted_features.append((feature, psi))

        drift_report = {
            "drift_detected": drift_detected,
            "threshold": self.threshold,
            "ks_test": ks_results,
            "psi_scores": psi_results,
            "drifted_features": drifted_features,
            "num_samples": len(X_new),
        }

        self.drift_history.append(drift_report)
        logger.info(f"Drift detection complete. Drift detected: {drift_detected}")

        return drift_report

    def get_drift_history(self) -> List[Dict]:
        """Get drift detection history."""
        return self.drift_history

    def statistical_summary_comparison(
        self, X_new: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compare baseline and new data statistics."""
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        comparison = {}

        for col in X_new.select_dtypes(include=[np.number]).columns:
            if col not in self.baseline_stats["mean"].index:
                continue

            baseline_mean = self.baseline_stats["mean"][col]
            new_mean = X_new[col].mean()
            mean_change = (new_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)

            comparison[col] = {
                "baseline_mean": baseline_mean,
                "new_mean": new_mean,
                "mean_change_pct": mean_change * 100,
                "baseline_std": self.baseline_stats["std"][col],
                "new_std": X_new[col].std(),
            }

        return comparison
