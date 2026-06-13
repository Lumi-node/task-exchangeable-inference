"""Runtime diagnostics for exchangeability assumptions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from exchangelib.repository import Task
from exchangelib.utils import compute_mmd_squared


@dataclass
class DiagnosticResult:
    """Container for diagnostic check results."""
    statistic: float
    threshold: float
    passed: bool
    message: str


class ExchangeabilityDiagnostic:
    """Runtime diagnostic checks for exchangeability assumptions.

    Monitors whether the exchangeability assumption holds during
    inference by computing the weighted MMD between the re-weighted
    historic distribution and the synthetic/target distribution.

    Parameters
    ----------
    bandwidth : float, optional
        RBF kernel bandwidth. Median heuristic if None.
    warning_threshold : float
        MMD threshold above which a warning is raised.
    error_threshold : float
        MMD threshold above which an error is raised.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        warning_threshold: float = 0.1,
        error_threshold: float = 0.5,
    ) -> None:
        self.bandwidth = bandwidth
        self.warning_threshold = warning_threshold
        self.error_threshold = error_threshold

    def compute_statistic(
        self,
        X_historic: np.ndarray,
        X_synthetic: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the (weighted) MMD between historic and synthetic data.

        Parameters
        ----------
        X_historic : (n, d) historic features
        X_synthetic : (m, d) synthetic features
        weights : (n,) importance weights for historic samples

        Returns
        -------
        mmd : float, the MMD statistic
        """
        X_historic = np.atleast_2d(X_historic)
        X_synthetic = np.atleast_2d(X_synthetic)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            w_norm = weights / weights.sum()
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(X_historic),
                size=len(X_historic),
                replace=True,
                p=w_norm,
            )
            X_reweighted = X_historic[indices]
        else:
            X_reweighted = X_historic

        return compute_mmd_squared(X_reweighted, X_synthetic, self.bandwidth)

    def check(
        self,
        X_historic: np.ndarray,
        X_synthetic: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> DiagnosticResult:
        """Run diagnostic check and return structured result."""
        stat = self.compute_statistic(X_historic, X_synthetic, weights)

        if stat > self.error_threshold:
            return DiagnosticResult(
                statistic=stat,
                threshold=self.error_threshold,
                passed=False,
                message=(
                    f"MMD={stat:.4f} exceeds error threshold "
                    f"{self.error_threshold:.4f}. Exchangeability assumption "
                    f"is likely violated — inference may be invalid."
                ),
            )

        if stat > self.warning_threshold:
            return DiagnosticResult(
                statistic=stat,
                threshold=self.warning_threshold,
                passed=True,
                message=(
                    f"MMD={stat:.4f} exceeds warning threshold "
                    f"{self.warning_threshold:.4f}. Exchangeability assumption "
                    f"may be weakly violated — treat results with caution."
                ),
            )

        return DiagnosticResult(
            statistic=stat,
            threshold=self.warning_threshold,
            passed=True,
            message=f"MMD={stat:.4f} is within acceptable range. Exchangeability holds.",
        )

    def raise_if_invalid(
        self,
        X_historic: np.ndarray,
        X_synthetic: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> DiagnosticResult:
        """Check exchangeability and raise/warn as appropriate."""
        result = self.check(X_historic, X_synthetic, weights)

        if not result.passed:
            raise ExchangeabilityViolation(result.message)

        if result.statistic > self.warning_threshold:
            warnings.warn(result.message, ExchangeabilityWarning, stacklevel=2)

        return result


class ExchangeabilityWarning(UserWarning):
    """Warning raised when exchangeability may be weakly violated."""


class ExchangeabilityViolation(Exception):
    """Raised when exchangeability assumption is strongly violated."""
