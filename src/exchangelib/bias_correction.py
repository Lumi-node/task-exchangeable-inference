"""Martingale-based bias correction for synthetic estimators."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from exchangelib.utils import martingale_residuals


class BiasCorrector:
    """Correct estimation bias when using synthetic or re-weighted data.

    Implements a martingale-based correction that adjusts a base estimator's
    predictions using importance-weighted residuals from historic data. The
    correction accounts for distribution shift between the synthetic
    generation process and the true data-generating mechanism.

    The corrected estimate is:
        theta_corrected = theta_raw - (1/n) sum_i w_i * (y_i - f(x_i))

    where w_i are kernel importance weights and f is the base estimator.
    """

    def __init__(self) -> None:
        self._correction_term: Optional[float] = None
        self._residuals: Optional[np.ndarray] = None

    def correct(
        self,
        estimator: Callable[[np.ndarray], np.ndarray],
        X_historic: np.ndarray,
        y_historic: np.ndarray,
        kernel_weights: np.ndarray,
        raw_estimate: float,
    ) -> float:
        """Apply martingale bias correction to a raw estimate.

        Parameters
        ----------
        estimator : callable
            A fitted estimator: X -> predicted y.
        X_historic : (n, d) historic features
        y_historic : (n,) historic targets
        kernel_weights : (n,) importance weights from KMM
        raw_estimate : float
            The uncorrected estimate from synthetic data.

        Returns
        -------
        corrected : float
            The bias-corrected estimate.
        """
        X_historic = np.atleast_2d(X_historic)
        y_historic = np.asarray(y_historic, dtype=float).ravel()
        kernel_weights = np.asarray(kernel_weights, dtype=float)

        predicted = estimator(X_historic).ravel()
        self._residuals = martingale_residuals(y_historic, predicted, kernel_weights)
        self._correction_term = float(self._residuals.mean())

        return raw_estimate - self._correction_term

    def correct_predictions(
        self,
        estimator: Callable[[np.ndarray], np.ndarray],
        X_historic: np.ndarray,
        y_historic: np.ndarray,
        kernel_weights: np.ndarray,
        raw_predictions: np.ndarray,
    ) -> np.ndarray:
        """Apply pointwise bias correction to a vector of predictions.

        Parameters
        ----------
        estimator : callable
            A fitted estimator: X -> predicted y.
        X_historic, y_historic, kernel_weights : historic data + weights
        raw_predictions : (m,) uncorrected predictions

        Returns
        -------
        corrected : (m,) bias-corrected predictions
        """
        X_historic = np.atleast_2d(X_historic)
        y_historic = np.asarray(y_historic, dtype=float).ravel()
        kernel_weights = np.asarray(kernel_weights, dtype=float)

        predicted = estimator(X_historic).ravel()
        self._residuals = martingale_residuals(y_historic, predicted, kernel_weights)
        self._correction_term = float(self._residuals.mean())

        return np.asarray(raw_predictions, dtype=float) - self._correction_term

    @property
    def correction_term(self) -> Optional[float]:
        """The last computed correction term (mean weighted residual)."""
        return self._correction_term

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """The last computed martingale residuals."""
        return self._residuals
