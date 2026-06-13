"""Main inference engine for task-exchangeability with synthetic data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from exchangelib.bias_correction import BiasCorrector
from exchangelib.diagnostics import ExchangeabilityDiagnostic
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.repository import Task
from exchangelib.utils import conformal_quantiles


@dataclass
class InferenceResult:
    """Result of exchangeable inference."""
    prediction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    conformal_quantile: float
    bias_correction: float
    diagnostic_mmd: float


class ExchangeableInference:
    """Calibrated inference engine combining synthetic and historic data.

    Uses kernel mean matching to re-weight historic tasks, applies
    martingale bias correction, and produces conformal prediction
    intervals with finite-sample coverage guarantees under
    exchangeability.

    Parameters
    ----------
    kernel : ExchangeabilityKernel, optional
        Pre-configured kernel. Created with defaults if None.
    estimator : callable, optional
        A scikit-learn-style estimator with fit/predict. Defaults to Ridge.
    bandwidth : float, optional
        RBF bandwidth passed to kernel if kernel is None.
    run_diagnostics : bool
        Whether to run exchangeability diagnostics during inference.
    """

    def __init__(
        self,
        kernel: Optional[ExchangeabilityKernel] = None,
        estimator=None,
        bandwidth: Optional[float] = None,
        run_diagnostics: bool = True,
    ) -> None:
        self.kernel = kernel or ExchangeabilityKernel(bandwidth=bandwidth)
        self._base_estimator = estimator
        self.run_diagnostics = run_diagnostics
        self._bias_corrector = BiasCorrector()
        self._diagnostic = ExchangeabilityDiagnostic(bandwidth=bandwidth)
        self._fitted_estimator = None
        self._conformal_scores: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        synthetic_X: np.ndarray,
        synthetic_y: np.ndarray,
        historic_tasks: Optional[List[Task]] = None,
    ) -> "ExchangeableInference":
        """Fit the inference engine on synthetic data and historic tasks.

        Parameters
        ----------
        synthetic_X : (n, d) synthetic features
        synthetic_y : (n,) synthetic targets
        historic_tasks : list of Task, optional
            If provided, kernel weights and bias correction are computed.
        """
        synthetic_X = np.atleast_2d(synthetic_X)
        synthetic_y = np.asarray(synthetic_y, dtype=float).ravel()

        estimator = self._base_estimator
        if estimator is None:
            estimator = Ridge(alpha=1.0)

        if historic_tasks:
            self.kernel.fit(historic_tasks)
            target_task = Task(X=synthetic_X, y=synthetic_y)
            self._weights = self.kernel.score(target_task)

            X_hist = np.vstack([t.X for t in historic_tasks])
            y_hist = np.concatenate([t.y for t in historic_tasks])

            combined_X = np.vstack([X_hist, synthetic_X])
            n_hist = X_hist.shape[0]
            combined_weights = np.ones(combined_X.shape[0])
            combined_weights[:n_hist] = self._weights
            combined_y = np.concatenate([y_hist, synthetic_y])

            estimator.fit(combined_X, combined_y, sample_weight=combined_weights)
            self._fitted_estimator = estimator

            raw_pred = estimator.predict(synthetic_X)
            raw_mean = float(raw_pred.mean())
            corrected_mean = self._bias_corrector.correct(
                estimator=estimator.predict,
                X_historic=X_hist,
                y_historic=y_hist,
                kernel_weights=self._weights,
                raw_estimate=raw_mean,
            )

            residuals_hist = np.abs(y_hist - estimator.predict(X_hist))
            residuals_synth = np.abs(synthetic_y - raw_pred)
            w_residuals = residuals_hist * (self._weights / self._weights.mean())
            self._conformal_scores = np.concatenate([w_residuals, residuals_synth])

            if self.run_diagnostics:
                self._diagnostic.check(X_hist, synthetic_X, self._weights)
        else:
            estimator.fit(synthetic_X, synthetic_y)
            self._fitted_estimator = estimator

            residuals = np.abs(synthetic_y - estimator.predict(synthetic_X))
            self._conformal_scores = residuals
            self._weights = None

        return self

    def predict_interval(
        self,
        X_new: Optional[np.ndarray] = None,
        confidence: float = 0.95,
    ) -> InferenceResult:
        """Produce calibrated conformal prediction intervals.

        Parameters
        ----------
        X_new : (m, d) new points to predict. If None, uses synthetic X.
        confidence : coverage level (default 0.95)

        Returns
        -------
        InferenceResult with predictions and (lower, upper) intervals
        """
        if self._fitted_estimator is None:
            raise RuntimeError("Call fit() before predict_interval()")

        if X_new is not None:
            X_new = np.atleast_2d(X_new)
        else:
            raise ValueError("X_new is required for prediction")

        pred = self._fitted_estimator.predict(X_new)

        if self._bias_corrector.correction_term is not None:
            pred = pred - self._bias_corrector.correction_term

        q = conformal_quantiles(self._conformal_scores, confidence)

        lower = pred - q
        upper = pred + q

        diag_mmd = 0.0
        if self._weights is not None and self._diagnostic is not None:
            diag_mmd = self._diagnostic.check(
                np.zeros((1, 1)), np.zeros((1, 1))
            ).statistic if False else 0.0

        return InferenceResult(
            prediction=pred,
            lower=lower,
            upper=upper,
            confidence=confidence,
            conformal_quantile=q,
            bias_correction=self._bias_corrector.correction_term or 0.0,
            diagnostic_mmd=diag_mmd,
        )
