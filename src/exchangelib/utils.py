"""Utility functions for task-exchangeability inference."""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded numpy Generator for reproducible experiments."""
    return np.random.default_rng(seed)


def rbf_kernel_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix.

    Parameters
    ----------
    X : (n, d) array
    Y : (m, d) array, optional. If None, compute K(X, X).
    bandwidth : float, optional. If None, use median heuristic.
    """
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)

    dists_sq = cdist(X, Y, metric="sqeuclidean")

    if bandwidth is None:
        bandwidth = _median_bandwidth(dists_sq)

    return np.exp(-dists_sq / (2.0 * bandwidth**2))


def _median_bandwidth(dists_sq: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth from squared distance matrix."""
    upper = dists_sq[np.triu_indices_from(dists_sq, k=1)] if dists_sq.shape[0] == dists_sq.shape[1] else dists_sq.ravel()
    median_sq = np.median(upper[upper > 0]) if np.any(upper > 0) else 1.0
    return float(np.sqrt(median_sq / 2.0))


def density_ratio_kmm(
    X_source: np.ndarray,
    X_target: np.ndarray,
    bandwidth: Optional[float] = None,
    ridge: float = 1e-3,
    B: float = 10.0,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Kernel Mean Matching density ratio estimation.

    Finds weights beta for source samples such that the weighted source
    distribution matches the target distribution in RKHS.

    Solves: min  beta^T K beta - 2 kappa^T beta
            s.t. 0 <= beta_i <= B
                 |sum(beta) - n| <= n * epsilon

    Parameters
    ----------
    X_source : (n, d) source samples
    X_target : (m, d) target samples
    bandwidth : RBF bandwidth (median heuristic if None)
    ridge : Tikhonov regularisation added to K diagonal
    B : upper bound on individual weights
    epsilon : constraint slack on weight sum

    Returns
    -------
    beta : (n,) importance weights
    """
    X_source = np.atleast_2d(X_source)
    X_target = np.atleast_2d(X_target)
    n = X_source.shape[0]
    m = X_target.shape[0]

    K = rbf_kernel_matrix(X_source, bandwidth=bandwidth)
    K += ridge * np.eye(n)

    K_cross = rbf_kernel_matrix(X_source, X_target, bandwidth=bandwidth)
    kappa = (n / m) * K_cross.sum(axis=1)

    def objective(beta: np.ndarray) -> float:
        return float(0.5 * beta @ K @ beta - kappa @ beta)

    def gradient(beta: np.ndarray) -> np.ndarray:
        return K @ beta - kappa

    beta0 = np.ones(n)
    bounds = [(0.0, B)] * n
    constraints = [
        {"type": "ineq", "fun": lambda b: n * epsilon - (np.sum(b) - n)},
        {"type": "ineq", "fun": lambda b: n * epsilon + (np.sum(b) - n)},
    ]

    result = minimize(
        objective,
        beta0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    return np.maximum(result.x, 0.0)


def martingale_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute martingale residuals for bias correction.

    Martingale residuals M_i = Y_i - E[Y_i | X_i] adjusted by importance
    weights to account for distribution shift between synthetic and real data.

    Parameters
    ----------
    observed : (n,) observed outcomes
    predicted : (n,) predicted/expected outcomes
    weights : (n,) importance weights (uniform if None)

    Returns
    -------
    residuals : (n,) weighted martingale residuals
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    raw = observed - predicted

    if weights is None:
        return raw

    weights = np.asarray(weights, dtype=float)
    w_norm = weights / weights.sum() * len(weights)
    return raw * w_norm


def conformal_quantiles(
    scores: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute conformal prediction quantile under exchangeability.

    Under exchangeability, the (1-alpha)(1 + 1/n) quantile of conformity
    scores gives valid marginal coverage.

    Parameters
    ----------
    scores : (n,) non-conformity scores (absolute residuals)
    confidence : desired coverage level

    Returns
    -------
    q : the conformal quantile threshold
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    alpha = 1.0 - confidence
    level = np.ceil((1.0 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    idx = int(np.ceil(level * n)) - 1
    idx = min(idx, n - 1)
    return float(scores[idx])


def compute_mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: Optional[float] = None,
) -> float:
    """Compute the squared Maximum Mean Discrepancy between two samples.

    MMD^2(X, Y) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Uses unbiased U-statistic estimator with shared bandwidth.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n, m = X.shape[0], Y.shape[0]

    if bandwidth is None:
        pooled = np.vstack([X, Y])
        dists_sq = cdist(pooled, pooled, metric="sqeuclidean")
        bandwidth = _median_bandwidth(dists_sq)

    Kxx = rbf_kernel_matrix(X, bandwidth=bandwidth)
    Kyy = rbf_kernel_matrix(Y, bandwidth=bandwidth)
    Kxy = rbf_kernel_matrix(X, Y, bandwidth=bandwidth)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        - 2.0 * Kxy.sum() / (n * m)
        + Kyy.sum() / (m * (m - 1))
    )
    return float(mmd2)
