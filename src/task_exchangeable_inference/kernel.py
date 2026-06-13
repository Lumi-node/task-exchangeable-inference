"""Exchangeability kernel via Kernel Mean Matching."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from task_exchangeable_inference.repository import Task
from task_exchangeable_inference.utils import density_ratio_kmm, rbf_kernel_matrix


class ExchangeabilityKernel:
    """Semi-parametric kernel for measuring task exchangeability.

    Uses Kernel Mean Matching (KMM) to estimate importance weights that
    re-weight historic task distributions to match a target task in RKHS.
    The quality of the match (low residual norm) indicates exchangeability.

    Parameters
    ----------
    bandwidth : float, optional
        RBF kernel bandwidth. Median heuristic if None.
    ridge : float
        Tikhonov regularisation on the kernel matrix diagonal.
    B : float
        Upper bound on individual KMM weights.
    epsilon : float
        Slack on the sum-to-n constraint.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        ridge: float = 1e-3,
        B: float = 10.0,
        epsilon: float = 0.1,
    ) -> None:
        self.bandwidth = bandwidth
        self.ridge = ridge
        self.B = B
        self.epsilon = epsilon
        self._historic_features: Optional[np.ndarray] = None
        self._task_boundaries: Optional[List[int]] = None
        self._fitted = False

    def fit(self, historic_tasks: List[Task]) -> "ExchangeabilityKernel":
        """Fit the kernel on historic tasks.

        Pools all historic task features and records task boundaries
        so per-task weights can be computed later.
        """
        if not historic_tasks:
            raise ValueError("Need at least one historic task")

        boundaries = [0]
        for t in historic_tasks:
            boundaries.append(boundaries[-1] + t.n_samples)

        self._historic_features = np.vstack([t.X for t in historic_tasks])
        self._task_boundaries = boundaries
        self._fitted = True
        return self

    def score(self, target_task: Task) -> np.ndarray:
        """Compute KMM importance weights for each historic sample.

        Returns weights beta such that the weighted historic distribution
        approximates the target task distribution in RKHS.

        Parameters
        ----------
        target_task : Task
            The target task whose distribution we want to match.

        Returns
        -------
        weights : (n_historic_samples,) importance weights
        """
        self._check_fitted()
        return density_ratio_kmm(
            X_source=self._historic_features,
            X_target=target_task.X,
            bandwidth=self.bandwidth,
            ridge=self.ridge,
            B=self.B,
            epsilon=self.epsilon,
        )

    def task_weights(self, target_task: Task) -> np.ndarray:
        """Compute per-task aggregate weights (mean weight per task).

        Returns
        -------
        task_weights : (n_tasks,) mean importance weight for each historic task
        """
        self._check_fitted()
        sample_weights = self.score(target_task)
        n_tasks = len(self._task_boundaries) - 1
        tw = np.zeros(n_tasks)
        for i in range(n_tasks):
            start = self._task_boundaries[i]
            end = self._task_boundaries[i + 1]
            tw[i] = sample_weights[start:end].mean()
        return tw

    def pairwise_matrix(self) -> np.ndarray:
        """Compute the kernel matrix over pooled historic features.

        Returns
        -------
        K : (n_total, n_total) RBF kernel matrix
        """
        self._check_fitted()
        return rbf_kernel_matrix(self._historic_features, bandwidth=self.bandwidth)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using the kernel")
