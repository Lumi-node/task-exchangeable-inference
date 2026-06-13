"""Formal exchangeability model with de Finetti representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from task_exchangeable_inference.repository import Task
from task_exchangeable_inference.utils import compute_mmd_squared, rbf_kernel_matrix


@dataclass
class ExchangeabilityResult:
    """Result of an exchangeability test."""
    is_exchangeable: bool
    p_value: float
    mmd_statistic: float
    threshold: float
    n_permutations: int


class ExchangeabilityModel:
    """Formal exchangeability test based on de Finetti's theorem.

    Under de Finetti's representation, an exchangeable sequence of tasks
    is conditionally i.i.d. given a latent mixing measure. This class
    tests whether a target task is exchangeable with a set of historic
    tasks by computing the Maximum Mean Discrepancy (MMD) between the
    target and historic distributions, with significance assessed via
    a permutation test.

    Parameters
    ----------
    bandwidth : float, optional
        RBF kernel bandwidth. Median heuristic if None.
    n_permutations : int
        Number of permutations for the test.
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        n_permutations: int = 500,
    ) -> None:
        self.bandwidth = bandwidth
        self.n_permutations = n_permutations

    def is_exchangeable(
        self,
        target_task: Task,
        historic_tasks: List[Task],
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> ExchangeabilityResult:
        """Test if target_task is exchangeable with historic_tasks.

        Uses a two-sample MMD permutation test. The null hypothesis is
        that target and historic data come from the same distribution
        (exchangeable). We reject if p < alpha.

        Parameters
        ----------
        target_task : Task
        historic_tasks : list of Task
        alpha : significance level
        seed : random seed for permutation test

        Returns
        -------
        ExchangeabilityResult with boolean, p-value, and diagnostics
        """
        if not historic_tasks:
            raise ValueError("Need at least one historic task")

        X_target = target_task.X
        X_historic = np.vstack([t.X for t in historic_tasks])

        observed_mmd = compute_mmd_squared(X_historic, X_target, self.bandwidth)

        rng = np.random.default_rng(seed)
        X_pooled = np.vstack([X_historic, X_target])
        n_h = X_historic.shape[0]
        n_total = X_pooled.shape[0]

        null_mmds = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = rng.permutation(n_total)
            X_a = X_pooled[perm[:n_h]]
            X_b = X_pooled[perm[n_h:]]
            null_mmds[i] = compute_mmd_squared(X_a, X_b, self.bandwidth)

        p_value = float(np.mean(null_mmds >= observed_mmd))

        sorted_null = np.sort(null_mmds)
        threshold_idx = int(np.ceil((1.0 - alpha) * self.n_permutations)) - 1
        threshold = float(sorted_null[min(threshold_idx, len(sorted_null) - 1)])

        return ExchangeabilityResult(
            is_exchangeable=(p_value >= alpha),
            p_value=p_value,
            mmd_statistic=observed_mmd,
            threshold=threshold,
            n_permutations=self.n_permutations,
        )

    def pairwise_exchangeability(
        self,
        tasks: List[Task],
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Test pairwise exchangeability among a list of tasks.

        Returns
        -------
        matrix : (n, n) boolean matrix where True means exchangeable
        """
        n = len(tasks)
        matrix = np.ones((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                result = self.is_exchangeable(
                    tasks[i], [tasks[j]], alpha=alpha, seed=seed
                )
                matrix[i, j] = result.is_exchangeable
                matrix[j, i] = result.is_exchangeable
        return matrix
