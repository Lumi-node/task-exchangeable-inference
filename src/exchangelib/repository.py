"""Task repository for storing and sampling historic tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from exchangelib.utils import set_seed


@dataclass
class Task:
    """A single supervised-learning task (X, y) pair."""
    X: np.ndarray
    y: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.atleast_2d(np.asarray(self.X, dtype=float))
        self.y = np.asarray(self.y, dtype=float).ravel()
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[0]} samples but y has {self.y.shape[0]}"
            )

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


class TaskRepository:
    """Store and sample historic tasks for exchangeability analysis."""

    def __init__(self) -> None:
        self._tasks: List[Task] = []

    def add_task(self, X: np.ndarray, y: np.ndarray, **metadata) -> Task:
        """Add a task (X, y) to the repository."""
        task = Task(X=X, y=y, metadata=metadata)
        self._tasks.append(task)
        return task

    def get_tasks(self) -> List[Task]:
        """Return all stored tasks."""
        return list(self._tasks)

    def sample_historic(
        self,
        num: int,
        seed: Optional[int] = None,
    ) -> List[Task]:
        """Sample `num` tasks uniformly at random (with replacement)."""
        if not self._tasks:
            raise ValueError("Repository is empty")
        rng = set_seed(seed) if seed is not None else np.random.default_rng()
        indices = rng.integers(0, len(self._tasks), size=num)
        return [self._tasks[i] for i in indices]

    def __len__(self) -> int:
        return len(self._tasks)

    def pool_features(self) -> np.ndarray:
        """Vertically stack all task features into a single (N, d) array."""
        if not self._tasks:
            raise ValueError("Repository is empty")
        return np.vstack([t.X for t in self._tasks])

    def pool_targets(self) -> np.ndarray:
        """Concatenate all task targets into a single (N,) array."""
        if not self._tasks:
            raise ValueError("Repository is empty")
        return np.concatenate([t.y for t in self._tasks])

    def summary_statistics(self) -> dict:
        """Compute summary statistics across all tasks."""
        X = self.pool_features()
        y = self.pool_targets()
        return {
            "n_tasks": len(self._tasks),
            "total_samples": X.shape[0],
            "n_features": X.shape[1],
            "X_mean": X.mean(axis=0),
            "X_std": X.std(axis=0),
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
        }
