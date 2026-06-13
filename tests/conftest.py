"""Shared fixtures for task_exchangeable_inference tests."""

import numpy as np
import pytest

from task_exchangeable_inference.repository import Task, TaskRepository


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_beta(rng):
    """True regression coefficients."""
    return rng.standard_normal(3)


@pytest.fixture
def exchangeable_repo(rng, linear_beta):
    """Repository of 5 exchangeable tasks (same DGP, shared beta)."""
    repo = TaskRepository()
    for _ in range(5):
        X = rng.standard_normal((50, 3))
        y = X @ linear_beta + rng.normal(0, 0.3, 50)
        repo.add_task(X, y)
    return repo


@pytest.fixture
def exchangeable_target(rng, linear_beta):
    """A target task drawn from the same DGP as exchangeable_repo."""
    X = rng.standard_normal((40, 3))
    y = X @ linear_beta + rng.normal(0, 0.3, 40)
    return Task(X=X, y=y)


@pytest.fixture
def non_exchangeable_target(rng):
    """A target task from a completely different DGP."""
    X = rng.uniform(5, 10, size=(40, 3))
    y = np.sin(X[:, 0]) * 10 + rng.normal(0, 0.1, 40)
    return Task(X=X, y=y)
