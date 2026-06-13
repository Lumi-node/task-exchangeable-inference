# Architecture of **Task‑Exchangeability** (`exchangelib`)

## 1. System Overview  

`exchangelib` implements the full statistical pipeline for *task‑exchangeability* inference on synthetic (or real) data.  The library is organized as a classic **src‑layout** Python package and provides a clean public API that can be imported as

```python
from exchangelib.bias_correction import correct, correct_predictions
from exchangelib.cli import build_parser, main
from exchangelib.diagnostics import compute_statistic, check, raise_if_invalid
from exchangelib.exchangeability import is_exchangeable, pairwise_exchangeability
from exchangelib.inference import fit, predict_interval
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.repository import Repository
from exchangelib.utils import set_seed, rbf_kernel_matrix, density_ratio_kmm, ...
```

The core idea is to treat a collection of **tasks** (datasets) as exchangeable objects.  A *kernel* is learned on historic tasks, a *bias‑correction* term is estimated, and finally conformal prediction intervals are produced for a new target task.  The modules are deliberately small and composable, making it easy to replace any component (e.g., the kernel, the bias‑correction strategy, or the diagnostics) without touching the rest of the code‑base.

---

## 2. Module Relationship Diagram  

```mermaid
graph TD
    %% Packages
    subgraph exchangelib
        A[utils.py] 
        B[repository.py] 
        C[kernel.py] 
        D[exchangeability.py] 
        E[diagnostics.py] 
        F[bias_correction.py] 
        G[inference.py] 
        H[cli.py]
    end

    %% Data objects
    Task[Task (namedtuple)]

    %% Relationships
    B -->|creates| Task
    B -->|samples historic| C
    C -->|uses| A
    C -->|produces| Kernel[ExchangeabilityKernel]
    D -->|calls| C
    D -->|calls| A
    E -->|validates| Task
    F -->|corrects| G
    G -->|fits| C
    G -->|uses| F
    G -->|uses| D
    G -->|uses| E
    H -->|CLI orchestrates| G
    H -->|CLI orchestrates| F
    H -->|CLI orchestrates| D
    H -->|CLI orchestrates| B
```

*Arrows indicate import/use dependencies; the direction is “imports → uses”.*  

- **`utils.py`** supplies low‑level numerical helpers (random seed, kernels, density‑ratio estimation, etc.).  
- **`repository.py`** stores tasks and provides sampling utilities.  
- **`kernel.py`** builds the *ExchangeabilityKernel* from historic tasks.  
- **`exchangeability.py`** contains the statistical tests that decide whether a new task is exchangeable with the historic pool.  
- **`diagnostics.py`** offers generic checks, warnings, and exception types used throughout the pipeline.  
- **`bias_correction.py`** implements the bias‑correction estimator and helper functions.  
- **`inference.py`** is the high‑level entry point that fits the kernel, applies bias correction, and returns conformal prediction intervals.  
- **`cli.py`** provides a small command‑line interface that wires all pieces together.

---

## 3. Module‑by‑Module Description  

### `src/exchangelib/utils.py`  
Utility functions that are pure‑numpy/scipy helpers:

| Function | Purpose |
|----------|---------|
| `set_seed(seed)` | Returns a reproducible `np.random.Generator`. |
| `rbf_kernel_matrix(X, gamma)` | Computes an RBF (Gaussian) kernel matrix. |
| `density_ratio_kmm(X_src, X_tgt, **kwargs)` | Implements Kernel Mean Matching to estimate importance weights (β). |
| `objective(beta)` / `gradient(beta)` | Objective and gradient for the KMM optimisation. |
| `martingale_residuals(y, y_hat)` | Residuals used in bias‑correction. |
| `conformal_quantiles(residuals, alpha)` | Quantile calculation for conformal intervals. |
| `compute_mmd_squared(X, Y, kernel)` | Maximum‑Mean‑Discrepancy squared between two samples. |

All functions are stateless and depend only on explicit inputs, making them easy to test and reuse.

---

### `src/exchangelib/repository.py`  
A lightweight in‑memory repository for tasks.

| Method | Description |
|--------|-------------|
| `n_samples()` | Total number of stored tasks. |
| `n_features()` | Dimensionality of the feature space (assumes all tasks share the same `X` shape). |
| `add_task(X, y, **metadata)` | Stores a new task and returns a `Task` named‑tuple (`X`, `y`, `metadata`). |
| `get_tasks()` | Returns a list of all stored `Task` objects. |
| `sample_historic(n)` | Randomly samples `n` historic tasks (used for kernel fitting). |
| `pool_features()` / `pool_targets()` | Concatenated feature/target matrices across all historic tasks. |
| `summary_statistics()` | Basic stats (means, variances) for quick diagnostics. |

The repository isolates data handling from the statistical core, allowing the same API to be backed by a database or a remote store in the future.

---

### `src/exchangelib/kernel.py`  
Implements the **ExchangeabilityKernel** class that learns a similarity measure over tasks.

```python
class ExchangeabilityKernel:
    def fit(self, historic_tasks: List[Task]) -> "ExchangeabilityKernel"
    def score(self, target_task: Task) -> np.ndarray
    def task_weights(self, target_task: Task) -> np.ndarray
    def pairwise_matrix(self) -> np.ndarray
```

- `fit` builds a kernel matrix on historic tasks (using `utils.rbf_kernel_matrix` or a custom kernel).  
- `score` evaluates the kernel between each historic task and a new target task, returning a similarity vector.  
- `task_weights` normalises the scores into a probability distribution (used for importance weighting).  
- `pairwise_matrix` returns the full historic‑historic kernel matrix, useful for diagnostics (e.g., MMD).

The class is deliberately immutable after `fit`; subsequent calls only read the stored matrices.

---

### `src/exchangelib/exchangeability.py`  
Statistical tests that decide whether a target task can be considered exchangeable with the historic pool.

| Function | Signature | Role |
|----------|-----------|------|
| `is_exchangeable(target_task, historic_tasks, alpha=0.05)` | Returns `True/False` based on a two‑sample test (e.g., MMD). |
| `pairwise_exchangeability(task_a, task_b)` | Returns a p‑value for the exchangeability of two individual tasks. |

Both functions rely on `utils.compute_mmd_squared` and raise `ExchangeabilityViolation` (or emit `ExchangeabilityWarning`) when the null hypothesis is rejected.

---

### `src/exchangelib/diagnostics.py`  
General purpose validation utilities.

| Function / Class | Purpose |
|------------------|---------|
| `compute_statistic(task, statistic)` | Wrapper to compute a user‑provided statistic on a task. |
| `check(condition, msg)` | Simple assertion that raises `ExchangeabilityViolation` if `condition` is `False`. |
| `raise_if_invalid(task, validator)` | Calls a validator and raises the appropriate exception. |
| `ExchangeabilityWarning` | Subclass of `UserWarning` for non‑fatal exchangeability concerns. |
| `ExchangeabilityViolation` | Custom exception used throughout the pipeline. |

These helpers keep the core modules tidy and provide a single place for future extensions (e.g., logging, metrics).

---

### `src/exchangelib/bias_correction.py`  
Implements the bias‑correction step that adjusts predictions for systematic drift between historic and target tasks.

| Function / Method | Signature | Description |
|-------------------|-----------|-------------|
| `correct(predictions, residuals, correction_term=None)` | Returns bias‑corrected predictions. |
| `correct_predictions(predictions, beta, correction_term=None)` | Applies importance weighting (`beta`) before correction. |
| `correction_term(self) -> Optional[float]` | Computes the scalar correction term (e.g., mean residual). |
| `residuals(self) -> Optional[np.ndarray]` | Returns residuals from a fitted model (used internally). |

The module can be used stand‑alone or via `inference.fit`, which automatically estimates the correction term from historic residuals.

---

### `src/exchangelib/inference.py`  
High‑level API that a user calls to obtain conformal prediction intervals for a new task.

| Function | Signature | What it does |
|----------|-----------|--------------|
| `fit(repository, target_task, alpha=0.1, bias=True)` | Returns a fitted `ExchangeabilityKernel` and optional bias‑correction object. |
| `predict_interval(fitted_kernel, target