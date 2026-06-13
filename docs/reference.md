# 📚 Task‑Exchangeability Library – API Reference

The **task_exchangeable_inference** package implements the full statistical pipeline for *task‑exchangeability* inference on synthetic data.  
All public objects are importable as:

```python
from task_exchangeable_inference.<module> import <name>
```

Below is the complete reference for every public symbol that the library ships.  
Only the functions and classes listed in the technical specification are documented – no hidden or undocumented APIs are exposed.

---

## `task_exchangeable_inference.bias_correction`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def correct(y_pred: np.ndarray, y_true: np.ndarray, *, bias: float = 0.0) -> np.ndarray` | Returns bias‑corrected predictions by subtracting a constant bias term (or adding a learned correction). | ```python\nfrom task_exchangeable_inference.bias_correction import correct\ny_corr = correct(y_pred, y_true, bias=0.12)\n``` |
| `def correct_predictions(predictions: np.ndarray, correction: float) -> np.ndarray` | Applies a scalar correction to a set of predictions. | ```python\nfrom task_exchangeable_inference.bias_correction import correct_predictions\ny_adj = correct_predictions(y_pred, correction=0.05)\n``` |

### Methods (intended for the `BiasCorrector` class – shown for completeness)

| Signature | Description | Example |
|---|---|---|
| `def correction_term(self) -> Optional[float]` | Returns the estimated correction term if the object has been fitted; otherwise `None`. | ```python\ncorr = corrector.correction_term()\n``` |
| `def residuals(self) -> Optional[np.ndarray]` | Returns the residuals (observed – corrected) after fitting; `None` if not fitted. | ```python\nres = corrector.residuals()\n``` |

---

## `task_exchangeable_inference.cli`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def build_parser() -> argparse.ArgumentParser` | Constructs an `argparse.ArgumentParser` with all CLI options for the library (data generation, fitting, prediction, etc.). | ```python\nimport argparse\nparser = build_parser()\nargs = parser.parse_args()\n``` |
| `def main(argv: Optional[Sequence[str]] = None) -> int` | Entry‑point used by the console script. Parses arguments, runs the requested pipeline, and returns an exit code. | ```python\nfrom task_exchangeable_inference.cli import main\nexit_code = main()\n``` |
| `def main()` | **Alias** for the same entry‑point; kept for backward compatibility. | ```python\nif __name__ == \"__main__\":\n    main()\n``` |

---

## `task_exchangeable_inference.diagnostics`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def compute_statistic(X: np.ndarray, y: np.ndarray) -> float` | Computes a diagnostic statistic (e.g., MMD, kernel density ratio) used to assess exchangeability. | ```python\nstat = compute_statistic(X, y)\n``` |
| `def check(stat: float, *, threshold: float = 0.05) -> bool` | Returns `True` if the statistic passes the given threshold, otherwise `False`. | ```python\nok = check(stat, threshold=0.01)\n``` |
| `def raise_if_invalid(stat: float, *, threshold: float = 0.05) -> None` | Raises `ExchangeabilityViolation` when the statistic exceeds the threshold. | ```python\nraise_if_invalid(stat, threshold=0.01)\n``` |

### Exceptions / Warnings

| Name | Description |
|---|---|
| `class ExchangeabilityWarning(UserWarning)` | Issued when a mild violation is detected (e.g., statistic close to threshold). |
| `class ExchangeabilityViolation(Exception)` | Raised when a hard violation occurs (statistic exceeds threshold). |

---

## `task_exchangeable_inference.exchangeability`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def is_exchangeable(X: np.ndarray, y: np.ndarray, *, alpha: float = 0.05) -> bool` | Tests global exchangeability of a dataset using a kernel‑based statistic; returns `True` if the null hypothesis cannot be rejected. | ```python\nfrom task_exchangeable_inference.exchangeability import is_exchangeable\nok = is_exchangeable(X, y, alpha=0.01)\n``` |
| `def pairwise_exchangeability(X: np.ndarray, y: np.ndarray, *, alpha: float = 0.05) -> np.ndarray` | Returns a boolean matrix `M` where `M[i, j]` indicates whether task *i* and *j* are pairwise exchangeable. | ```python\nM = pairwise_exchangeability(X, y)\n``` |

---

## `task_exchangeable_inference.inference`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def fit(X: np.ndarray, y: np.ndarray, *, kernel: str = \"rbf\", **kernel_kwargs) -> dict` | Fits the exchangeability model (kernel, density‑ratio weights, etc.) and returns a dictionary containing fitted objects (e.g., kernel, weights). | ```python\nmodel = fit(X, y, kernel=\"rbf\", gamma=0.5)\n``` |
| `def predict_interval(model: dict, X_new: np.ndarray, *, level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]` | Produces lower/upper prediction intervals for new tasks using the fitted `model`. | ```python\nlower, upper = predict_interval(model, X_new, level=0.95)\n``` |

---

## `task_exchangeable_inference.kernel`

### Class `ExchangeabilityKernel`

| Signature | Description | Example |
|---|---|---|
| `def fit(self, historic_tasks: List[Task]) -> "ExchangeabilityKernel"` | Learns kernel parameters from a list of historic `Task` objects and returns `self` for chaining. | ```python\nkernel = ExchangeabilityKernel().fit(tasks)\n``` |
| `def score(self, target_task: Task) -> np.ndarray` | Computes similarity scores between the `target_task` and each historic task. | ```python\nscores = kernel.score(new_task)\n``` |
| `def task_weights(self, target_task: Task) -> np.ndarray` | Returns normalized weights (e.g., density‑ratio) for historic tasks conditioned on the target. | ```python\nw = kernel.task_weights(new_task)\n``` |
| `def pairwise_matrix(self) -> np.ndarray` | Returns the full pairwise kernel matrix for the historic tasks used during fitting. | ```python\nK = kernel.pairwise_matrix()\n``` |

---

## `task_exchangeable_inference.repository`

### Class `TaskRepository`

| Signature | Description | Example |
|---|---|---|
| `def n_samples(self) -> int` | Number of stored tasks (rows). | ```python\nn = repo.n_samples()\n``` |
| `def n_features(self) -> int` | Dimensionality of the feature space. | ```python\np = repo.n_features()\n``` |
| `def add_task(self, X: np.ndarray, y: np.ndarray, **metadata) -> Task` | Inserts a new task (features `X`, target `y`) with optional metadata; returns a `Task` object. | ```python\ntask = repo.add_task(X_i, y_i, source=\"sim\")\n``` |
| `def get_tasks(self) -> List[Task]` | Returns all stored `Task` objects. | ```python\ntasks = repo.get_tasks()\n``` |
| `def sample_historic(self, n: int, rng: np.random.Generator) -> List[Task]` | Randomly samples `n` historic tasks (with replacement) using the provided RNG. | ```python\nsample = repo.sample_historic(10, rng)\n``` |
| `def pool_features(self) -> np.ndarray` | Returns a matrix of all historic features (`n_samples × n_features`). | ```python\nX_pool = repo.pool_features()\n``` |
| `def pool_targets(self) -> np.ndarray` | Returns a vector of all historic targets. | ```python\ny_pool = repo.pool_targets()\n``` |
| `def summary_statistics(self) -> dict` | Computes basic stats (mean, std, min, max) for features and targets. | ```python\nstats = repo.summary_statistics()\n``` |

---

## `task_exchangeable_inference.utils`

### Functions

| Signature | Description | Example |
|---|---|---|
| `def set_seed(seed: int) -> np.random.Generator` | Creates a reproducible `Generator` seeded with `seed`. | ```python\nrng = set_seed(42)\n``` |
| `def rbf_kernel_matrix(X: np.ndarray, *, gamma: float = 1.0) -> np.ndarray` | Computes the RBF (Gaussian) kernel matrix `K_{ij}=exp(-γ‖x_i‑x_j‖²)`. | ```python\n