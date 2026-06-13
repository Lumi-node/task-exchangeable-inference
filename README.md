<p align="center">
  <img src="assets/hero.jpg" alt="Task Exchange" width="900">
</p>

<h1 align="center">Task Exchange</h1>

<p align="center">
  <strong>Statistical inference with synthetic data via task exchangeability.</strong>
</p>

<p align="center">
  <a href="https://github.com/Lumi-node/task-exchangeable-inference"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
  <a href="https://github.com/Lumi-node/task-exchangeable-inference/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://pypi.org/project/task-exchangeable-inference/"><img src="https://img.shields.io/badge/python-%3E%3D3.10-blue.svg" alt="Python"></a>
  <a href="https://github.com/Lumi-node/task-exchangeable-inference/actions"><img src="https://img.shields.io/badge/tests-41-success.svg" alt="Tests"></a>
</p>

---

Task Exchange provides a principled pipeline for **task‑exchangeability** inference. By treating historic tasks as exchangeable draws from a latent distribution, the library builds kernel‑based representations that enable bias‑corrected predictions on synthetic data. This approach yields calibrated confidence intervals even when the synthetic generator deviates from the true data‑generating process.

The package is pure Python, works with NumPy, SciPy, and scikit‑learn, and ships with a command‑line interface for quick diagnostics. It is ideal for researchers exploring synthetic data validation, causal inference, or any setting where task‑level exchangeability assumptions are relevant.

---

## Quick Start

```bash
pip install task-exchangeable-inference
```

```python
from task_exchangeable_inference.repository import Repository
from task_exchangeable_inference.exchangeability import is_exchangeable
from task_exchangeable_inference.inference import fit, predict_interval

# Load historic tasks
repo = Repository()
repo.add_task(X=np.random.randn(100, 5), y=np.random.randn(100))

# Verify exchangeability
assert is_exchangeable(repo.get_tasks())

# Fit the exchangeability kernel and make predictions
kernel = fit(repo.get_tasks())
preds, intervals = predict_interval(kernel, X_new=np.random.randn(10, 5))
print(preds, intervals)
```

## What Can You Do?

### Bias Correction
```python
from task_exchangeable_inference.bias_correction import correct, correct_predictions

# Correct a single prediction
adj = correct(prediction=0.5, correction_term=0.1)

# Apply correction to an array of predictions
adj_array = correct_predictions(predictions=[0.5, 0.6], correction_terms=[0.1, 0.05])
```

### Diagnostics
```python
from task_exchangeable_inference.diagnostics import compute_statistic, check, ExchangeabilityWarning

stat = compute_statistic(data=np.random.randn(100))
check(stat)  # raises warning if exchangeability is violated
```

### Kernel Estimation
```python
from task_exchangeable_inference.kernel import ExchangeabilityKernel

kernel = ExchangeabilityKernel()
kernel.fit(historic_tasks=repo.get_tasks())
weights = kernel.task_weights(target_task=repo.get_tasks()[0])
pairwise = kernel.pairwise_matrix()
```

## Architecture

```
task_exchangeable_inference
├── __init__.py                # package entry point
├── bias_correction.py         # bias‑correction utilities
├── cli.py                     # command‑line interface
├── diagnostics.py             # validation helpers and warnings
├── exchangeability.py         # core exchangeability tests
├── inference.py               # fitting & interval prediction
├── kernel.py                  # kernel construction & scoring
├── repository.py              # task storage & sampling
└── utils.py                   # low‑level numerical helpers
```

The **Repository** stores historic tasks and provides sampling utilities.  
**exchangeability** functions assess whether tasks satisfy the exchangeability assumption.  
**kernel** builds a kernel matrix from historic tasks, exposing methods to score new tasks and retrieve pairwise relationships.  
**bias_correction** uses the kernel to compute correction terms for predictions.  
**inference** ties everything together, fitting the kernel and producing calibrated prediction intervals.  
The **CLI** (`task_exchangeable_inference.cli`) offers a quick entry point for running diagnostics on a dataset.

## API Reference

### `task_exchangeable_inference.bias_correction`

- `correct(prediction: float, correction_term: float) -> float`
- `correct_predictions(predictions: Sequence[float], correction_terms: Sequence[float]) -> List[float]`
- `correction_term(self) -> Optional[float]`
- `residuals(self) -> Optional[np.ndarray]`

### `task_exchangeable_inference.diagnostics`

- `compute_statistic(data: np.ndarray) -> float`
- `check(statistic: float) -> None`
- `raise_if_invalid(statistic: float) -> None`
- `class ExchangeabilityWarning(UserWarning)`
- `class ExchangeabilityViolation(Exception)`

### `task_exchangeable_inference.exchangeability`

- `is_exchangeable(tasks: Sequence[Task]) -> bool`
- `pairwise_exchangeability(tasks: Sequence[Task]) -> np.ndarray`

### `task_exchangeable_inference.inference`

- `fit(tasks: Sequence[Task]) -> ExchangeabilityKernel`
- `predict_interval(kernel: ExchangeabilityKernel, X_new: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]`

### `task_exchangeable_inference.kernel`

- `fit(self, historic_tasks: List[Task]) -> "ExchangeabilityKernel"`
- `score(self, target_task: Task) -> np.ndarray`
- `task_weights(self, target_task: Task) -> np.ndarray`
- `pairwise_matrix(self) -> np.ndarray`

### `task_exchangeable_inference.repository`

- `add_task(self, X: np.ndarray, y: np.ndarray, **metadata) -> Task`
- `get_tasks(self) -> List[Task]`
- `sample_historic(self, n: int) -> List[Task]`
- `pool_features(self) -> np.ndarray`
- `pool_targets(self) -> np.ndarray`
- `summary_statistics(self) -> dict`
- `n_samples(self) -> int`
- `n_features(self) -> int`

### `task_exchangeable_inference.utils`

- `set_seed(seed: int) -> np.random.Generator`
- `rbf_kernel_matrix(X: np.ndarray, gamma: float) -> np.ndarray`
- `density_ratio_kmm(...)`  *(see source for full signature)*
- `objective(beta: np.ndarray) -> float`
- `gradient(beta: np.ndarray) -> np.ndarray`
- `martingale_residuals(...)`
- `conformal_quantiles(...)`
- `compute_mmd_squared(...)`

## Research Background

Task Exchangeability builds on the theory of **exchangeable sequences** (de Finetti, 1937) and recent work on **kernel mean matching** for covariate shift (Gretton et al., 2009). The bias‑correction framework follows the methodology described in *"Statistical Inference with Synthetic Data via Task Exchangeability"* (Young & Automate Capture Research, 2024). Full references are available in the `docs/` site.

## Testing

The library is covered by **41** unit tests located in the `tests/` directory. Run the test suite with:

```bash
pytest -v
```


## Contributing

We welcome contributions! Please:

1. Fork the repo.
2. Create a feature branch (`git checkout -b feat/my-feature`).
3. Write tests for your changes.
4. Submit a Pull Request.

See `CONTRIBUTING.md` for detailed guidelines.

## Citation

If you use Task Exchange in your research, please cite:

```
Young, A. (2024). Statistical Inference with Synthetic Data via Task Exchangeability.
Automate Capture Research. https://github.com/Lumi-node/task-exchangeable-inference
```

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.