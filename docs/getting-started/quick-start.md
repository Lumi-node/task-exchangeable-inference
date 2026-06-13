# Quick‑Start Guide – **exchangelib**

*Task‑exchangeability* inference is a statistical framework for leveraging historic tasks (e.g., previous experiments, past runs of a model) to make calibrated predictions on a new target task. The **exchangelib** package implements the full pipeline:

1. **Repository** – store and sample historic tasks.  
2. **Exchangeability checks** – diagnose whether historic tasks are exchangeable with the target.  
3. **Kernel weighting** – compute importance weights (KMM) for each historic task.  
4. **Bias correction** – adjust predictions for any remaining systematic error.  
5. **Inference** – fit a predictive model and produce calibrated prediction intervals.  

Below you will find everything you need to get started: installation, a minimal end‑to‑end example, and a couple of more advanced snippets.

---

## 1. Installation

```bash
# Clone the repo (or download the source)
git clone https://github.com/yourorg/exchangelib.git
cd exchangelib

# Install in editable mode (includes the test dependencies)
pip install -e .
```

> **Tip** – The package follows the *src* layout, so the import path is simply `exchangelib`.

---

## 2. Minimal End‑to‑End Example

```python
import numpy as np
from exchangelib.repository import Repository
from exchangelib.exchangeability import is_exchangeable
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.bias_correction import correct, correct_predictions
from exchangelib.inference import fit, predict_interval
from exchangelib.diagnostics import ExchangeabilityWarning

# -------------------------------------------------
# 1️⃣  Build a repository of historic tasks
# -------------------------------------------------
repo = Repository()
rng = np.random.default_rng(42)

# Simulate 20 historic tasks, each with 100 samples and 5 features
for i in range(20):
    X = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    # Linear relationship with task‑specific slope
    beta = rng.normal(scale=0.5, size=5)
    y = X @ beta + rng.normal(scale=0.2, size=100)
    repo.add_task(X, y, task_id=f"task_{i}")

# -------------------------------------------------
# 2️⃣  Load the target task (new data)
# -------------------------------------------------
X_target = rng.normal(loc=0.1, scale=1.0, size=(80, 5))
y_target = X_target @ rng.normal(scale=0.5, size=5) + rng.normal(scale=0.2, size=80)

# -------------------------------------------------
# 3️⃣  Diagnose exchangeability
# -------------------------------------------------
if not is_exchangeable(repo.get_tasks(), X_target, y_target):
    # The diagnostics module will raise a warning we can catch
    print("⚠️  Historic tasks may not be exchangeable with the target.")
else:
    print("✅  Exchangeability holds – proceeding with weighting.")

# -------------------------------------------------
# 4️⃣  Fit the kernel that yields task‑specific weights
# -------------------------------------------------
kernel = ExchangeabilityKernel()
kernel = kernel.fit(repo.get_tasks())          # learns a kernel on historic tasks
weights = kernel.task_weights(target_task=X_target)   # shape: (n_historic,)

# -------------------------------------------------
# 5️⃣  Fit a predictive model on the weighted historic data
# -------------------------------------------------
model = fit(repo, weights)                     # returns a scikit‑learn‑like estimator

# -------------------------------------------------
# 6️⃣  Obtain calibrated prediction intervals
# -------------------------------------------------
lower, upper = predict_interval(model, X_target, alpha=0.10)   # 90 % interval

# -------------------------------------------------
# 7️⃣  (Optional) Bias‑correction of the intervals
# -------------------------------------------------
# Compute the correction term (if any) and adjust the bounds
corr = correct(model, X_target, y_target)      # returns a scalar or None
if corr is not None:
    lower, upper = correct_predictions(lower, upper, corr)

print(f"Prediction interval for the target task: [{lower.mean():.3f}, {upper.mean():.3f}]")
```

### What the script does

| Step | Function(s) used | Purpose |
|------|------------------|---------|
| 1️⃣   | `Repository.add_task` | Store synthetic historic tasks. |
| 2️⃣   | – | Load the new (target) data. |
| 3️⃣   | `is_exchangeable` | Quick diagnostic; raises `ExchangeabilityWarning` if violated. |
| 4️⃣   | `ExchangeabilityKernel.fit` & `ExchangeabilityKernel.task_weights` | Compute importance weights via kernel mean matching (KMM). |
| 5️⃣   | `fit` | Fit a regression model (by default a ridge regression) on the weighted historic pool. |
| 6️⃣   | `predict_interval` | Produce conformal prediction intervals. |
| 7️⃣   | `correct`, `correct_predictions` | Optional bias‑correction of the interval endpoints. |

---

## 3. More Detailed Usage Patterns

### 3.1. Sampling Historic Tasks

Sometimes you want a *random* subset of historic tasks (e.g., for bootstrap analysis).

```python
from exchangelib.repository import Repository

repo = Repository()
# ... (add many tasks) ...

# Sample 10 historic tasks uniformly at random
sampled = repo.sample_historic(n=10, random_state=123)
print(f"Sampled {len(sampled)} tasks.")
```

### 3.2. Pairwise Exchangeability Matrix

The `exchangeability` module can also compute a full pairwise matrix, useful for visualisation.

```python
from exchangelib.exchangeability import pairwise_exchangeability
import matplotlib.pyplot as plt

tasks = repo.get_tasks()
matrix = pairwise_exchangeability(tasks)   # shape (n_tasks, n_tasks)

plt.imshow(matrix, cmap="viridis")
plt.title("Pairwise Exchangeability")
plt.colorbar(label="p‑value")
plt.show()
```

### 3.3. Custom Kernel & Diagnostics

If you need a custom kernel (e.g., a different bandwidth), you can instantiate the kernel directly and use the diagnostics utilities to validate the kernel matrix.

```python
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.diagnostics import compute_statistic, raise_if_invalid

kernel = ExchangeabilityKernel(bandwidth=0.8)   # custom bandwidth
kernel = kernel.fit(repo.get_tasks())

# Validate the kernel matrix (e.g., ensure PSD)
K = kernel.pairwise_matrix()
stat = compute_statistic(K)
raise_if_invalid(stat)   # raises ExchangeabilityViolation if something is wrong
```

---

## 4. Command‑Line Interface (CLI)

The package ships with a tiny CLI for quick experiments.

```bash
# Show help
python -m exchangelib.cli --help

#