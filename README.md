<p align="center">
  <img src="assets/hero.jpg" alt="Task Exchange" width="900">
</p>

<h1 align="center">Task Exchange</h1>

<p align="center"><strong>Statistical inference with synthetic data via task exchangeability.</strong></p>

<p align="center">
  <a href="https://github.com/Lumi-node/task-exchangeable-inference"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
  <a href="https://github.com/Lumi-node/task-exchangeable-inference/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Lumi-node/task-exchangeable-inference/actions"><img src="https://img.shields.io/badge/tests-41-success.svg" alt="Tests"></a>
  <a href="https://lumi-node.github.io/task-exchangeable-inference/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Docs"></a>
</p>

---

`task-exchangeable-inference` provides valid statistical inference when your data are synthetic. It tests whether real and synthetic tasks are exchangeable, builds a kernel‑reweighted reference distribution, applies bias correction, and returns confidence intervals with proper coverage.

## Installation

```bash
pip install git+https://github.com/Lumi-node/task-exchangeable-inference.git
```

Requires Python ≥ 3.10. To work on the project locally:

```bash
git clone https://github.com/Lumi-node/task-exchangeable-inference.git
cd task-exchangeable-inference
pip install -e ".[dev]"
pytest -q
```

## Quick Start

```python
import numpy as np
from task_exchangeable_inference import ExchangeableInference

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + rng.standard_normal(100) * 0.1

# Fit the exchangeability-corrected estimator on synthetic data
engine = ExchangeableInference()
engine.fit(X, y)

# Predict with calibrated 95% intervals
result = engine.predict_interval(rng.standard_normal((5, 3)), confidence=0.95)
print(result)
```

## Features

- **Exchangeability testing** between real and synthetic tasks
- **Kernel‑reweighted** reference distribution
- **Bias correction** for synthetic‑data estimates
- **Confidence intervals** with diagnostics

## Modules

| Module | Description |
|--------|-------------|
| `bias_correction` | Martingale-based bias correction for synthetic estimators. |
| `cli` | Command-line interface for task_exchangeable_inference. |
| `diagnostics` | Runtime diagnostics for exchangeability assumptions. |
| `exchangeability` | Formal exchangeability model with de Finetti representation. |
| `inference` | Main inference engine for task-exchangeability with synthetic data. |
| `kernel` | Exchangeability kernel via Kernel Mean Matching. |
| `repository` | Task repository for storing and sampling historic tasks. |
| `utils` | Utility functions for task-exchangeability inference. |

## Documentation

📖 Full documentation: [https://lumi-node.github.io/task-exchangeable-inference/](https://lumi-node.github.io/task-exchangeable-inference/)
📄 Technical paper: see [`paper/`](paper/) for the LaTeX source and compiled PDF.

> This is a reference implementation produced by an autonomous research pipeline. It is not published to PyPI; install from source as shown above.

## License

[MIT](LICENSE) © Andrew Young / Automate Capture Research
