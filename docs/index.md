# Task Exchange

**Statistical inference with synthetic data via task exchangeability.**

`task-exchangeable-inference` provides valid statistical inference when your data are synthetic. It tests whether real and synthetic tasks are exchangeable, builds a kernel‑reweighted reference distribution, applies bias correction, and returns confidence intervals with proper coverage.

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

See [Installation](getting-started/installation.md) and the [Quick Start guide](getting-started/quick-start.md) to go further, or the [API Reference](reference.md).
