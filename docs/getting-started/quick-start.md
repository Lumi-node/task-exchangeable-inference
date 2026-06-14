# Quick Start

The following example runs end‑to‑end against the installed package:

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

For the full public API, see the [API Reference](../reference.md). For how the
pieces fit together, see [Architecture](../architecture.md).
