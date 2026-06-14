# API Reference

`Task Exchange` is imported as `task_exchangeable_inference`. The public API:

```python
import task_exchangeable_inference
```

### `ExchangeabilityKernel`

Key methods: `fit()`, `pairwise_matrix()`, `score()`, `task_weights()`

### `ExchangeabilityModel`

Key methods: `is_exchangeable()`, `pairwise_exchangeability()`

### `ExchangeableInference`

Key methods: `fit()`, `predict_interval()`

### `TaskRepository`

Key methods: `add_task()`, `get_tasks()`, `pool_features()`, `pool_targets()`, `sample_historic()`

