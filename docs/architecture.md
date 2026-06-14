# Architecture

Task Exchange is organized around a small set of modules that form a pipeline:

```mermaid
flowchart LR
    A["Repository"]
    B["Exchangeability test"]
    C["Kernel"]
    D["Inference"]
    E["Bias correction"]
    A --> B
    B --> C
    C --> D
    D --> E
```

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
