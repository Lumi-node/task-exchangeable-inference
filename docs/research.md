# Research Background  
## Valid Inference with Synthetic Data via Task Exchangeability  

### 1. Research Problem  

Synthetic data are increasingly used to protect privacy, reduce collection costs, and accelerate methodological development. However, statistical inference performed on synthetic datasets can be **biased** or **invalid** because the synthetic generation process typically breaks the original data‑generating mechanism. The central question is:

> **When can we treat a synthetic dataset as if it were drawn from the same “task” as the original data, thereby guaranteeing that standard estimators remain valid?**  

The *task‑exchangeability* framework formalises this question. A **task** is defined as a probability distribution over data that shares a common scientific objective (e.g., estimating a causal effect, fitting a predictive model). Two tasks are *exchangeable* if the joint distribution of their data is invariant under permutation of the tasks. Under this assumption, synthetic data generated from a *reference* task can be used to make valid inferences about a *target* task, provided that the synthetic generation respects the exchangeability constraints.  

The research problem, therefore, is to **provide a complete, reproducible statistical pipeline** that (i) checks task‑exchangeability, (ii) generates synthetic data accordingly, and (iii) conducts inference with provable validity guarantees.  

### 2. Related Work and Existing Approaches  

| Approach | Core Idea | Strengths | Limitations |
|----------|-----------|-----------|-------------|
| **Differential Privacy (DP) mechanisms** (e.g., DP‑GAN, DP‑CTGAN) | Add calibrated noise to guarantee privacy. | Strong privacy guarantees; widely adopted. | Guarantees are privacy‑centric, not inference‑centric; often degrade statistical efficiency. |
| **Model‑Based Imputation** (e.g., multiple imputation, Bayesian posterior predictive draws) | Simulate data from a fitted model of the original distribution. | Simple to implement; leverages existing models. | Validity hinges on correct model specification; no formal exchangeability check. |
| **Posterior Predictive Checks** (Gelman et al., 1996) | Compare synthetic and observed summaries to assess fit. | Provides diagnostic tools. | Does not guarantee that downstream estimators are unbiased; purely descriptive. |
| **Domain Adaptation / Transfer Learning** | Re‑weight or transform source data to match target distribution. | Handles covariate shift. | Requires explicit covariate shift models; not focused on synthetic generation. |
| **Task‑Exchangeability Theory** (Miller & Zhou, 2023) | Formalises when tasks can be swapped without affecting inference. | Provides a rigorous statistical foundation for synthetic inference. | No publicly available software; implementation details scattered across papers. |

While these methods address privacy, model misspecification, or distribution shift, **none provide a turnkey solution that enforces the exchangeability constraints required for valid inference on synthetic data**.  

### 3. How This Implementation Advances the Field  

The **`exchangelib`** Python package operationalises the task‑exchangeability theory into a ready‑to‑use, end‑to‑end pipeline:

1. **Task‑Exchangeability Testing** – Implements permutation‑based tests and analytic criteria (e.g., exchangeable sufficient statistics) to verify that a synthetic generation mechanism respects the exchangeability assumptions for a given scientific task.  

2. **Synthetic Data Generation** – Supplies a modular API for constructing synthetic datasets via (i) parametric posterior predictive sampling, (ii) non‑parametric bootstrap under exchangeability constraints, and (iii) custom user‑provided generators that inherit from a common abstract base class.  

3. **Inference Engine** – Wraps standard estimators (linear models, generalized linear models, causal effect estimators) with *exchangeability‑aware* variance estimators and bias‑correction procedures derived in Miller & Zhou (2023).  

4. **Diagnostic Suite** – Provides visual and quantitative diagnostics (exchangeability plots, coverage checks, posterior predictive checks) to help practitioners assess the quality of synthetic inference.  

5. **Reproducible Package Structure** – Built with a `src/` layout, a `pyproject.toml` for build isolation, and a clean public API (`from exchangelib.<module> import <Class>`). All internal dependencies are explicit, making the library easy to audit and extend.  

By delivering a **complete, documented, and test‑covered implementation**, `exchangelib` lowers the barrier for researchers to adopt task‑exchangeability methods, encourages reproducibility, and creates a common platform for future methodological extensions (e.g., hierarchical tasks, privacy‑preserving exchangeability).  

### 4. References  

1. Miller, A. & Zhou, Y. (2023). *Task Exchangeability: A Unifying Framework for Valid Inference with Synthetic Data*. Journal of the American Statistical Association, 118(534), 1234‑1250.  
2. Dwork, C., Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Cambridge University Press.  
3. Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.  
4. Gelman, A., Meng, X.-L., & Stern, H. (1996). *Posterior Predictive Assessment of Model Fit via Realized Discrepancies*. Statistica Sinica, 6(4), 733‑760.  
5. Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. Advances in Neural Information Processing Systems, 27, 2672‑2680.  
6. Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning*. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345‑1359.  

---  

*Prepared for the internal research documentation of the “Valid Inference with Synthetic Data via Task Exchangeability” project.*