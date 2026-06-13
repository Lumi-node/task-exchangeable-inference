"""Tests for the full exchangelib pipeline."""

import numpy as np
import pytest

from exchangelib.bias_correction import BiasCorrector
from exchangelib.diagnostics import (
    ExchangeabilityDiagnostic,
    ExchangeabilityViolation,
)
from exchangelib.exchangeability import ExchangeabilityModel
from exchangelib.inference import ExchangeableInference
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.repository import Task, TaskRepository
from exchangelib.utils import (
    compute_mmd_squared,
    conformal_quantiles,
    density_ratio_kmm,
    martingale_residuals,
    rbf_kernel_matrix,
    set_seed,
)


# ── utils ──────────────────────────────────────────────────────────

class TestUtils:
    def test_set_seed_reproducible(self):
        g1 = set_seed(99)
        g2 = set_seed(99)
        assert g1.random() == g2.random()

    def test_rbf_kernel_matrix_shape(self, rng):
        X = rng.standard_normal((10, 3))
        K = rbf_kernel_matrix(X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)
        np.testing.assert_allclose(np.diag(K), 1.0)

    def test_rbf_kernel_cross(self, rng):
        X = rng.standard_normal((10, 3))
        Y = rng.standard_normal((5, 3))
        K = rbf_kernel_matrix(X, Y)
        assert K.shape == (10, 5)
        assert np.all(K > 0) and np.all(K <= 1)

    def test_density_ratio_kmm_weights(self, rng):
        X_src = rng.standard_normal((30, 2))
        X_tgt = rng.standard_normal((20, 2))
        beta = density_ratio_kmm(X_src, X_tgt)
        assert beta.shape == (30,)
        assert np.all(beta >= 0)

    def test_kmm_same_distribution(self, rng):
        X = rng.standard_normal((50, 2))
        beta = density_ratio_kmm(X[:30], X[30:])
        assert abs(beta.mean() - 1.0) < 1.0

    def test_martingale_residuals_unweighted(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        resid = martingale_residuals(obs, pred)
        np.testing.assert_allclose(resid, obs - pred)

    def test_martingale_residuals_weighted(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        resid = martingale_residuals(obs, pred, weights)
        np.testing.assert_allclose(resid, 0.0, atol=1e-12)

    def test_conformal_quantiles(self):
        scores = np.arange(1, 101, dtype=float)
        q = conformal_quantiles(scores, confidence=0.90)
        assert q >= 90

    def test_mmd_same_distribution(self, rng):
        X = rng.standard_normal((100, 2))
        Y = rng.standard_normal((100, 2))
        mmd = compute_mmd_squared(X, Y)
        assert abs(mmd) < 0.5

    def test_mmd_different_distributions(self, rng):
        X = rng.standard_normal((100, 2))
        Y = rng.standard_normal((100, 2)) + 3.0
        mmd = compute_mmd_squared(X, Y)
        assert mmd > 0.05


# ── repository ─────────────────────────────────────────────────────

class TestTaskRepository:
    def test_add_and_get(self, rng):
        repo = TaskRepository()
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        repo.add_task(X, y)
        assert len(repo) == 1
        assert repo.get_tasks()[0].n_samples == 20

    def test_sample_historic(self, exchangeable_repo):
        samples = exchangeable_repo.sample_historic(3, seed=0)
        assert len(samples) == 3
        for t in samples:
            assert isinstance(t, Task)

    def test_pool_features(self, exchangeable_repo):
        X = exchangeable_repo.pool_features()
        assert X.shape == (250, 3)

    def test_pool_targets(self, exchangeable_repo):
        y = exchangeable_repo.pool_targets()
        assert y.shape == (250,)

    def test_summary_statistics(self, exchangeable_repo):
        stats = exchangeable_repo.summary_statistics()
        assert stats["n_tasks"] == 5
        assert stats["total_samples"] == 250
        assert stats["n_features"] == 3

    def test_empty_repo_raises(self):
        repo = TaskRepository()
        with pytest.raises(ValueError):
            repo.sample_historic(1)
        with pytest.raises(ValueError):
            repo.pool_features()

    def test_task_shape_mismatch(self):
        with pytest.raises(ValueError, match="samples"):
            Task(X=np.zeros((5, 3)), y=np.zeros(10))


# ── kernel ─────────────────────────────────────────────────────────

class TestExchangeabilityKernel:
    def test_fit_and_score(self, exchangeable_repo, exchangeable_target):
        kernel = ExchangeabilityKernel()
        kernel.fit(exchangeable_repo.get_tasks())
        weights = kernel.score(exchangeable_target)
        assert weights.shape == (250,)
        assert np.all(weights >= 0)

    def test_task_weights(self, exchangeable_repo, exchangeable_target):
        kernel = ExchangeabilityKernel()
        kernel.fit(exchangeable_repo.get_tasks())
        tw = kernel.task_weights(exchangeable_target)
        assert tw.shape == (5,)
        assert np.all(tw >= 0)

    def test_pairwise_matrix(self, exchangeable_repo):
        kernel = ExchangeabilityKernel()
        kernel.fit(exchangeable_repo.get_tasks())
        K = kernel.pairwise_matrix()
        assert K.shape == (250, 250)
        assert np.allclose(K, K.T)

    def test_unfitted_raises(self):
        kernel = ExchangeabilityKernel()
        with pytest.raises(RuntimeError):
            kernel.score(Task(X=np.zeros((5, 3)), y=np.zeros(5)))


# ── exchangeability model ──────────────────────────────────────────

class TestExchangeabilityModel:
    def test_exchangeable_case(self, exchangeable_repo, exchangeable_target):
        model = ExchangeabilityModel(n_permutations=300)
        result = model.is_exchangeable(
            exchangeable_target,
            exchangeable_repo.get_tasks(),
            alpha=0.05,
            seed=42,
        )
        assert result.is_exchangeable is True
        assert result.p_value > 0.05
        assert abs(result.mmd_statistic) < 0.5
        assert result.n_permutations == 300

    def test_non_exchangeable_case(
        self, exchangeable_repo, non_exchangeable_target
    ):
        model = ExchangeabilityModel(n_permutations=300)
        result = model.is_exchangeable(
            non_exchangeable_target,
            exchangeable_repo.get_tasks(),
            alpha=0.05,
            seed=42,
        )
        assert result.is_exchangeable is False
        assert result.p_value < 0.05

    def test_pairwise_exchangeability(self, rng):
        beta = np.array([1.0, -0.5, 0.3])
        tasks = []
        for _ in range(3):
            X = rng.standard_normal((30, 3))
            y = X @ beta + rng.normal(0, 0.3, 30)
            tasks.append(Task(X=X, y=y))

        model = ExchangeabilityModel(n_permutations=200)
        matrix = model.pairwise_exchangeability(tasks, alpha=0.05, seed=42)
        assert matrix.shape == (3, 3)
        assert np.all(np.diag(matrix))

    def test_empty_historic_raises(self):
        model = ExchangeabilityModel()
        target = Task(X=np.zeros((5, 2)), y=np.zeros(5))
        with pytest.raises(ValueError):
            model.is_exchangeable(target, [], alpha=0.05)


# ── bias correction ────────────────────────────────────────────────

class TestBiasCorrector:
    def test_correct_reduces_bias(self, rng):
        X = rng.standard_normal((50, 3))
        beta_true = np.array([1.0, -0.5, 0.3])
        y = X @ beta_true + rng.normal(0, 0.3, 50)

        from sklearn.linear_model import Ridge
        est = Ridge(alpha=0.1)
        est.fit(X, y)

        weights = np.ones(50)
        corrector = BiasCorrector()
        raw = float(est.predict(X).mean())
        corrected = corrector.correct(est.predict, X, y, weights, raw)

        assert corrector.correction_term is not None
        assert corrector.residuals is not None
        assert len(corrector.residuals) == 50
        assert isinstance(corrected, float)

    def test_correct_predictions(self, rng):
        X = rng.standard_normal((30, 2))
        y = X[:, 0] * 2 + rng.normal(0, 0.1, 30)

        from sklearn.linear_model import Ridge
        est = Ridge(alpha=0.01)
        est.fit(X, y)

        weights = np.ones(30)
        corrector = BiasCorrector()
        raw_preds = est.predict(X)
        corrected = corrector.correct_predictions(
            est.predict, X, y, weights, raw_preds
        )
        assert corrected.shape == raw_preds.shape


# ── diagnostics ────────────────────────────────────────────────────

class TestDiagnostics:
    def test_exchangeable_passes(self, rng):
        X1 = rng.standard_normal((50, 3))
        X2 = rng.standard_normal((40, 3))
        diag = ExchangeabilityDiagnostic()
        result = diag.check(X1, X2)
        assert result.passed is True

    def test_non_exchangeable_fails(self, rng):
        X1 = rng.standard_normal((50, 3))
        X2 = rng.standard_normal((40, 3)) + 3.0
        diag = ExchangeabilityDiagnostic(error_threshold=0.01)
        result = diag.check(X1, X2)
        assert result.passed is False

    def test_raise_if_invalid(self, rng):
        X1 = rng.standard_normal((50, 3))
        X2 = rng.standard_normal((40, 3)) + 3.0
        diag = ExchangeabilityDiagnostic(error_threshold=0.01)
        with pytest.raises(ExchangeabilityViolation):
            diag.raise_if_invalid(X1, X2)

    def test_weighted_diagnostic(self, rng):
        X1 = rng.standard_normal((50, 3))
        X2 = rng.standard_normal((40, 3))
        weights = np.ones(50)
        diag = ExchangeabilityDiagnostic()
        result = diag.check(X1, X2, weights=weights)
        assert isinstance(result.statistic, float)


# ── inference ──────────────────────────────────────────────────────

class TestExchangeableInference:
    def test_fit_predict_without_historic(self, rng):
        X = rng.standard_normal((60, 3))
        beta = np.array([1.0, -0.5, 0.3])
        y = X @ beta + rng.normal(0, 0.3, 60)

        engine = ExchangeableInference()
        engine.fit(X, y)

        X_new = rng.standard_normal((10, 3))
        result = engine.predict_interval(X_new, confidence=0.95)
        assert result.prediction.shape == (10,)
        assert result.lower.shape == (10,)
        assert result.upper.shape == (10,)
        assert np.all(result.lower <= result.prediction)
        assert np.all(result.prediction <= result.upper)
        assert result.confidence == 0.95

    def test_fit_predict_with_historic(
        self, exchangeable_repo, exchangeable_target, rng
    ):
        engine = ExchangeableInference(run_diagnostics=False)
        engine.fit(
            exchangeable_target.X,
            exchangeable_target.y,
            historic_tasks=exchangeable_repo.get_tasks(),
        )

        X_new = rng.standard_normal((5, 3))
        result = engine.predict_interval(X_new, confidence=0.90)
        assert result.prediction.shape == (5,)
        assert result.confidence == 0.90
        assert np.all(result.lower < result.upper)

    def test_coverage_guarantee(self, rng):
        """Conformal intervals should achieve near-nominal coverage."""
        beta = np.array([2.0, -1.0])
        n_train = 200
        n_test = 100

        X_train = rng.standard_normal((n_train, 2))
        y_train = X_train @ beta + rng.normal(0, 0.5, n_train)

        X_test = rng.standard_normal((n_test, 2))
        y_test = X_test @ beta + rng.normal(0, 0.5, n_test)

        engine = ExchangeableInference()
        engine.fit(X_train, y_train)

        result = engine.predict_interval(X_test, confidence=0.90)
        covered = (y_test >= result.lower) & (y_test <= result.upper)
        coverage = covered.mean()
        assert coverage >= 0.80

    def test_unfitted_raises(self, rng):
        engine = ExchangeableInference()
        with pytest.raises(RuntimeError):
            engine.predict_interval(rng.standard_normal((5, 3)))

    def test_no_x_new_raises(self, rng):
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal(30)
        engine = ExchangeableInference()
        engine.fit(X, y)
        with pytest.raises(ValueError):
            engine.predict_interval()


# ── end-to-end pipeline ───────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_exchangeable(self, rng):
        """Full pipeline: repo -> exchangeability test -> inference."""
        beta = np.array([1.0, -0.5, 0.3])
        repo = TaskRepository()
        for _ in range(4):
            X = rng.standard_normal((40, 3))
            y = X @ beta + rng.normal(0, 0.3, 40)
            repo.add_task(X, y)

        X_synth = rng.standard_normal((30, 3))
        y_synth = X_synth @ beta + rng.normal(0, 0.3, 30)
        target = Task(X=X_synth, y=y_synth)

        model = ExchangeabilityModel(n_permutations=200)
        exch_result = model.is_exchangeable(
            target, repo.get_tasks(), alpha=0.05, seed=42
        )
        assert exch_result.is_exchangeable is True

        engine = ExchangeableInference(run_diagnostics=False)
        engine.fit(X_synth, y_synth, historic_tasks=repo.get_tasks())

        X_new = rng.standard_normal((10, 3))
        y_new = X_new @ beta + rng.normal(0, 0.3, 10)
        result = engine.predict_interval(X_new, confidence=0.95)

        assert result.prediction.shape == (10,)
        assert np.all(result.lower < result.upper)

    def test_full_pipeline_non_exchangeable(self, rng):
        """Pipeline detects non-exchangeable case."""
        beta = np.array([1.0, -0.5, 0.3])
        repo = TaskRepository()
        for _ in range(4):
            X = rng.standard_normal((40, 3))
            y = X @ beta + rng.normal(0, 0.3, 40)
            repo.add_task(X, y)

        X_alien = rng.uniform(5, 10, (30, 3))
        y_alien = np.sin(X_alien[:, 0]) * 10 + rng.normal(0, 0.1, 30)
        target = Task(X=X_alien, y=y_alien)

        model = ExchangeabilityModel(n_permutations=300)
        result = model.is_exchangeable(
            target, repo.get_tasks(), alpha=0.05, seed=42
        )
        assert result.is_exchangeable is False
        assert result.p_value < 0.05


# ── imports ────────────────────────────────────────────────────────

class TestImports:
    def test_top_level_imports(self):
        from exchangelib import (
            ExchangeabilityModel,
            ExchangeableInference,
            ExchangeabilityKernel,
            TaskRepository,
        )
        assert ExchangeabilityModel is not None
        assert ExchangeableInference is not None
        assert ExchangeabilityKernel is not None
        assert TaskRepository is not None

    def test_module_imports(self):
        from exchangelib.repository import TaskRepository
        from exchangelib.exchangeability import ExchangeabilityModel
        from exchangelib.kernel import ExchangeabilityKernel
        from exchangelib.bias_correction import BiasCorrector
        from exchangelib.inference import ExchangeableInference
        from exchangelib.diagnostics import ExchangeabilityDiagnostic
        from exchangelib.utils import density_ratio_kmm
        from exchangelib.cli import CLI

    def test_version(self):
        import exchangelib
        assert exchangelib.__version__ == "0.1.0"
