"""Command-line interface for exchangelib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


class CLI:
    """Simple CLI for running exchangeability inference from the command line."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="exchangelib",
            description="Task-exchangeability inference with synthetic data",
        )
        sub = parser.add_subparsers(dest="command")

        run_parser = sub.add_parser("run", help="Run inference pipeline")
        run_parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Directory containing CSV files (one per historic task)",
        )
        run_parser.add_argument(
            "--synthetic",
            type=str,
            required=True,
            help="Path to synthetic data CSV (features + target column)",
        )
        run_parser.add_argument(
            "--confidence",
            type=float,
            default=0.95,
            help="Confidence level for prediction intervals (default: 0.95)",
        )
        run_parser.add_argument(
            "--target-col",
            type=str,
            default="y",
            help="Name of the target column in CSVs (default: y)",
        )
        run_parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducibility",
        )

        test_parser = sub.add_parser("test", help="Run exchangeability test")
        test_parser.add_argument(
            "--data-dir",
            type=str,
            required=True,
            help="Directory with historic task CSVs",
        )
        test_parser.add_argument(
            "--synthetic",
            type=str,
            required=True,
            help="Path to synthetic data CSV",
        )
        test_parser.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="Significance level (default: 0.05)",
        )
        test_parser.add_argument(
            "--target-col",
            type=str,
            default="y",
            help="Name of the target column",
        )

        demo_parser = sub.add_parser("demo", help="Run demo with synthetic data")
        demo_parser.add_argument(
            "--n-tasks",
            type=int,
            default=5,
            help="Number of historic tasks (default: 5)",
        )
        demo_parser.add_argument(
            "--n-samples",
            type=int,
            default=50,
            help="Samples per task (default: 50)",
        )
        demo_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed",
        )

        return parser

    @staticmethod
    def main(argv=None) -> int:
        parser = CLI.build_parser()
        args = parser.parse_args(argv)

        if args.command is None:
            parser.print_help()
            return 0

        if args.command == "demo":
            return CLI._run_demo(args)
        elif args.command == "run":
            return CLI._run_inference(args)
        elif args.command == "test":
            return CLI._run_test(args)

        return 0

    @staticmethod
    def _run_demo(args) -> int:
        from exchangelib.exchangeability import ExchangeabilityModel
        from exchangelib.inference import ExchangeableInference
        from exchangelib.repository import Task, TaskRepository

        rng = np.random.default_rng(args.seed)
        repo = TaskRepository()

        beta_true = rng.standard_normal(3)
        for _ in range(args.n_tasks):
            X = rng.standard_normal((args.n_samples, 3))
            y = X @ beta_true + rng.normal(0, 0.5, args.n_samples)
            repo.add_task(X, y)

        X_synth = rng.standard_normal((args.n_samples, 3))
        y_synth = X_synth @ beta_true + rng.normal(0, 0.5, args.n_samples)

        model = ExchangeabilityModel(n_permutations=200)
        target = Task(X=X_synth, y=y_synth)
        result = model.is_exchangeable(
            target, repo.get_tasks(), alpha=0.05, seed=args.seed
        )
        print(f"Exchangeability test: p={result.p_value:.4f}, "
              f"exchangeable={result.is_exchangeable}")

        engine = ExchangeableInference()
        engine.fit(X_synth, y_synth, historic_tasks=repo.get_tasks())
        X_new = rng.standard_normal((5, 3))
        interval = engine.predict_interval(X_new, confidence=0.95)
        print(f"\nPrediction intervals (95% confidence):")
        for i in range(len(X_new)):
            print(f"  x[{i}]: {interval.prediction[i]:.3f} "
                  f"[{interval.lower[i]:.3f}, {interval.upper[i]:.3f}]")
        print(f"\nBias correction: {interval.bias_correction:.4f}")
        print(f"Conformal quantile: {interval.conformal_quantile:.4f}")

        return 0

    @staticmethod
    def _run_inference(args) -> int:
        import pandas as pd
        from exchangelib.inference import ExchangeableInference
        from exchangelib.repository import TaskRepository

        repo = TaskRepository()
        data_dir = Path(args.data_dir)
        for csv_path in sorted(data_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            y = df[args.target_col].values
            X = df.drop(columns=[args.target_col]).values
            repo.add_task(X, y)

        if len(repo) == 0:
            print(f"No CSV files found in {data_dir}", file=sys.stderr)
            return 1

        synth_df = pd.read_csv(args.synthetic)
        y_synth = synth_df[args.target_col].values
        X_synth = synth_df.drop(columns=[args.target_col]).values

        engine = ExchangeableInference()
        engine.fit(X_synth, y_synth, historic_tasks=repo.get_tasks())
        result = engine.predict_interval(X_synth, confidence=args.confidence)

        print(f"Inference complete ({len(repo)} historic tasks)")
        print(f"Bias correction: {result.bias_correction:.4f}")
        print(f"Conformal quantile ({args.confidence*100:.0f}%): "
              f"{result.conformal_quantile:.4f}")
        coverage = np.mean(
            (y_synth >= result.lower) & (y_synth <= result.upper)
        )
        print(f"Empirical coverage: {coverage:.4f}")

        return 0

    @staticmethod
    def _run_test(args) -> int:
        import pandas as pd
        from exchangelib.exchangeability import ExchangeabilityModel
        from exchangelib.repository import Task, TaskRepository

        repo = TaskRepository()
        data_dir = Path(args.data_dir)
        for csv_path in sorted(data_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            y = df[args.target_col].values
            X = df.drop(columns=[args.target_col]).values
            repo.add_task(X, y)

        synth_df = pd.read_csv(args.synthetic)
        y_synth = synth_df[args.target_col].values
        X_synth = synth_df.drop(columns=[args.target_col]).values

        model = ExchangeabilityModel(n_permutations=500)
        target = Task(X=X_synth, y=y_synth)
        result = model.is_exchangeable(
            target, repo.get_tasks(), alpha=args.alpha
        )

        print(f"MMD statistic: {result.mmd_statistic:.6f}")
        print(f"Threshold ({(1-args.alpha)*100:.0f}%): {result.threshold:.6f}")
        print(f"p-value: {result.p_value:.4f}")
        print(f"Exchangeable: {result.is_exchangeable}")

        return 0


def main():
    sys.exit(CLI.main())


if __name__ == "__main__":
    main()
