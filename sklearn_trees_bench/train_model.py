#!/usr/bin/env python
"""Train tree/forest models on synthetic data and measure performance."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Model registry
MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
}


def generate_synthetic_data(
    task="classification",
    n_samples=1000,
    n_features=20,
    cardinality="high",
):
    """Generate synthetic data for training.

    Parameters
    ----------
    task : str, default="classification"
        Task type: "classification" or "regression"
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=20
        Number of features
    cardinality : str, default="high"
        Feature cardinality: "high" (default), "medium", "low", or "binary"

    Returns
    -------
    X, y : tuple
        Feature matrix and target vector
    """
    rng = np.random.default_rng()

    if cardinality == "high":
        X = rng.normal(size=(n_samples, n_features))
    elif cardinality == "medium":
        X = rng.geometric(0.02, size=(n_samples, n_features))
    elif cardinality == "low":
        X = rng.geometric(0.15, size=(n_samples, n_features))
    elif cardinality == "binary":
        X = rng.integers(0, 2, size=(n_samples, n_features))
    else:
        raise ValueError(f"Unknown cardinality: {cardinality}")

    X = X.astype(np.float32, copy=False)
    coef = rng.normal(size=n_features)
    y = X @ coef
    y += rng.permutation(y)

    if task == "classification":
        y = (y > np.median(y)).astype(int)
    elif task != "regression":
        raise ValueError(f"Unknown task: {task}")

    return X, y


def find_n_samples_for_target(
    *,
    model_class,
    model_params: dict,
    task: str,
    n_features: int,
    cardinality: str,
    target_fit_s: float,
    start_n_samples: int,
) -> int:
    n_samples = max(10, start_n_samples)

    while True:
        X, y = generate_synthetic_data(
            task=task,
            n_samples=n_samples,
            n_features=n_features,
            cardinality=cardinality,
        )
        model = model_class(**model_params)
        start_time = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - start_time
        if elapsed >= target_fit_s:
            return n_samples

        if str(n_samples)[0] == "2":
            n_samples = n_samples // 2 * 5
        else:
            n_samples *= 2

        if n_samples > 10_000_000:
            raise RuntimeError(
                "Failed to reach target fit time; try a lower target_fit_s."
            )


def train_and_measure(model_class, X, y, **model_params):
    """Train a model and measure training time and prediction time.

    Parameters
    ----------
    model_class : class
        Model class to instantiate
    X : array-like
        Training features
    y : array-like
        Training targets
    **model_params : dict
        Parameters to pass to model constructor

    Returns
    -------
    dict
        Dictionary containing timing results and model info
    """
    # Create model
    model = model_class(**model_params)

    # Measure training time
    start_time = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter() - start_time

    # Measure prediction time
    start_time = time.perf_counter()
    _ = model.predict(X)
    predict_time = time.perf_counter() - start_time

    return {
        "train_time": train_time,
        "predict_time": predict_time,
    }


def main():
    """Main entry point for the training script."""
    import sklearn
    assert "sklearn-env" in sklearn.__path__[0]

    parser = argparse.ArgumentParser(
        description="Train tree/forest models on synthetic data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DecisionTreeRegressor",
        choices=list(MODELS.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of times to repeat the training (default: 3)",
    )
    parser.add_argument(
        "--data-params",
        type=str,
        default="{}",
        help=(
            "Dataset parameters as JSON string (keys: n_samples, n_features, "
            "cardinality, target_fit_s)"
        ),
    )
    parser.add_argument(
        "--model-params",
        type=str,
        default="{}",
        help="Model parameters as JSON string (default: '{}')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to append results (JSONL format)",
    )

    args = parser.parse_args()

    # Parse model parameters
    model_params = json.loads(args.model_params)
    dataset_params = json.loads(args.data_params)

    data_params = {
        "n_samples": 1000,
        "n_features": 20,
        "cardinality": "high",
        "target_fit_s": None,
        **dataset_params,
    }

    # Get model class
    model_class = MODELS[args.model]

    # Determine task type from model name
    task = "classification" if "Classifier" in args.model else "regression"

    n_samples = data_params["n_samples"]
    if data_params["target_fit_s"] is not None:
        print("Finding n_samples to reach target fit time...")
        n_samples = find_n_samples_for_target(
            model_class=model_class,
            model_params=model_params,
            task=task,
            n_features=data_params["n_features"],
            cardinality=data_params["cardinality"],
            target_fit_s=data_params["target_fit_s"],
            start_n_samples=data_params["n_samples"],
        )
        print(f"  target_fit_s: {data_params['target_fit_s']}")
        print(f"  resolved n_samples: {n_samples}")


    X, y = generate_synthetic_data(
        task=task,
        n_samples=n_samples,
        n_features=data_params["n_features"],
        cardinality=data_params["cardinality"],
    )
    results = []
    for _ in range(args.n_repeats):
        timing = train_and_measure(
            model_class, X, y, **model_params
        )
        results.append(timing)
        print(f"train={timing['train_time']:.4f}s")

    # Compute statistics
    train_times = [r["train_time"] for r in results]
    predict_times = [r["predict_time"] for r in results]

    summary = {
        "model": args.model,
        "n_samples": n_samples,
        "n_features": data_params["n_features"],
        "cardinality": data_params["cardinality"],
        "n_repeats": args.n_repeats,
        "target_fit_s": data_params["target_fit_s"],
        "model_params": model_params,
        "train_time_mean": float(np.mean(train_times)),
        "train_time_std": float(np.std(train_times)),
        "train_time_min": float(np.min(train_times)),
        "train_time_max": float(np.max(train_times)),
        "predict_time_mean": float(np.mean(predict_times)),
        "predict_time_std": float(np.std(predict_times)),
        "predict_time_min": float(np.min(predict_times)),
        "predict_time_max": float(np.max(predict_times)),
        "all_results": results,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(
        f"Train time: {summary['train_time_mean']:.4f} ± {summary['train_time_std']:.4f}s"
    )
    print(
        f"Predict time: {summary['predict_time_mean']:.4f} ± {summary['predict_time_std']:.4f}s"
    )

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix != ".jsonl":
            output_path = output_path.with_suffix(".jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(summary))
            f.write("\n")
        print(f"\nResults appended to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
