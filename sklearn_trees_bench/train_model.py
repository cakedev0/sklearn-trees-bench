#!/usr/bin/env python
"""Train tree/forest models on synthetic data and measure performance."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Model registry
MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
}


def generate_synthetic_data(task="classification", n_samples=1000, n_features=20, random_state=42):
    """Generate synthetic data for training.
    
    Parameters
    ----------
    task : str, default="classification"
        Task type: "classification" or "regression"
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=20
        Number of features
    random_state : int, default=42
        Random state for reproducibility
    
    Returns
    -------
    X, y : tuple
        Feature matrix and target vector
    """
    if task == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            random_state=random_state,
        )
    elif task == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return X, y


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
    predictions = model.predict(X)
    predict_time = time.perf_counter() - start_time
    
    return {
        "train_time": train_time,
        "predict_time": predict_time,
    }


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train tree/forest models on synthetic data"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
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
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples in synthetic dataset (default: 1000)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of features in synthetic dataset (default: 20)",
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
        help="Output file to save results (JSON format)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Parse model parameters
    try:
        model_params = json.loads(args.model_params)
    except json.JSONDecodeError as e:
        parser.error(f"Invalid JSON for --model-params: {e}")
    
    # Get model class
    model_class = MODELS[args.model]
    
    # Determine task type from model name
    task = "classification" if "Classifier" in args.model else "regression"
    
    # Generate synthetic data
    print(f"Generating synthetic {task} data...")
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_features: {args.n_features}")
    X, y = generate_synthetic_data(
        task=task,
        n_samples=args.n_samples,
        n_features=args.n_features,
        random_state=args.random_state,
    )
    
    # Train model n_repeats times
    print(f"\nTraining {args.model} {args.n_repeats} times...")
    print(f"  Model parameters: {model_params}")
    
    results = []
    for i in range(args.n_repeats):
        print(f"  Iteration {i + 1}/{args.n_repeats}...", end=" ", flush=True)
        timing = train_and_measure(model_class, X, y, random_state=args.random_state + i, **model_params)
        results.append(timing)
        print(f"train={timing['train_time']:.4f}s, predict={timing['predict_time']:.4f}s")
    
    # Compute statistics
    train_times = [r["train_time"] for r in results]
    predict_times = [r["predict_time"] for r in results]
    
    summary = {
        "model": args.model,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_repeats": args.n_repeats,
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
    print(f"Train time: {summary['train_time_mean']:.4f} ± {summary['train_time_std']:.4f}s")
    print(f"Predict time: {summary['predict_time_mean']:.4f} ± {summary['predict_time_std']:.4f}s")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
