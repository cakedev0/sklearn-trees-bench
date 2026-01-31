#!/usr/bin/env python
"""Orchestrate benchmarks across different scikit-learn branches and parameter grids."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product, chain
import time

import numpy as np


def checkout_sklearn_branch(sklearn_path, branch):
    try:
        subprocess.run(
            ["git", "checkout", branch],
            cwd=sklearn_path,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Checked out branch: {branch}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error checking out branch {branch}: {e.stderr}")
        return False


def warmup(s):
    t = time.perf_counter()
    X = np.random.rand(1000, 1000)
    Y = np.empty_like(X)
    while time.perf_counter() - t < s:
        np.dot(X, X.T, out=Y)


def run_benchmark(
    python_executable,
    env,
    model,
    model_params,
    data_params,
    n_repeats,
    output_file,
):
    # Build command
    cmd = [
        python_executable, "-m", "sklearn_trees_bench.train_model",
        "--model", model,
        "--n-repeats", str(n_repeats),
        "--data-params", json.dumps(data_params),
        "--model-params", json.dumps(model_params),
        "--output", str(output_file),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def normalize_grid(grid: dict | None) -> dict:
    if not grid:
        return {}
    normalized = {}
    for key, value in grid.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def generate_combinations(grid: dict | None):
    grid = normalize_grid(grid)
    if not grid:
        return [{}]

    param_names = list(grid.keys())
    param_values = list(grid.values())
    for values in product(*param_values):
        yield dict(zip(param_names, values))


def generate_scenarios(config: dict):
    dataset_grids = config.get("datasets", [None])
    dataset_combinations = list(chain(*[
        generate_combinations(dataset_grid)
        for dataset_grid in dataset_grids
    ]))

    for model_name, param_grid in config["models"].items():
        param_combinations = generate_combinations(param_grid)
        yield from product([model_name], param_combinations, dataset_combinations)


def update_scenarios_with_n_samples(scenarios, outputs):
    for (_, _, data_params), output in zip(scenarios, outputs):
        if "target_fit_s" not in data_params:
            continue
        data_params.pop("target_fit_s")
        data_params["n_samples"] = output['data_params']['n_samples']


def main():
    """Main entry point for orchestration script."""
    parser = argparse.ArgumentParser(
        description="Orchestrate benchmarks across different scikit-learn branches"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON config file with benchmark parameters",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: dict = json.load(f)

    # Setup paths
    sklearn_path = Path('scikit-learn')
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate config
    required_keys = ["models", "branches", "datasets"]
    for key in required_keys:
        if key not in config:
            parser.error(f"Config file missing required key: {key}")

    n_repeats = config.get("n_repeats", 1)

    # Resolve runner environment
    python_executable = str(sklearn_path / "sklearn-env" / "bin" / "python")
    if not Path(python_executable).exists():
        print("Error: scikit-learn environment not found.")
        print(f"Expected interpreter at: {python_executable}")
        print("Build scikit-learn in scikit-learn/sklearn-env first.")
        return 1
    repo_root = Path(__file__).resolve().parents[1]
    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{run_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    )

    # Run benchmarks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    s = config.get('warmup_s', 0)
    if s:
        print(f'Warming up for {s}s...')
        warmup(s)
        print('Warmup done')

    print("=" * 80)
    print("BENCHMARK ORCHESTRATION")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Models: {', '.join(config['models'].keys())}")
    print(f"Branches: {', '.join(config['branches'])}")
    print(f"n_repeats: {n_repeats}")
    print("=" * 80)

    scenarios = list(generate_scenarios(config))

    for branch_index, branch in enumerate(config["branches"]):
        print(f"\nProcessing branch: {branch}")

        # Checkout branch
        if not checkout_sklearn_branch(sklearn_path, branch):
            print(f"  Skipping branch {branch} due to checkout error")
            continue

        branch = branch.replace("/", "_")
        output_file = output_dir / f"{timestamp}_{branch}.jsonl"

        print(f'  Evaluating {len(scenarios)} scenarios:')

        successful_indices = []
        # Run benchmarks for this branch
        for scenario_index, (model_name, params, dataset_params) in enumerate(scenarios):
            dataset_label = dataset_params if dataset_params else "defaults"
            print(f"    params={params}, dataset={dataset_label}")

            # Run benchmark
            run_benchmark(
                python_executable,
                run_env,
                model_name,
                params,
                dataset_params,
                n_repeats,
                output_file,
            )

        if branch_index == 0:
            with output_file.open("r") as f:
                outputs = [json.loads(line) for line in f if line.strip()]
            assert len(scenarios) == len(outputs)
            update_scenarios_with_n_samples(scenarios, outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
