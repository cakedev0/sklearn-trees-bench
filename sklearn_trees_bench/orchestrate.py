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

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"    Error running benchmark: {e.stderr}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

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
        print(f"\n  Model: {model_name}")

        param_combinations = generate_combinations(param_grid)
        yield from product([model_name], param_combinations, dataset_combinations)


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

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Setup paths
    sklearn_path = Path('scikit-learn')
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate config
    required_keys = ["models", "branches", "datasets"]
    for key in required_keys:
        if key not in config:
            parser.error(f"Config file missing required key: {key}")

    n_repeats = config.get("n_repeats", 3)

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

    print("=" * 80)
    print("BENCHMARK ORCHESTRATION")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Models: {', '.join(config['models'].keys())}")
    print(f"Branches: {', '.join(config['branches'])}")
    print(f"n_repeats: {n_repeats}")
    print("=" * 80)

    for branch in config["branches"]:
        print(f"\nProcessing branch: {branch}")

        # Checkout branch
        if not checkout_sklearn_branch(sklearn_path, branch):
            print(f"  Skipping branch {branch} due to checkout error")
            continue

        branch = branch.replace("/", "_")
        output_file = output_dir / f"{timestamp}_{branch}.jsonl"

        # Run benchmarks for this branch
        for model_name, params, dataset_params in generate_scenarios(config):
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
