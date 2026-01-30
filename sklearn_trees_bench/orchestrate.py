#!/usr/bin/env python
"""Orchestrate benchmarks across different scikit-learn branches and parameter grids."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product


def setup_sklearn_submodule(sklearn_path):
    """Initialize and update the scikit-learn submodule.

    Parameters
    ----------
    sklearn_path : Path
        Path to the scikit-learn submodule

    Returns
    -------
    bool
        True if setup successful, False otherwise
    """
    if not sklearn_path.exists():
        print(f"Error: scikit-learn submodule not found at {sklearn_path}")
        print("Please initialize the submodule first:")
        print(
            "  git submodule add https://github.com/scikit-learn/scikit-learn.git scikit-learn"
        )
        print("  git submodule update --init --recursive")
        return False
    return True


def checkout_sklearn_branch(sklearn_path, branch):
    """Checkout a specific branch of scikit-learn.

    Parameters
    ----------
    sklearn_path : Path
        Path to the scikit-learn submodule
    branch : str
        Branch name or commit hash to checkout

    Returns
    -------
    bool
        True if checkout successful, False otherwise
    """
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


def get_sklearn_version(python_executable):
    """Get the currently installed scikit-learn version.

    Returns
    -------
    str
        Version string
    """
    try:
        result = subprocess.run(
            [python_executable, "-c", "import sklearn; sklearn.show_versions()"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_benchmark(
    python_executable,
    env,
    model,
    model_params,
    n_samples,
    n_features,
    n_repeats,
    output_file,
):
    # Build command
    cmd = [
        python_executable,
        "-m",
        "sklearn_trees_bench.train_model",
        "--model",
        model,
        "--n-samples",
        str(n_samples),
        "--n-features",
        str(n_features),
        "--n-repeats",
        str(n_repeats),
        "--model-params",
        json.dumps(model_params),
        "--output",
        str(output_file),
    ]

    try:
        start_offset = output_file.stat().st_size if output_file.exists() else 0
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

        # Load and return the last JSONL entry written by this run.
        last_line = None
        with open(output_file, "r") as f:
            f.seek(start_offset)
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line is None:
            raise ValueError("No results written to output file.")
        return json.loads(last_line)
    except subprocess.CalledProcessError as e:
        print(f"    Error running benchmark: {e.stderr}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def generate_combinations(grid: dict | None):
    if not grid:
        return [{}]

    param_names = list(grid.keys())
    param_values = list(grid.values())
    for values in product(*param_values):
        yield dict(zip(param_names, values))


def generate_scenarios(config: dict):
    n_samples_list = config.get("n_samples", [1000])
    n_features_list = config.get("n_features", [20])

    for model_name, param_grid in config["models"].items():
        print(f"\n  Model: {model_name}")

        param_combinations = generate_combinations(param_grid)

        yield from product([model_name], param_combinations, n_samples_list, n_features_list)


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
    parser.add_argument(
        "--sklearn-path",
        type=str,
        default="scikit-learn",
        help="Path to scikit-learn submodule (default: 'scikit-learn')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: 'results')",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Setup paths
    sklearn_path = Path(args.sklearn_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate config
    required_keys = ["models", "branches"]
    for key in required_keys:
        if key not in config:
            parser.error(f"Config file missing required key: {key}")

    # Setup defaults
    n_samples_list = config.get("n_samples", [1000])
    n_features_list = config.get("n_features", [20])
    n_repeats = config.get("n_repeats", 3)

    # Check sklearn submodule
    if not setup_sklearn_submodule(sklearn_path):
        return 1

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
    all_results = []
    output_file = output_dir / f"{timestamp}_results.jsonl"

    print("=" * 80)
    print("BENCHMARK ORCHESTRATION")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Models: {', '.join(config['models'].keys())}")
    print(f"Branches: {', '.join(config['branches'])}")
    print(f"n_samples: {n_samples_list}")
    print(f"n_features: {n_features_list}")
    print(f"n_repeats: {n_repeats}")
    print("=" * 80)

    for branch in config["branches"]:
        print(f"\nProcessing branch: {branch}")

        # Checkout branch
        if not checkout_sklearn_branch(sklearn_path, branch):
            print(f"  Skipping branch {branch} due to checkout error")
            continue

        sklearn_version = get_sklearn_version(python_executable)

        # Run benchmarks for this branch
        for model_name, params, n_samples, n_features in generate_scenarios(config):
            print(f"    n_samples={n_samples}, n_features={n_features}, params={params}")

            # Run benchmark
            result = run_benchmark(
                python_executable,
                run_env,
                model_name,
                params,
                n_samples,
                n_features,
                n_repeats,
                output_file,
            )

            if result:
                result["branch"] = branch
                result["sklearn_version"] = sklearn_version
                result["timestamp"] = timestamp
                all_results.append(result)
                print(f"    ✓ train={result['train_time_mean']:.4f}s")
            else:
                print(f"    ✗ Failed")

    # Save combined results
    summary_file = output_dir / f"{timestamp}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("ORCHESTRATION COMPLETE")
    print("=" * 80)
    print(f"Total benchmarks: {len(all_results)}")
    print(f"Summary saved to: {summary_file}")
    print(f"JSONL results appended to: {output_file}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
