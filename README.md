# sklearn-trees-bench

Benchmark suite for scikit-learn decision trees, random forests, and other tree based models.

## Features

- 🌲 **Train Models**: Benchmark decision trees and random forests on synthetic data
- 🔄 **Orchestration**: Run parameter grids across different scikit-learn git branches
- 📊 **Visualization**: Analyze results using pandas and matplotlib in Jupyter notebooks

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync

# Or install the package in development mode
uv pip install -e .
```

## Quick Start

### 1. Sanity Check

Verify the scikit-learn submodule environment can run a single benchmark:

```bash
./scikit-learn/sklearn-env/bin/python sklearn_trees_bench/train_model.py \
  --model DecisionTreeClassifier --n-repeats 1
```

### 2. Orchestrate Benchmarks

Run benchmarks across multiple configurations and scikit-learn branches:

```bash
# First, add scikit-learn as a submodule (one-time setup)
git submodule add https://github.com/scikit-learn/scikit-learn.git scikit-learn
git submodule update --init --recursive

# Build scikit-learn in scikit-learn/sklearn-env (see SETUP.md), then run:
uv run orchestrate --config example_config.json
```

Example configuration file (`example_config.json`):

```json
{
  "models": {
    "DecisionTreeClassifier": {
      "max_depth": [5, 10, 20],
      "min_samples_split": [2, 10]
    },
    "RandomForestClassifier": {
      "n_estimators": [10, 50, 100],
      "max_depth": [5, 10]
    }
  },
  "branches": ["main", "1.3.X"],
  "n_samples": [1000, 5000],
  "n_features": [20, 50],
  "n_repeats": 3
}
```

### 3. Visualize Results

Open the Jupyter notebook to analyze and visualize results:

```bash
uv run jupyter notebook visualize_results.ipynb
```

## Usage Examples

### Available Models

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`
- `RandomForestClassifier`
- `RandomForestRegressor`

### Command Line Options

#### train-model

```
usage: train-model [-h] --model {DecisionTreeClassifier,DecisionTreeRegressor,RandomForestClassifier,RandomForestRegressor}
                   [--n-repeats N_REPEATS] [--n-samples N_SAMPLES] [--n-features N_FEATURES]
                   [--cardinality {high,medium,low,binary}] [--target-fit-s TARGET_FIT_S]
                   [--model-params MODEL_PARAMS] [--output OUTPUT]

options:
  --model               Model to train (required)
  --n-repeats           Number of times to repeat training (default: 3)
  --n-samples           Number of samples in synthetic dataset (default: 1000)
  --n-features          Number of features in synthetic dataset (default: 20)
  --cardinality         Feature cardinality (default: high)
  --target-fit-s        Auto-scale n_samples until fit time reaches this target (seconds)
  --model-params        Model parameters as JSON string (default: '{}')
  --output              Output file to append results (JSONL format)
```

#### orchestrate

```
usage: orchestrate [-h] --config CONFIG

options:
  --config              JSON config file with benchmark parameters (required)
```

## Project Structure

```
sklearn-trees-bench/
├── sklearn_trees_bench/
│   ├── __init__.py
│   ├── train_model.py      # Single model training script
│   └── orchestrate.py      # Orchestration script
├── example_config.json     # Example configuration
├── visualize_results.ipynb # Jupyter notebook for visualization
├── pyproject.toml          # Project configuration
└── README.md
```

## License

This project is open source and available under the MIT License.
