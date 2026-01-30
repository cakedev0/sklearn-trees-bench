# Setup Instructions

## Setting up scikit-learn submodule

The orchestration script can benchmark different branches of scikit-learn. To enable this functionality, you need to add scikit-learn as a git submodule:

```bash
# Add scikit-learn as a submodule
git submodule add https://github.com/scikit-learn/scikit-learn.git scikit-learn
git submodule update --init --recursive

# Commit the submodule
git add .gitmodules scikit-learn
git commit -m "Add scikit-learn submodule"
```

## Running benchmarks

These benchmarks assume scikit-learn is installed in `scikit-learn/sklearn-env`.

### With submodule (different branches)

To test different scikit-learn branches:

```bash
# Make sure you have the submodule set up (see above)
uv run orchestrate --config example_config.json
```

The orchestration script will:
1. Checkout each branch specified in the config
2. Use the scikit-learn environment in `scikit-learn/sklearn-env` to rebuild on import
3. Run all benchmarks for that branch
4. Move to the next branch

## Configuration file format

Example `config.json`:

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

### Configuration options

- **models**: Dictionary where keys are model names and values are parameter grids
  - Each parameter can have a list of values to test
  - Empty dict `{}` means use default parameters
- **branches**: List of git branch names or commit hashes to test
- **n_samples**: List of sample counts for synthetic datasets
- **n_features**: List of feature counts for synthetic datasets
- **n_repeats**: Number of times to repeat each benchmark (for statistical stability)

## Example workflows

### Quick test with submodule environment

```bash
# Create a minimal config
cat > quick_test.json << 'ENDCONFIG'
{
  "models": {
    "DecisionTreeClassifier": {"max_depth": [5]}
  },
  "branches": ["current"],
  "n_samples": [1000],
  "n_features": [20],
  "n_repeats": 3
}
ENDCONFIG

# Run with the submodule environment
uv run orchestrate --config quick_test.json
```

### Compare multiple scikit-learn versions

```bash
# Set up submodule first
git submodule add https://github.com/scikit-learn/scikit-learn.git scikit-learn
git submodule update --init --recursive

# Use example config with multiple branches
uv run orchestrate --config example_config.json
```

### Analyze results

```bash
# Open Jupyter notebook
uv run jupyter notebook visualize_results.ipynb
```
