# Usage Examples

This document provides practical examples of using the sklearn-trees-bench toolkit.

## Basic Usage

### Example 1: Quick benchmark of a single model

```bash
uv run train-model \
  --model DecisionTreeClassifier \
  --n-samples 1000 \
  --n-features 20 \
  --n-repeats 5
```

**Output:**
```
Generating synthetic classification data...
  n_samples: 1000
  n_features: 20

Training DecisionTreeClassifier 5 times...
  Model parameters: {}
  Iteration 1/5... train=0.0050s, predict=0.0002s
  ...

SUMMARY
Model: DecisionTreeClassifier
Train time: 0.0048 ± 0.0002s
Predict time: 0.0002 ± 0.0000s
```

### Example 2: Compare tree depths

```bash
# Benchmark shallow tree
uv run train-model \
  --model DecisionTreeClassifier \
  --model-params '{"max_depth": 3}' \
  --output results/shallow_tree.json

# Benchmark deep tree
uv run train-model \
  --model DecisionTreeClassifier \
  --model-params '{"max_depth": 20}' \
  --output results/deep_tree.json
```

### Example 3: Random Forest with custom parameters

```bash
uv run train-model \
  --model RandomForestClassifier \
  --n-samples 5000 \
  --n-features 50 \
  --n-repeats 3 \
  --model-params '{"n_estimators": 100, "max_depth": 10, "min_samples_split": 10}' \
  --output results/rf_custom.json
```

### Example 4: Regression task

```bash
uv run train-model \
  --model RandomForestRegressor \
  --n-samples 2000 \
  --model-params '{"n_estimators": 50, "max_features": "sqrt"}' \
  --output results/rf_regression.json
```

## Orchestration Examples

### Example 5: Parameter grid search (current environment)

Create a config file `param_grid.json`:
```json
{
  "models": {
    "DecisionTreeClassifier": {
      "max_depth": [3, 5, 10, 15, 20],
      "min_samples_split": [2, 5, 10]
    }
  },
  "branches": ["current"],
  "n_samples": [1000, 5000],
  "n_features": [20, 50],
  "n_repeats": 5
}
```

Run the benchmarks:
```bash
uv run orchestrate \
  --config param_grid.json \
  --output-dir results/param_grid \
  --skip-install
```

This will run: 5 depths × 3 min_samples_split × 2 n_samples × 2 n_features = 60 benchmarks

### Example 6: Compare RandomForest vs DecisionTree

Create `rf_vs_dt.json`:
```json
{
  "models": {
    "DecisionTreeClassifier": {
      "max_depth": [5, 10]
    },
    "RandomForestClassifier": {
      "n_estimators": [10, 50, 100],
      "max_depth": [5, 10]
    }
  },
  "branches": ["current"],
  "n_samples": [1000, 5000, 10000],
  "n_features": [20],
  "n_repeats": 3
}
```

```bash
uv run orchestrate --config rf_vs_dt.json --skip-install
```

### Example 7: Compare scikit-learn branches (requires submodule)

First, set up the submodule:
```bash
git submodule add https://github.com/scikit-learn/scikit-learn.git scikit-learn
git submodule update --init --recursive
```

Create `branch_comparison.json`:
```json
{
  "models": {
    "DecisionTreeClassifier": {
      "max_depth": [10]
    }
  },
  "branches": ["main", "1.3.X", "1.4.X"],
  "n_samples": [1000, 5000],
  "n_features": [20],
  "n_repeats": 5
}
```

Run the comparison:
```bash
uv run orchestrate --config branch_comparison.json
```

This will:
1. Checkout `main` branch, install scikit-learn, run benchmarks
2. Checkout `1.3.X` branch, install scikit-learn, run benchmarks
3. Checkout `1.4.X` branch, install scikit-learn, run benchmarks
4. Save all results with branch and version information

## Analyzing Results

### Example 8: Load and analyze results in Python

```python
import json
import pandas as pd

# Load orchestration results
with open('results/20260129_205904_summary.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Compare models
print(df.groupby('model')['train_time_mean'].describe())

# Find fastest configuration
fastest = df.loc[df['train_time_mean'].idxmin()]
print(f"Fastest: {fastest['model']} with params {fastest['model_params']}")
```

### Example 9: Visualize in Jupyter notebook

```bash
# Start Jupyter
uv run jupyter notebook

# Open visualize_results.ipynb
# The notebook will automatically load the latest results
```

### Example 10: Export to CSV for external analysis

```python
import json
import pandas as pd

with open('results/summary.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Flatten model_params
params_df = pd.json_normalize(df['model_params'])
df_flat = pd.concat([df.drop('model_params', axis=1), params_df], axis=1)

# Export
df_flat.to_csv('results/benchmark_results.csv', index=False)
```

## Advanced Workflows

### Example 11: Benchmark across different hardware

```bash
# Run on CPU
uv run orchestrate --config config.json --output-dir results/cpu

# Copy results to different machine with GPU, then:
# uv run orchestrate --config config.json --output-dir results/gpu

# Compare in notebook by loading both result files
```

### Example 12: Continuous benchmarking

```bash
#!/bin/bash
# benchmark_cron.sh - Run daily benchmarks

DATE=$(date +%Y%m%d)
uv run orchestrate \
  --config production_config.json \
  --output-dir results/daily/$DATE \
  --skip-install

# Archive results
tar -czf results/daily/${DATE}.tar.gz results/daily/$DATE/
```

### Example 13: Custom analysis script

```python
#!/usr/bin/env python3
"""Find performance regressions between branches."""

import json
from pathlib import Path
import pandas as pd

def compare_branches(file1, file2):
    with open(file1) as f:
        data1 = pd.DataFrame(json.load(f))
    with open(file2) as f:
        data2 = pd.DataFrame(json.load(f))
    
    # Merge on model and params
    merged = data1.merge(
        data2,
        on=['model', 'n_samples', 'n_features'],
        suffixes=('_old', '_new')
    )
    
    # Calculate regression
    merged['speedup'] = (
        merged['train_time_mean_old'] / merged['train_time_mean_new']
    )
    
    # Find regressions (slowdowns > 5%)
    regressions = merged[merged['speedup'] < 0.95]
    
    if len(regressions) > 0:
        print("⚠️  Performance regressions detected:")
        for _, row in regressions.iterrows():
            print(f"  {row['model']}: {row['speedup']*100:.1f}% of original speed")
    else:
        print("✓ No significant regressions")

if __name__ == "__main__":
    compare_branches(
        'results/main_summary.json',
        'results/feature_branch_summary.json'
    )
```

## Tips and Best Practices

1. **Start small**: Use `--n-repeats 2` and small datasets for testing
2. **Use --skip-install**: When comparing parameters on the same sklearn version
3. **Separate results**: Use descriptive output directories for different experiments
4. **Version control configs**: Keep your config files in git for reproducibility
5. **Monitor resources**: Large forests with many samples can use significant memory
6. **Statistical significance**: Use `n_repeats >= 5` for reliable measurements
7. **Document experiments**: Add metadata files explaining what each benchmark tests

## Common Configurations

### Quick smoke test
```json
{
  "models": {"DecisionTreeClassifier": {}},
  "branches": ["current"],
  "n_samples": [100],
  "n_features": [10],
  "n_repeats": 2
}
```

### Comprehensive benchmark
```json
{
  "models": {
    "DecisionTreeClassifier": {
      "max_depth": [5, 10, 15, 20, null],
      "min_samples_split": [2, 5, 10, 20]
    },
    "RandomForestClassifier": {
      "n_estimators": [10, 50, 100, 200],
      "max_depth": [5, 10, 15, 20, null]
    }
  },
  "branches": ["current"],
  "n_samples": [1000, 5000, 10000, 50000],
  "n_features": [10, 20, 50, 100],
  "n_repeats": 10
}
```

### Scaling test
```json
{
  "models": {
    "RandomForestClassifier": {
      "n_estimators": [100]
    }
  },
  "branches": ["current"],
  "n_samples": [100, 500, 1000, 5000, 10000, 50000, 100000],
  "n_features": [20],
  "n_repeats": 5
}
```
