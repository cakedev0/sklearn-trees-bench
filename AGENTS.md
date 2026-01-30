# Repository Guidelines

## Project Structure & Module Organization
- `sklearn_trees_bench/` contains the CLI entry points: `train_model.py` for single runs and `orchestrate.py` for multi-branch benchmarks.
- `example_config.json` shows benchmark configuration patterns.
- `results/` is the default output directory for JSON benchmark artifacts and summaries.
- `visualize_results.ipynb` is the notebook for analysis and plotting.
- `scikit-learn/` is an optional git submodule used for benchmarking specific scikit-learn branches.

## Build, Test, and Development Commands
- `uv sync` installs dependencies into the managed environment.
- `uv pip install -e .` installs the package in editable mode for CLI development.
- `uv run train-model --model DecisionTreeClassifier --n-repeats 5` runs a single benchmark.
- `uv run orchestrate --config example_config.json` runs grid benchmarks.
- `uv run jupyter notebook visualize_results.ipynb` opens the results notebook.
- `uv run ruff format` and `uv run ruff check` format and lint the code.

## Coding Style & Naming Conventions
- Python ≥3.10; 4-space indentation; prefer clear, descriptive names mirroring CLI flags (e.g., `n_samples`, `n_features`).
- Use NumPy-style docstrings for public functions and keep argument defaults consistent with CLI defaults.
- Keep output file naming consistent with orchestration patterns (timestamp, branch, model, params).

## Testing Guidelines
- This repository does not currently ship a dedicated test suite; `uv run pytest` is available if tests are added.
- Avoid running the `scikit-learn/` submodule test suite unless explicitly needed; it is large and unrelated to this benchmark harness.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, sentence-case (e.g., “Add comprehensive usage examples”).
- Prefer one logical change per commit; include benchmark config changes alongside code when relevant.
- PRs should include a clear summary, the commands used to reproduce results, and any new/updated JSON config files.
- If results are attached, note the output directory and the scikit-learn branch/version tested.

## Configuration & Submodule Notes
- Orchestration requires a config JSON with `models` and `branches` keys; see `example_config.json`.
- To benchmark specific scikit-learn branches, initialize the submodule: `git submodule update --init --recursive`.
