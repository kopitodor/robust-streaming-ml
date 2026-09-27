# robust-streaming-ml

A portfolio foundation for a reproducible, deployment-oriented streaming audio-classification project. The planned work demonstrates ML and algorithms engineering practices: validated data splits, explainable baselines, controlled robustness testing, and CPU-ready inference.

This repository intentionally contains no dataset, trained model, measured metric, or performance claim yet. See [DESIGN.md](DESIGN.md) for the bounded proposal and [TASKS.md](TASKS.md) for the delivery plan.

## Repository layout

- `src/robust_streaming_ml/` — package code: configuration schema (`config.py`), seeding (`seed.py`), run directories (`run_dir.py`)
- `tests/` — automated tests (pytest)
- `configs/` — versioned experiment configuration
- `scripts/` — explicit operational entry points
- `docs/` — supporting documentation

## Environment setup

Requires Python >= 3.10 (declared in `pyproject.toml`). The checks recorded in [RESULTS.md](RESULTS.md) were run with CPython 3.14.2 on Windows; other versions are not yet verified.

With [uv](https://docs.astral.sh/uv/) (recommended; uses the committed `uv.lock`, CPU-only PyTorch wheels, and the `dev` dependency group):

```bash
uv sync
```

With pip (pip >= 25.1 for `--group`; this does not use the lockfile and installs the default PyTorch build for your platform; this path has not yet been exercised):

```bash
python -m venv .venv
python -m pip install -e . --group dev
```

## Development commands

Run all checks (format, lint, type check, tests); exits non-zero if any fails:

```bash
uv run python scripts/check.py
```

Individual commands:

| Purpose | Command |
| --- | --- |
| Format | `uv run ruff format .` |
| Format check | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy` |
| Tests | `uv run pytest` |

## Configuration, seeds, and run directories

- Configurations are YAML files validated by `robust_streaming_ml.config.load_config`. Unknown keys, wrong types, and out-of-range values raise `ConfigError`. See `configs/example.yaml`.
- `robust_streaming_ml.seed.seed_everything(seed, deterministic=True)` seeds Python, NumPy, and PyTorch and enables deterministic PyTorch algorithms. `numpy_generator` and `torch_generator` return explicitly seeded, independent generators.
- `robust_streaming_ml.run_dir.create_run_dir(config)` creates a new run directory and records the configuration and environment metadata.

### Artifact-directory conventions

- Every run writes to its own directory `<run.output_root>/<run_id>/` (default root `runs/`), where `run_id` is `<UTC timestamp>_<experiment name>_<8 hex chars>`. Existing run directories are never reused or overwritten.
- Each run directory contains `config.yaml` (the validated configuration used) and `metadata.json` (run ID, UTC creation time, seed, determinism flag, config SHA-256, Python/platform/package versions, PyTorch runtime details, git commit and dirty state, and command-line arguments). `git` is `null` when the revision cannot be determined.
- Later milestones add their outputs (checkpoints, predictions, reports) inside the same run directory.
- `runs/`, `artifacts/`, `checkpoints/`, `data/`, and model files are git-ignored; do not commit them. Any result reported in `RESULTS.md` must reference a run directory produced this way.

## Status

Milestone 1 (repository tooling and test baseline) is complete. Next: Milestone 2 in `TASKS.md`.
