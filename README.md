# robust-streaming-ml

A portfolio foundation for a reproducible, deployment-oriented streaming audio-classification project. The planned work demonstrates ML and algorithms engineering practices: validated data splits, explainable baselines, controlled robustness testing, and CPU-ready inference.

This repository intentionally contains no dataset, trained model, measured metric, or performance claim yet. See [DESIGN.md](DESIGN.md) for the bounded proposal and [TASKS.md](TASKS.md) for the delivery plan.

## Repository layout

- `src/robust_streaming_ml/` — package code: configuration schema (`config.py`), seeding (`seed.py`), run start and run directories (`run_dir.py`)
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

Run all checks (format, lint, type check, tests, distribution); exits non-zero if any fails:

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
| Distribution | `uv run python scripts/check_dist.py` |

The distribution check needs `uv` on PATH and runs offline from the local uv cache populated by `uv sync`. It:
1. builds the wheel with the locked `hatchling` (`--no-build-isolation`) and checks its contents;
2. creates a fresh temporary environment containing only the locked runtime dependencies and pytest, then installs the wheel into it;
3. confirms `robust_streaming_ml` is imported from that environment's site-packages, then runs `tests/test_import.py` there.

## Configuration, seeds, and run directories

- Configurations are YAML files validated by `robust_streaming_ml.config.load_config`. Unknown keys, wrong types, and out-of-range values raise `ConfigError`. See `configs/example.yaml`.
- `robust_streaming_ml.seed.seed_everything(seed, deterministic=True)` seeds Python, NumPy, and PyTorch and enables deterministic PyTorch algorithms. `numpy_generator` and `torch_generator` return explicitly seeded, independent generators.
- `robust_streaming_ml.run_dir.start_run(config)` is the single entry point for starting a run; every training, evaluation, corruption, or benchmark entry point must begin with it. In order, it:
  1. resolves the git commit and dirty state of the repository containing the imported package's own source directory. Callers cannot choose a different directory, and the package files must be tracked by that repository. Runs therefore require an editable install of a git checkout (the default with `uv sync`). A non-editable install, such as a wheel in site-packages, fails with `ProvenanceError`, even when that environment sits inside a checkout;
  2. calls `seed_everything(config.seed, deterministic=config.deterministic)` and verifies the resulting PyTorch/cuDNN/environment state matches the configuration;
  3. creates the run directory and writes `config.yaml` and `metadata.json`.
- **Unavailable-revision policy: fail.** If git cannot resolve the revision (not a repository, no commits, git not installed, or git refusing the repository, e.g. "dubious ownership"), `start_run` raises `ProvenanceError` containing git's diagnostic. This happens before seeding or creating any directory. There is no manual revision override. Fix the git problem (for "dubious ownership", follow git's printed `safe.directory` instruction) and rerun.
- A dirty working tree does not stop a run, but it is recorded as `source.dirty: true`. Results reported in `RESULTS.md` should come from runs with `dirty: false`.

### Artifact-directory conventions

- Every run writes to its own directory `<run.output_root>/<run_id>/` (default root `runs/`), where `run_id` is `<UTC timestamp>_<experiment name>_<8 hex chars>`. Existing run directories are never reused or overwritten.
- Each run directory contains `config.yaml` (the validated configuration used) and `metadata.json`: run ID, UTC creation time, config SHA-256, `source` (git `commit` and `dirty`; dirty covers modified tracked files and untracked, non-ignored files), `seeding` (the seed and determinism state observed after seeding was applied: `torch_deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`), and `environment` (Python/platform/package versions, PyTorch runtime details, command-line arguments).
- Later milestones add their outputs (checkpoints, predictions, reports) inside the same run directory.
- `runs/`, `artifacts/`, `checkpoints/`, `data/`, and model files are git-ignored; do not commit them. Any result reported in `RESULTS.md` must reference a run directory created by `start_run`.

## Status

Milestone 1 (repository tooling and test baseline) is complete. Next: Milestone 2 in `TASKS.md`.
