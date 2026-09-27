# robust-streaming-ml

A portfolio foundation for a reproducible, deployment-oriented streaming audio-classification project. The planned work demonstrates ML and algorithms engineering practices: validated data splits, explainable baselines, controlled robustness testing, and CPU-ready inference.

This repository intentionally contains no dataset, trained model, measured metric, or performance claim yet. See [DESIGN.md](DESIGN.md) for the bounded proposal and [TASKS.md](TASKS.md) for the delivery plan.

## Repository layout

- `src/robust_streaming_ml/` — package code: configuration schema (`config.py`), seeding (`seed.py`), run start and run directories (`run_dir.py`), dataset acquisition/validation (`data/`), log-mel features (`features.py`), the CNN (`model.py`), checkpoints (`checkpoint.py`), and training (`training/`)
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

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push and pull request on `ubuntu-latest` with Python 3.10 (`UV_PYTHON=3.10`). It installs uv 0.12.13, runs `uv sync --locked` (which fills uv's cache), then runs `uv run python scripts/check.py`. The distribution check inside it stays offline.

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
- `runs/`, `artifacts/`, `checkpoints/`, the top-level `/data/` folder, and model files are git-ignored; do not commit them. Any result reported in `RESULTS.md` must reference a run directory created by `start_run`.

## Dataset acquisition

The dataset is Speech Commands v0.02 (10-keyword subset), licensed CC BY 4.0. Attribution is required: see [docs/datasets/speech_commands_v0.02.md](docs/datasets/speech_commands_v0.02.md) and the [data contract](docs/data-contract.md).

```bash
uv run python scripts/acquire_speech_commands.py
```

This downloads about 2.4 GB over HTTPS to `data/raw/speech_commands_v0.02.tar.gz` and accepts the archive only if its size and SHA-256 match the pinned values. It then extracts to `data/raw/speech_commands_v0.02/`. Extraction rejects absolute paths, `..`, links, and device files, and checks the pinned `LICENSE`, `README.md`, and split-list hashes. A completed extraction is marked by `.acquisition.json`. Re-running reuses a verified archive and a completed extraction. A mismatched archive or an incomplete directory stops the script with an error; it is never overwritten. Everything under the top-level `data/` folder is git-ignored.

Then validate the extraction and build the manifest:

```bash
uv run python scripts/build_speech_commands_manifest.py
```

This decodes every file in the ten label folders with a strict WAV parser. It enforces the [data contract](docs/data-contract.md):
- format (mono, 16 kHz, 16-bit PCM, 1 to 16,000 frames);
- filename identity;
- official split lists, with train derived from everything else;
- duplicate content (`audio_sha256`);
- speaker leakage;
- a byte-for-byte cross-check against the pinned archive.

It writes `data/manifests/speech_commands_v0.02_kw10.csv` plus a JSON sidecar, and deterministic reports in `data/reports/`. The manifest's SHA-256 is pinned in code, so a manifest with any other hash is rejected.

The command fails closed: on any violation it exits non-zero, removes any existing manifest, and writes a `failed` report listing every error. It needs the code revision to be resolvable by git, like `start_run`.

The real-data integration test (`pytest -m integration`) runs automatically when the local extraction is present, and is skipped otherwise.

## Baseline training

```bash
uv run python scripts/train_baseline.py --config configs/kw10_cnn_baseline.yaml
```

This trains the 61,052-parameter log-mel CNN described in [docs/baseline.md](docs/baseline.md), using the train split, and selects checkpoints by validation loss. The test split is never loaded.

Before `start_run` creates anything, it checks:
- the config against the pinned, validated manifest, the manifest sidecar, and the validation report;
- that the source tree is clean (`--allow-dirty-source` is for development runs only).

Each run directory holds:
- `config.yaml` and `metadata.json`;
- `train_log.jsonl`, with one line per epoch;
- `checkpoints/{best,last}.{pt,json}`, in the versioned format that loads with `weights_only=True`;
- `summary.json`.

Like `start_run`, training needs git to resolve the source revision.

## Status

Milestones 1 (repository tooling) and 2 (dataset acquisition and validation) are complete. Milestone 3 (baseline training) code is in place; see `TASKS.md` and `RESULTS.md` for its status.
