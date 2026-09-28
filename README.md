# robust-streaming-ml

A portfolio project: a reproducible, deployment-oriented streaming keyword classifier on Speech Commands v0.02, covering 10 keywords. It demonstrates ML and algorithms engineering practices:
- validated data splits;
- a small, documented CNN baseline;
- calibrated evaluation;
- controlled robustness testing;
- streaming-window inference;
- an ONNX export verified against PyTorch;
- a documented CPU latency benchmark;
- a local demo.

The repository contains code, tests, protocols, and documentation only. The dataset, checkpoints, ONNX exports, run directories, and demo outputs are generated locally under git-ignored folders. Every measured number is in [RESULTS.md](RESULTS.md), tied to a run directory and a committed protocol. It is not a state-of-the-art, production, or real-time claim. See [DESIGN.md](DESIGN.md) for the design and acceptance status, and [TASKS.md](TASKS.md) for the delivery record.

## Results at a glance

All from RESULTS.md, single model and seed, official test split (4,074 clips, duplicates retained):

| What | Measured |
| --- | --- |
| Clean test accuracy / macro F1 (`kw10-clean-eval-v1`) | 0.9639 / 0.9639; NLL 0.1011, ECE 0.0082 (raw) |
| Largest robustness degradation (`kw10-robustness-v1`) | accuracy drop 0.500 (0.9639 → 0.4639, resample via 2 kHz). Opus at `compression_level` 0.9 and 0.95 changes accuracy by at most 0.002. |
| ONNX vs PyTorch parity (`kw10-onnx-v1`) | max \|Δlogit\| 1.8e-4 over 39,703 windows (tolerance 1e-2); streaming decision = clip-level prediction on 4,074/4,074 clips |
| CPU latency (`kw10-latency-v1`, AMD Ryzen 7 5800H, 1 thread) | ONNX streaming step p50 2.43 ms, p95 2.71 ms; the PyTorch eager reference was faster (p50 1.95 ms) |

## Repository layout

- `src/robust_streaming_ml/` — package code: configuration schema (`config.py`), seeding (`seed.py`), run start and run directories (`run_dir.py`), dataset acquisition/validation (`data/`), log-mel features (`features.py`), the CNN (`model.py`), checkpoints (`checkpoint.py`), training (`training/`), evaluation and calibration (`evaluation/`), the robustness suite (`robustness/`), streaming-window inference (`streaming/`), ONNX export, runtime, and parity (`export/`), the CPU latency benchmark (`benchmark/`), and the local demo (`demo/`)
- `tests/` — automated tests (pytest)
- `configs/` — versioned experiment configuration
- `scripts/` — explicit operational entry points
- `docs/` — supporting documentation

## Environment setup

Requires Python >= 3.10 (declared in `pyproject.toml`). The checks and runs recorded in [RESULTS.md](RESULTS.md) used CPython 3.14.2 on Windows, and the full check was also rehearsed locally with CPython 3.10 (see RESULTS.md). The GitHub Actions workflow itself has not run yet. Formal ONNX results require ONNX Runtime 1.30.0, the version locked for Python ≥ 3.11; Python 3.10 locks 1.23.2, which is used only by tests.

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

## Evaluation

```bash
uv run python scripts/evaluate_baseline.py --config configs/kw10_cnn_baseline_eval.yaml
```

This runs protocol `kw10-clean-eval-v1` ([docs/evaluation-protocol.md](docs/evaluation-protocol.md)) on the pinned Milestone 3 checkpoint.

Before `start_run`, it checks:
- the pinned checkpoint SHA-256 and its compatibility;
- that the checkpoint came from a clean tree whose commit is an ancestor of HEAD, and that it agrees with the training run's metadata;
- the validated-manifest gate;
- that the source tree is clean.

It then predicts on validation, fits temperature scaling on validation only, and writes `calibration.json`. Only then does it predict on test, and it computes accuracy, macro F1, the confusion matrix, NLL, and 15-bin ECE, raw and calibrated.

It writes to the new run directory:
- `predictions/*.csv`, from which every aggregate can be regenerated;
- `calibration.json`;
- `report.json`, `report.md`, and `summary.json`.

A duplicate-content sensitivity view is reported separately and never replaces the official result.

## Robustness suite

```bash
uv run python scripts/evaluate_robustness.py --config configs/kw10_cnn_baseline_robustness.yaml
```

This runs protocol `kw10-robustness-v1` ([docs/robustness-protocol.md](docs/robustness-protocol.md)) on the official test split, using the same checkpoint with raw probabilities and no refitting. It covers 5 corruptions (noise, reverb, resample, clipping, Opus codec) at 3 fixed severities each. Each clip's seed is derived from its identity and the protocol seed.

It fails closed on:
- the checkpoint and manifest gates;
- the pinned Milestone 4 baseline report;
- the exact codec backend (`soundfile` 0.14.0 / libsndfile 1.2.2, installed by `uv sync`);
- a dirty tree;
- a failure to reproduce the Milestone 4 clean test logits bit-for-bit.

It writes per-condition predictions and corruption manifests, plus `report.json` and `report.md`, with degradation = clean − corrupted.

## Streaming and ONNX export

```bash
uv run python scripts/export_onnx.py --config configs/kw10_cnn_baseline_onnx.yaml
```

The protocols are [docs/streaming-onnx-protocol.md](docs/streaming-onnx-protocol.md):
- **`kw10-streaming-v1`:**
  - 1 s windows, 100 ms hop, integer sample timestamps;
  - no startup padding, and one zero-padded final window;
  - every window inferred on its own (batch 1) with raw probabilities;
  - a causal mean over the last 5 windows;
  - bit-identical to an independent offline reference for any chunking.
- **`kw10-onnx-v1`:**
  - the validated log-mel front end (`torch.stft` exported as ONNX `STFT`) plus the CNN, opset 18, `waveform [batch, 16000]` → `logits [batch, 10]`;
  - metadata tying the model to the checkpoint, labels, features, manifest, source revision, and toolchain;
  - ONNX Runtime 1.30.0 on `CPUExecutionProvider` only;
  - predeclared parity tolerances (logits 1e-2, probabilities 1e-3).

It fails closed on:
- the checkpoint, manifest, and pinned Milestone 4 baseline gates;
- the exact export toolchain (torch 2.14.0+cpu, onnx 1.23.0, onnxscript 0.7.2) and onnxruntime version;
- a dirty tree;
- a PyTorch path that does not reproduce the Milestone 4 test logits bit-for-bit.

It writes `onnx/model.onnx` and its sidecar, then compares PyTorch and ONNX Runtime on every test clip and on concatenation streams of test clips. The outputs are `parity/*.csv`, `report.json`, `report.md`, and `summary.json`. A tolerance violation still writes the full report, but the run's status is `failed` and the command exits 1.

## CPU latency benchmark

```bash
uv run python scripts/benchmark_latency.py --config configs/kw10_onnx_benchmark.yaml
```

This runs protocol `kw10-latency-v1` ([docs/benchmark-protocol.md](docs/benchmark-protocol.md)) on the pinned Milestone 6 ONNX export. The export's report SHA-256 is pinned and it must record passed parity.

It times three paths, each at batch 1 with 1 and 4 threads:
- the deployment-oriented **streaming step**: push 100 ms of audio, then window, ONNX Runtime inference (log-mel feature extraction is inside the graph), softmax, and aggregation;
- `session.run` alone;
- a PyTorch eager reference step.

Each block has 500 warm-up steps and 5,000 individually timed steps, repeated 3 times. The benchmark reports nearest-rank p50/p95 and keeps every raw timing in `timings.csv`. It records the OS, CPU model, logical and physical cores, and runtime versions.

It fails closed on the checkpoint, manifest, and export gates, the ONNX Runtime version, and a dirty tree. The numbers describe one local machine; they are not a production or real-time claim.

## Local demo

```bash
uv run python scripts/demo.py --audio path/to/clip.wav
uv run python scripts/demo.py --audio path/to/clip.wav --corruption noise --severity snr_10db --open
uv run python scripts/demo.py --list-corruptions
```

The demo streams one local WAV file through the pinned, parity-verified ONNX model. The file must be 16 kHz mono 16-bit PCM, and nothing is resampled. Audio is pushed in 100 ms chunks, like a live source.

It prints each window's predicted class, confidence, causal aggregate, and ONNX Runtime inference time, then the stream decision. It can also apply one `kw10-robustness-v1` corruption and show clean and corrupted output side by side.

It writes a self-contained HTML page (no scripts or external resources) and a JSON record under the git-ignored `artifacts/demo/`. The page shows the model, checkpoint, export, dataset version, manifest, configuration, and limitations. Nothing is uploaded, and no server, share link, or network service is involved. `--open` opens the local page in your browser.

Demo output is not a measured result, and the model has no silence or unknown-word rejection. See [docs/demo.md](docs/demo.md).

## Reproducing the pipeline

Run these in order from the repository root after `uv sync`. Each step fails closed and writes a new git-ignored run directory.

```bash
uv run python scripts/acquire_speech_commands.py
uv run python scripts/build_speech_commands_manifest.py
uv run python scripts/train_baseline.py --config configs/kw10_cnn_baseline.yaml
uv run python scripts/evaluate_baseline.py --config configs/kw10_cnn_baseline_eval.yaml
uv run python scripts/evaluate_robustness.py --config configs/kw10_cnn_baseline_robustness.yaml
uv run python scripts/export_onnx.py --config configs/kw10_cnn_baseline_onnx.yaml
uv run python scripts/benchmark_latency.py --config configs/kw10_onnx_benchmark.yaml
uv run python scripts/demo.py --audio path/to/clip.wav
```

- **Pins:** the committed configs pin the recorded runs' checkpoint and report SHA-256s. After retraining or re-running a step, update the pins in the downstream configs; a pin mismatch is refused, never ignored.
- **Clean tree:** formal runs need a clean tree and a git revision that `start_run` can resolve.
- **"Dubious ownership":** in a checkout where git reports this (as in this one), use a process-scoped override rather than global git configuration:

  ```bash
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$(pwd)" uv run python scripts/...
  ```

## Status

All seven milestones in `TASKS.md` are complete; see `RESULTS.md`:
1. repository tooling;
2. dataset acquisition and validation;
3. baseline training;
4. clean evaluation and calibration (`kw10-clean-eval-v1`);
5. robustness suite (`kw10-robustness-v1`);
6. streaming and ONNX inference (`kw10-streaming-v1` / `kw10-onnx-v1`);
7. CPU benchmark (`kw10-latency-v1`), documentation, and demo.

Known gaps:
- the GitHub Actions workflow has not run on GitHub;
- results cover one model, one seed, one machine, and one runtime version;
- the model is closed-set, with no silence or unknown-word rejection.
