# Tasks

## 1. Repository tooling and test baseline

- [x] Configure package metadata, supported Python versions, formatting, linting, type checking, and test runner.
  - hatchling package with `py.typed`; `requires-python >= 3.10` (only 3.14.2 verified so far); ruff (format + lint), mypy (strict, pydantic plugin), pytest; `uv.lock` with CPU-only PyTorch index.
- [x] Add a minimal import test and CI-ready local commands.
  - `tests/test_import.py`; `scripts/check.py` runs all checks and exits non-zero on failure. `scripts/check_dist.py` builds the wheel offline, installs it into an isolated temporary environment with the locked dependencies, and runs the import smoke tests there. No CI workflow file is added yet.
- [x] Define configuration schema and deterministic seed utility.
  - `config.py` (strict YAML schema, `ConfigError`), `seed.py` (`seed_everything`, `numpy_generator`, `torch_generator`), `run_dir.py` (`start_run`: resolves git provenance from the package's own source directory only, or fails with `ProvenanceError`, applies and verifies seeding, then writes `config.yaml` + `metadata.json`), with unit tests including temporary-git-repository provenance tests.
- [x] Document environment setup and artifact-directory conventions.
  - See README sections "Environment setup", "Development commands", and "Artifact-directory conventions".

## 2. Dataset acquisition and validation

- [x] Select one public dataset and define the data contract (planning only; nothing downloaded).
  - Speech Commands v2 (`v0.02`), 10 labels, official validation/test lists; `_silence_`/`_unknown_` out of scope. See `docs/decisions/0001-dataset-speech-commands-v2.md` and `docs/data-contract.md`.
- [x] Verify and record source, version, archive SHA-256, license text, and attribution from the actual download (data contract section 1).
  - `docs/datasets/speech_commands_v0.02.md`: archive SHA-256 pinned (MD5 matches the server's published hash), CC BY 4.0 read from `LICENSE`, attribution and citation recorded, split lists cross-checked structurally.
- [x] Human sign-off on the license (`license_verified_by` / `license_verified_on` in the verification record): project owner, 2026-09-27.
- [x] Implement acquisition without committing data.
  - `robust_streaming_ml.data.acquisition` + `scripts/acquire_speech_commands.py`: pinned SHA-256/size download over HTTPS, safe extraction (no absolute/`..`/link/device members), pinned-file hash checks, idempotent re-runs; tested on synthetic archives. `.gitignore` now anchors `/data/` so the `data` package is not ignored.
- [x] Generate the manifest CSV and JSON sidecar exactly per the data contract schema (section 7), deterministically.
  - `scripts/build_speech_commands_manifest.py` writes `data/manifests/speech_commands_v0.02_kw10.{csv,json}` and `data/reports/…_validation.{json,md}` (git-ignored; see contract amendment 1). 38,546 rows; SHA-256 `f811a20d…ab6131c`, pinned in `SPEECH_COMMANDS_KW10`.
- [x] Implement the contract's validation rules (sections 2–6) with tests on small synthetic fixtures: format/duration, labels, split-list handling, duplicates, and speaker leakage.
  - Strict WAV parser, row checks, split-list parsing, content-hash duplicates, speaker leakage, and an archive cross-check, each with synthetic-fixture tests for its failure modes. A real-data integration test runs when `data/raw` is present. Result on the real data: passed, 0 errors, 1,477 within-split duplicate-content warnings (640 groups, all single-speaker).

## 3. Baseline training

- [x] Implement versioned log-mel feature extraction and fixed-window policy.
  - `features.py`: `start_zero_pad` window (never crops), HTK log-mel with a fingerprint checked on checkpoint load; `window` and `features` config sections, both version 1.
- [x] Implement a small PyTorch CNN and explicit training configuration.
  - `model.py` (`kws_cnn_v1`, 61,052 parameters), `TrainingConfig` (`schema_version` 1), `configs/kw10_cnn_baseline.yaml`, versioned checkpoints (`checkpoint.py`), and `training/` (manifest compatibility gate, verified audio loading, seeded loaders, training loop). See `docs/baseline.md`.
- [x] Train with recorded seed, manifest, configuration, dependency context, and checkpoints.
  - Run `20260927T210911Z_kw10-cnn-baseline_6c9d6cdf` (commit `5babe3f`, seed 1234, manifest `f811a20d…`): 30 epochs, best checkpoint at epoch 18 by validation loss. `metadata.json` records provenance, seeding, and the environment; `uv.lock` pins dependencies. Details in RESULTS.md. The test split was not used.
- [x] Add regression tests for feature and model shapes.
  - Tests cover:
    - feature shapes, determinism, filterbank values, and fingerprints;
    - model shapes and the pinned parameter count;
    - checkpoint round trip and every rejection path;
    - config validation;
    - the manifest gate, test-split refusal, and batches that don't depend on worker count;
    - synthetic end-to-end runs: artifacts, reproducibility, and fail-closed paths.

## 4. Evaluation and calibration

- [x] Implement accuracy, macro F1, confusion matrix, ECE, and NLL.
  - `evaluation/metrics.py`, defined in `docs/evaluation-protocol.md` §5: zero-division gives 0, the confusion matrix uses a fixed label order, NLL is computed from float64 log-softmax, and ECE is top-label with 15 equal-width bins.
- [x] Add validation/test evaluation reports and prediction artifacts.
  - `scripts/evaluate_baseline.py`: formal run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` (commit `a3d208f`, checkpoint `d21f8928…`). It wrote the prediction CSVs, `calibration.json`, `report.json`/`.md`, and `summary.json`. Results and limitations are in RESULTS.md.
- [x] Implement and document a calibration procedure if justified by validation data.
  - Temperature scaling (`evaluation/calibration.py`): a golden-section search on validation NLL over `β = 1/T` in [0.05, 20], fitted only on validation (any other split is refused) and persisted in `calibration.json` with compatibility metadata. Raw and calibrated results are always both reported; the fitted value is in RESULTS.md.
- [x] Verify metrics against fixed fixtures.
  - `tests/test_metrics.py`: hand-computed confusion matrix, accuracy, per-class and macro F1, zero division, NLL, and ECE (including bin edges).
- [x] Decide and document how evaluation treats within-split duplicate audio. The test split has 119 groups covering 240 files, i.e. 121 redundant copies (see `docs/datasets/speech_commands_v0.02.md`). The manifest keeps them all.
  - Decision (`docs/evaluation-protocol.md` §7): the official result retains every row. A secondary view that keeps one row per `audio_sha256` (the smallest path) is reported separately and never replaces it.

## 5. Robustness suite

- [x] Define deterministic noise, reverberation, resampling, clipping, and codec-artifact protocols.
  - `docs/robustness-protocol.md` (`kw10-robustness-v1`) and `robustness/corruptions.py`: 5 version-1 transforms × 3 severities, per-example seeds from `relative_path`, and a pinned Opus backend (`soundfile` 0.14.0 / libsndfile 1.2.2). Committed before any real run.
- [x] Generate corruption manifests with severity and seed metadata.
  - `conditions/<corruption>/<severity>/corruption_manifest.csv`: per clip, the source hash, corruption, version, severity, seed, output SHA-256, and measurements.
- [x] Evaluate every corruption/severity pair and compute degradation from clean metrics.
  - Run `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` (commit `e024ee8`): all 15 conditions on the full test split, raw probabilities, degradation against the Milestone 4 raw test metrics. See RESULTS.md.
- [x] Present failures and limitations alongside aggregate tables.
  - RESULTS.md lists every condition, including negative degradation, and records that there were no failures or non-runs, with the limitations stated.

## 6. Streaming and ONNX inference

- [x] Specify window, hop, padding, state, and aggregation semantics.
  - `docs/streaming-onnx-protocol.md`: `kw10-streaming-v1` (1 s window, 100 ms hop, no startup padding, one zero-padded final window, batch-1 per-window inference, causal mean of the last 5 windows) and `kw10-onnx-v1` (opset 18, tensor contract, ONNX Runtime 1.30.0 CPU, parity tolerances). Committed before any streaming or export code.
- [x] Implement streaming-window inference and tests for boundaries.
  - `streaming/windowing.py` (windower and an independent offline reference) and `streaming/classifier.py` (batch-1 prediction, causal aggregation, PyTorch backend).
  - `tests/test_streaming.py` covers window and hop boundaries, arbitrary chunkings (bit-identical to offline), startup and final padding, short and empty audio, reset, timestamps and order, and independence from future audio.
- [ ] Export ONNX with recorded opset and input contract.
- [ ] Compare ONNX and PyTorch outputs using fixed fixtures.

## 7. Benchmarking, documentation, and demo

- [ ] Define reproducible CPU latency protocol including warm-up, iterations, threads, runtime, and hardware metadata.
- [ ] Record p50/p95 per-window latency without extrapolating to production.
- [ ] Build the local user-facing demo.
- [ ] Complete result tables, limitations, reproducibility instructions, and demo walkthrough.
