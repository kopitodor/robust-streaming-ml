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

- [ ] Select one public, license-supported dataset; record source, version, license, attribution, and access procedure.
- [ ] Implement acquisition without committing data.
- [ ] Generate a manifest with file identity, label, provenance, and checksum fields.
- [ ] Validate audio readability, label coverage, duplicates, grouping constraints, and disjoint splits.

## 3. Baseline training

- [ ] Implement versioned log-mel feature extraction and fixed-window policy.
- [ ] Implement a small PyTorch CNN and explicit training configuration.
- [ ] Train with recorded seed, manifest, configuration, dependency context, and checkpoints.
- [ ] Add regression tests for feature and model shapes.

## 4. Evaluation and calibration

- [ ] Implement accuracy, macro F1, confusion matrix, ECE, and NLL.
- [ ] Add validation/test evaluation reports and prediction artifacts.
- [ ] Implement and document a calibration procedure if justified by validation data.
- [ ] Verify metrics against fixed fixtures.

## 5. Robustness suite

- [ ] Define deterministic noise, reverberation, resampling, clipping, and codec-artifact protocols.
- [ ] Generate corruption manifests with severity and seed metadata.
- [ ] Evaluate every corruption/severity pair and compute degradation from clean metrics.
- [ ] Present failures and limitations alongside aggregate tables.

## 6. Streaming and ONNX inference

- [ ] Specify window, hop, padding, state, and aggregation semantics.
- [ ] Implement streaming-window inference and tests for boundaries.
- [ ] Export ONNX with recorded opset and input contract.
- [ ] Compare ONNX and PyTorch outputs using fixed fixtures.

## 7. Benchmarking, documentation, and demo

- [ ] Define reproducible CPU latency protocol including warm-up, iterations, threads, runtime, and hardware metadata.
- [ ] Record p50/p95 per-window latency without extrapolating to production.
- [ ] Build the local user-facing demo.
- [ ] Complete result tables, limitations, reproducibility instructions, and demo walkthrough.
