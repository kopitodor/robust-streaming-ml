# Results

The experiment results table holds the Milestone 4 clean evaluation (protocol `kw10-clean-eval-v1`, docs/evaluation-protocol.md). The robustness table holds Milestone 5 (protocol `kw10-robustness-v1`, docs/robustness-protocol.md). Milestone 6 streaming and ONNX parity results (protocols `kw10-streaming-v1` and `kw10-onnx-v1`, docs/streaming-onnx-protocol.md) are in their own section below. The latency table holds the Milestone 7 CPU benchmark (protocol `kw10-latency-v1`, docs/benchmark-protocol.md); details are in its section below.

## Repository check log

Engineering checks only; these are not experiment results. Each row is a command that was actually executed, with its observed outcome.

Environment: Windows 11 (`Windows-11-10.0.26200-SP0`), CPython 3.14.2, uv 0.12.13, torch 2.14.0+cpu, numpy 2.5.3, pydantic 2.13.5, ruff 0.16.9, mypy 2.3.1, pytest 9.1.1.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 1 | `uv sync` | Exit 0; environment synced from `uv.lock` (torch 2.14.0+cpu installed) |
| 2026-09-27 | 1 | `uv lock --check` | Exit 0; lockfile consistent with `pyproject.toml` |
| 2026-09-27 | 1 | `uv run ruff format --check .` | 18 files already formatted |
| 2026-09-27 | 1 | `uv run ruff check .` | All checks passed |
| 2026-09-27 | 1 | `uv run mypy` | No issues found in 10 source files (strict mode) |
| 2026-09-27 | 1 | `uv run pytest` | 59 passed |
| 2026-09-27 | 1 | `uv run python scripts/check.py` | All four checks passed; exit 0 |

Observation (superseded by the review-fix rows below): with the Milestone 1 code, run metadata in this checkout recorded `git: null`, because git refuses the repository with a "dubious ownership" error (the directory is owned by a different Windows user).

### Review fixes (REVIEW.md P1 + provenance tests)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 1 | `uv run python scripts/check.py` | Format: 20 files already formatted; lint: all checks passed; mypy: no issues in 11 source files; pytest: 72 passed; exit 0 |
| 2026-09-27 | 1 | `start_run(load_config("configs/example.yaml"), root=<scratch dir>)` in this checkout via `uv run python` | Raised `ProvenanceError` with git's "detected dubious ownership" diagnostic; exit 1; no run directory created |
| 2026-09-27 | 1 | Same call with process-scoped `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=<repo path>` | Succeeded; recorded `source.commit` = `2535fc33630a9616cbb5408b9a6476360b8dd0ba` (HEAD at the time) and `dirty: true` (uncommitted review-fix changes present); `seeding` recorded seed 1234, `torch_deterministic_algorithms: true`, `cudnn_deterministic: true`, `cudnn_benchmark: false`, `PYTHONHASHSEED: "1234"` |

## Experiment results

| Run ID | Dataset/version | Split manifest | Model/config | Seed | Split | Accuracy | Macro F1 | ECE | NLL | Artifact link | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` | Speech Commands v0.02, 10 labels | `f811a20d…ab6131c` | `kws_cnn_v1` checkpoint `best.pt` (epoch 18, `d21f8928…d267e`), `configs/kw10_cnn_baseline_eval.yaml` | 1234 | validation (3703) | 0.9614 | 0.9614 | 0.0125 | 0.1380 | `runs/20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a/report.json` | raw probabilities (T = 1) |
| `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` | Speech Commands v0.02, 10 labels | `f811a20d…ab6131c` | `kws_cnn_v1` checkpoint `best.pt` (epoch 18, `d21f8928…d267e`), `configs/kw10_cnn_baseline_eval.yaml` | 1234 | validation (3703) | 0.9614 | 0.9614 | 0.0075 | 0.1282 | `runs/20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a/report.json` | temperature T = 1.3921; in-sample (T fitted on this split) |
| `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` | Speech Commands v0.02, 10 labels | `f811a20d…ab6131c` | `kws_cnn_v1` checkpoint `best.pt` (epoch 18, `d21f8928…d267e`), `configs/kw10_cnn_baseline_eval.yaml` | 1234 | test (4074) | 0.9639 | 0.9639 | 0.0082 | 0.1011 | `runs/20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a/report.json` | raw probabilities (T = 1) |
| `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` | Speech Commands v0.02, 10 labels | `f811a20d…ab6131c` | `kws_cnn_v1` checkpoint `best.pt` (epoch 18, `d21f8928…d267e`), `configs/kw10_cnn_baseline_eval.yaml` | 1234 | test (4074) | 0.9639 | 0.9639 | 0.0081 | 0.1019 | `runs/20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a/report.json` | temperature T = 1.3921; held-out |

## Robustness results

| Run ID | Split | Corruption | Severity | Seed | Accuracy | Accuracy degradation | Macro F1 | Macro F1 degradation | Protocol/artifact link | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | noise v1 | snr_20db | 1729 (per-example derived) | 0.9126 | +0.0513 | 0.9146 | +0.0493 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/noise/snr_20db/` | achieved SNR 20.00 dB |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | noise v1 | snr_10db | 1729 (per-example derived) | 0.8540 | +0.1100 | 0.8596 | +0.1043 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/noise/snr_10db/` | achieved SNR 10.00 dB |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | noise v1 | snr_0db | 1729 (per-example derived) | 0.6679 | +0.2960 | 0.6604 | +0.3035 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/noise/snr_0db/` | achieved SNR 0.00 dB |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | reverb v1 | rt60_0.3s_drr_+6db | 1729 (per-example derived) | 0.9593 | +0.0047 | 0.9593 | +0.0046 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/reverb/rt60_0.3s_drr_+6db/` | DRR +6.0 dB, RIR 4800 samples |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | reverb v1 | rt60_0.6s_drr_0db | 1729 (per-example derived) | 0.9141 | +0.0498 | 0.9149 | +0.0489 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/reverb/rt60_0.6s_drr_0db/` | DRR +0.0 dB, RIR 9600 samples |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | reverb v1 | rt60_1.0s_drr_-6db | 1729 (per-example derived) | 0.8338 | +0.1301 | 0.8365 | +0.1274 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/reverb/rt60_1.0s_drr_-6db/` | DRR -6.0 dB, RIR 16000 samples |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | resample v1 | via_8000hz | 1729 (per-example derived) | 0.9497 | +0.0142 | 0.9497 | +0.0141 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/resample/via_8000hz/` | FIR 65 taps |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | resample v1 | via_4000hz | 1729 (per-example derived) | 0.8505 | +0.1134 | 0.8502 | +0.1137 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/resample/via_4000hz/` | FIR 129 taps |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | resample v1 | via_2000hz | 1729 (per-example derived) | 0.4639 | +0.5000 | 0.4270 | +0.5369 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/resample/via_2000hz/` | FIR 257 taps |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | clipping v1 | at_0.5_peak | 1729 (per-example derived) | 0.9470 | +0.0169 | 0.9470 | +0.0169 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/clipping/at_0.5_peak/` | mean clipped fraction 0.0185 |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | clipping v1 | at_0.25_peak | 1729 (per-example derived) | 0.9055 | +0.0584 | 0.9058 | +0.0581 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/clipping/at_0.25_peak/` | mean clipped fraction 0.0686 |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | clipping v1 | at_0.1_peak | 1729 (per-example derived) | 0.8152 | +0.1487 | 0.8124 | +0.1515 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/clipping/at_0.1_peak/` | mean clipped fraction 0.1631 |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | codec v1 | opus_level_0.9 | 1729 (per-example derived) | 0.9637 | +0.0002 | 0.9636 | +0.0003 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/codec/opus_level_0.9/` | mean 35.2 kbit/s (range 18.8-53.3) |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | codec v1 | opus_level_0.95 | 1729 (per-example derived) | 0.9659 | -0.0020 | 0.9659 | -0.0020 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/codec/opus_level_0.95/` | mean 24.3 kbit/s (range 15.7-39.6) |
| `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a` | test (4074) | codec v1 | opus_level_1.0 | 1729 (per-example derived) | 0.9354 | +0.0285 | 0.9356 | +0.0282 | `kw10-robustness-v1`; `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/conditions/codec/opus_level_1.0/` | mean 12.7 kbit/s (range 11.2-26.6) |

## Latency measurements

| Run ID | Model/runtime | CPU/hardware | OS | Threads | Window/hop | Warm-up | Timed iterations | Batch size | p50 ms | p95 ms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | ONNX export `e1db0406…9e8b` (from `d21f8928…d267e`); ONNX Runtime 1.30.0 CPUExecutionProvider | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 1 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 2.430 | 2.707 | **Primary.** ONNX streaming step: push → window → inference (feature extraction included) → softmax → aggregate; nearest rank; local, uncontrolled machine; no real-time claim |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | ONNX export `e1db0406…9e8b` (from `d21f8928…d267e`); ONNX Runtime 1.30.0 CPUExecutionProvider | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 1 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 2.314 | 2.523 | `session.run` alone (feature extraction included, no streaming bookkeeping); nearest rank; local, uncontrolled machine; no real-time claim |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | PyTorch 2.14.0+cpu eager reference (checkpoint `d21f8928…d267e`); not the export | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 1 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 1.953 | 2.441 | PyTorch eager streaming step; reference only; nearest rank; local, uncontrolled machine; no real-time claim |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | ONNX export `e1db0406…9e8b` (from `d21f8928…d267e`); ONNX Runtime 1.30.0 CPUExecutionProvider | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 4 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 2.515 | 3.056 | ONNX streaming step; nearest rank; local, uncontrolled machine; no real-time claim |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | ONNX export `e1db0406…9e8b` (from `d21f8928…d267e`); ONNX Runtime 1.30.0 CPUExecutionProvider | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 4 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 2.419 | 2.958 | `session.run` alone; nearest rank; local, uncontrolled machine; no real-time claim |
| `20260928T004340Z_kw10-onnx-latency_1df4024f` | PyTorch 2.14.0+cpu eager reference (checkpoint `d21f8928…d267e`); not the export | AMD Ryzen 7 5800H, 8 physical / 16 logical cores, nominal 3,194 MHz | Windows 11 (10.0.26200), CPython 3.14.2 | 4 intra-op (1 inter-op) | 1 s / 100 ms (16,000 / 1,600 samples) | 500 per block | 5,000 × 3 repetitions (15,000 pooled) | 1 | 1.410 | 1.896 | PyTorch eager streaming step; reference only; nearest rank; local, uncontrolled machine; no real-time claim |

### Provenance source lock and distribution check

Added tool version: hatchling 1.32.4 (locked, dev group). uv 0.12.13 and CPython 3.14.2 as above.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 1 | `uv lock` then `uv sync` | Lock updated with hatchling 1.32.4 and its dependencies; environment synced |
| 2026-09-27 | 1 | `uv lock --check` | Exit 0; 40 packages resolved |
| 2026-09-27 | 1 | `uv run python scripts/check_dist.py` with `src/robust_streaming_ml/py.typed` temporarily moved away (negative control) | Wheel built; failed with "wheel is missing files: ['robust_streaming_ml/py.typed']"; exit 1; file restored afterwards |
| 2026-09-27 | 1 | `uv run python scripts/check.py` | Format: 21 files already formatted; lint: all checks passed; mypy: no issues in 12 source files; pytest: 78 passed; dist: wheel built offline (8 files), installed into a temporary environment, imported from that environment's site-packages, 4 import smoke tests passed; exit 0 |
| 2026-09-27 | 1 | `start_run(load_config("configs/example.yaml"), root=<scratch dir>)` in this checkout via `uv run python` | Raised `ProvenanceError` with git's "detected dubious ownership" diagnostic; exit 1 |
| 2026-09-27 | 1 | Same call with process-scoped `safe.directory` (`GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_0`/`GIT_CONFIG_VALUE_0`) | Succeeded; recorded `source.commit` = `2fe501c9e9387445cbf3023150fb3c4e6fe15785` (HEAD at the time) and `dirty: true` (these changes were uncommitted) |

### CI workflow (local rehearsal only)

The GitHub Actions workflow itself has not run yet; these rows are local runs on Windows only.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 1 | In a fresh `git clone` of commit `efd4a06` with `UV_PYTHON=3.10`: `uv sync --locked` | Exit 0; CPython 3.10.21 environment created from `uv.lock` |
| 2026-09-27 | 1 | Same clone: `uv run python scripts/check.py` | Failed at the types step: mypy reported 3 `type-arg` errors in `tests/test_seed.py` for a bare `np.ndarray` (numpy 2.2.6 is locked for Python 3.10); format, lint, tests (78 passed), and dist passed |
| 2026-09-27 | 1 | Same clone with the annotation fix (`npt.NDArray[np.float64]`): `uv run python scripts/check.py` | All five checks passed on CPython 3.10.21: mypy no issues in 12 files, 78 tests passed, distribution check passed offline; exit 0 |
| 2026-09-27 | 1 | `uv run python scripts/check.py` (main checkout, CPython 3.14.2) | All five checks passed; 78 tests passed; exit 0 |

## Dataset verification log

Data verification only; these are not experiment results. Details and all values: `docs/datasets/speech_commands_v0.02.md`.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 2 | `curl -sSI https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` | Failed: certificate check `SEC_E_WRONG_PRINCIPAL` |
| 2026-09-27 | 2 | `curl -sSI http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` and the same object at `https://storage.googleapis.com/download.tensorflow.org/...` | Both HTTP 200 with identical ETag, generation, size (2428923189), and MD5 header |
| 2026-09-27 | 2 | `curl -fsSL --retry 3 -o data/raw/speech_commands_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz` | Exit 0; 2428923189 bytes; completed 2026-09-27T18:45:33Z |
| 2026-09-27 | 2 | Python `hashlib` over the archive | SHA-256 `af14739e…68ff58`; MD5 matches the server's `x-goog-hash` and ETag |
| 2026-09-27 | 2 | Python `tarfile` listing, plus extraction of `LICENSE`, `README.md`, and both split lists | 105,829 word `.wav` files (matches README); lists clean and disjoint; 10-keyword subset 30769/3703/4074 train/validation/test; 0 speakers crossing splits; `LICENSE` is CC BY 4.0 |

### Acquisition implementation

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 2 | `uv run python scripts/check.py` (first run with the acquisition code) | Failed at dist: the wheel was missing `robust_streaming_ml/data/*.py`, because the unanchored `data/` pattern in `.gitignore` also excluded the package folder. Fixed by anchoring it to `/data/`. |
| 2026-09-27 | 2 | `uv run python scripts/check.py` (after the fix) | Format: 28 files; lint passed; mypy: no issues in 16 files; pytest: 99 passed (21 acquisition tests on synthetic archives); dist: 10-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-27 | 2 | `uv run python scripts/acquire_speech_commands.py` (archive already downloaded) | Exit 0 in 2m45s; archive re-verified against the pinned SHA-256; 105,841 files extracted to `data/raw/speech_commands_v0.02/`; pinned `LICENSE`, `README.md`, and split-list hashes matched |
| 2026-09-27 | 2 | Same command, re-run | Exit 0 in 2.7s; verified archive and completed extraction reused |
| 2026-09-27 | 2 | `git status --short --ignored` | `data/` ignored; no dataset files staged or tracked |

### Validation and manifest

Data validation only; these are not experiment results. Details: `docs/datasets/speech_commands_v0.02.md` and `docs/data-contract.md`.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-27 | 2 | Read-only header/filename survey script over the 38,546 files in the ten label folders (before writing the validator) | All files: single 16-byte PCM `fmt ` chunk (mono, 16000 Hz, 16-bit) plus `data`; RIFF size matches; no trailing bytes; 8-hex-character speaker IDs; canonical indices; no stray entries. Took 223 s single-threaded (cold). |
| 2026-09-27 | 2 | `uv run python scripts/check.py` at commit `67e6570` (integration test included because the local extraction is present) | Format: 38 files; lint passed; mypy: no issues in 26 files; pytest: 222 passed; dist: 14-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-27 | 2 | `uv run pytest -m integration -v` | `test_local_extraction_satisfies_the_contract` PASSED (1 passed, 221 deselected, 38 s) |
| 2026-09-27 | 2 | `RSML_DATA_DIR=<empty dir> uv run pytest tests/test_speech_commands.py -rs` | 2 passed, 1 skipped ("local Speech Commands extraction/archive not present") |
| 2026-09-27 | 2 | `uv run python scripts/build_speech_commands_manifest.py` (git refuses this checkout: dubious ownership) | Exit 1: `ProvenanceError` before validation; nothing written under `data/manifests` or `data/reports` |
| 2026-09-27 | 2 | Same command, with process-scoped `safe.directory` (`GIT_CONFIG_COUNT`/`KEY_0`/`VALUE_0`), at clean commit `67e6570` | Exit 0 in 39.6 s: passed, 0 errors; 38,546 rows (train 30,769 · validation 3,703 · test 4,074); manifest SHA-256 `f811a20d…ab6131c` equals the pin; warnings: 1,477 `duplicate_audio_within_split_label`; sidecar `source` = `67e6570…`, `dirty: false` |
| 2026-09-27 | 2 | Same command again, then `cmp` and `diff` against the first run's outputs | Exit 0 in 28.4 s; CSV, JSON report, and Markdown report byte-identical; sidecar identical except `created_at_utc` |

## Training runs (Milestone 3)

These are training records, not evaluation results. The validation loss and accuracy below were measured during training, on the validation split, and used only to select the checkpoint. The held-out test split was not loaded. The experiment-results table above stays empty until the Milestone 4 evaluation.

| Run ID | Commit | Config | Seed | Manifest SHA-256 | Epochs | Best epoch (by validation loss) | Validation loss / accuracy at best | Last epoch validation loss / accuracy | Train loss / accuracy, last epoch | Parameters | Wall time | Hardware / threads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260927T210911Z_kw10-cnn-baseline_6c9d6cdf` | `5babe3f` (clean) | `configs/kw10_cnn_baseline.yaml` (config SHA-256 `ad8242da…a582`) | 1234 (deterministic) | `f811a20d…ab6131c` | 30 | 18 | 0.13797 / 0.96138 (3,703 examples) | 0.13908 / 0.96327 | 0.01869 / 0.99649 (30,769 examples) | 61,052 | 32 min 24 s (audio load 22.7 s; epochs 51.5–106.5 s) | AMD Ryzen 7 5800H, Windows 11, CPython 3.14.2, torch 2.14.0+cpu, 8 threads |

- Checkpoints are in the git-ignored run directory `runs/20260927T210911Z_kw10-cnn-baseline_6c9d6cdf/checkpoints/`: `best.pt` (epoch 18, SHA-256 `d21f8928…d267e`, 259,243 bytes) and `last.pt` (epoch 30, SHA-256 `06037b23…cafd2`).
- `summary.json` records rows loaded as train 30,769, validation 3,703, and test 0.
- The best checkpoint improved at epochs 1, 2, 3, 5, 6, 8, 12, 13, and 18. After epoch 18, validation loss stayed between 0.1386 and 0.157 while train loss fell from 0.0440 to 0.0187.
- Feature fingerprint: `8368ac49…00ba1`.

### Commands (Milestone 3)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 3 | `uv run python scripts/check.py` on the tree committed as `5babe3f` (Milestone 2 integration test included) | Format: 54 files; lint passed; mypy: no issues in 41 files; pytest: 329 passed; dist: 21-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-28 | 3 | `uv run python scripts/train_baseline.py --config configs/kw10_cnn_baseline.yaml` (git refuses this checkout) | Exit 1: `ProvenanceError` before `start_run`; no `runs/` directory created |
| 2026-09-28 | 3 | Same command with process-scoped `safe.directory` (`GIT_CONFIG_COUNT`/`KEY_0`/`VALUE_0`) at clean commit `5babe3f` | Exit 0 after 32 min 24 s; run `20260927T210911Z_kw10-cnn-baseline_6c9d6cdf` as in the table above |
| 2026-09-28 | 3 | `load_checkpoint(best.pt, expect_manifest_sha256=…, expect_labels=…, expect_window=…, expect_features=…)` via `uv run python` | Loaded: epoch 18, source `5babe3f` (clean), splits used train and validation, SHA-256 matched the sidecar |

## Evaluation (Milestone 4)

**Protocol** `kw10-clean-eval-v1` ([docs/evaluation-protocol.md](docs/evaluation-protocol.md)) was committed at `a3d208f` before any evaluation ran.

**Run** `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a`:
- clean commit `a3d208f`, seed 1234, deterministic, 8 threads, batch size 256;
- checkpoint `runs/20260927T210911Z_kw10-cnn-baseline_6c9d6cdf/checkpoints/best.pt` (SHA-256 `d21f89283a128820e93b00d6e104805b3423fd47f2ed8791ba7775d5bf9d267e`), from training run `20260927T210911Z_kw10-cnn-baseline_6c9d6cdf`, epoch 18, trained at clean commit `5babe3f`;
- manifest `f811a20d…ab6131c`; validation 3,703 rows, test 4,074 rows (train not loaded).

**Calibration:** temperature scaling fitted on the **validation predictions only** (3,703 rows):
- `T = 1.3921393146428631` (`β = 1/T = 0.7183189135467654`), not at a search bound;
- validation NLL 0.1380 → 0.1282 (in-sample);
- the fit input is `predictions/validation.csv`, SHA-256 `f2bf21f1…dac378`. `calibration.json` was written before any test inference.

**Official results** (all rows, duplicates retained) are in the experiment results table above. The official test evaluation ran once for this protocol and checkpoint. One further reproducibility re-run (identical outputs, below) was not used for anything, and nothing was selected or changed after seeing test results.

**Reading the results:**
- **Raw vs calibrated:** temperature scaling cannot change the argmax, so accuracy, macro F1, and the confusion matrix are identical raw and calibrated.
  - On **test**, calibration lowered ECE slightly (0.0082 → 0.0081) and raised NLL slightly (0.1011 → 0.1019). **The in-sample validation improvement did not carry over to test NLL.**
  - Both are reported; the raw result is primary.
- **Test errors:** 147 of 4,074 (accuracy 0.9639).
- **Largest test confusions** (true → predicted): go → no: 15; on → off: 13; down → no: 11; off → up: 10; up → off: 9.
- **Test per-class F1** (raw), in label order (yes, no, up, down, left, right, on, off, stop, go): 0.9786 | 0.9591 | 0.9548 | 0.9471 | 0.9674 | 0.9898 | 0.9614 | 0.9446 | 0.9902 | 0.9456.
- **Consistency check:** raw validation accuracy equals Milestone 3's training-time validation accuracy for this checkpoint exactly (0.9613826627059141). Validation NLL agrees to about 1e-8 (0.1379662737 vs 0.1379662621); training computed it in float32, evaluation in float64.

**Secondary, duplicate-content sensitivity** (one row per `audio_sha256`; not the official result):

| Split | Probabilities | Rows kept | Accuracy | Macro F1 | NLL | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | raw | 3692 (11 removed) | 0.9613 | 0.9613 | 0.1384 | 0.0125 |
| validation | calibrated | 3692 (11 removed) | 0.9613 | 0.9613 | 0.1285 | 0.0076 |
| test | raw | 3953 (121 removed) | 0.9646 | 0.9646 | 0.0984 | 0.0075 |
| test | calibrated | 3953 (121 removed) | 0.9646 | 0.9646 | 0.0998 | 0.0083 |

**Limitations:**
- **Duplicates:** the official numbers count 121 redundant test copies (119 groups, 240 files) and 11 validation copies (11 groups, 22 files); every group is one speaker and one label. The sensitivity view above shows the effect.
- **Near-duplicates** are not detected.
- **Closed set:** a 10-label classifier on clean 1 s clips; no silence or unknown speech. These results say nothing about rejecting other speech, robustness, or streaming.
- **Single run:** one model trained with one seed; no confidence intervals.
- **ECE** uses 15 equal-width bins. Most test predictions fall in the top bin (3,718 of 4,074 have confidence ≥ 14/15).
- **Artifacts** are under the git-ignored `runs/20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a/`: `predictions/{validation,test}.csv` (logits and raw probabilities), `predictions/*_calibrated.csv`, `calibration.json`, `report.json` (full precision, per-class metrics, confusion matrices, ECE bins, artifact SHA-256), `report.md`, and `summary.json`.

### Commands (Milestone 4)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 4 | `uv run python scripts/check.py` on the tree committed as `a3d208f` (Milestone 2 integration test included) | Format: 66 files; lint passed; mypy: no issues in 52 files; pytest: 382 passed; dist: 27-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-28 | 4 | `uv run python scripts/evaluate_baseline.py --config configs/kw10_cnn_baseline_eval.yaml` (git refuses this checkout) | Exit 1: `ProvenanceError` during the checkpoint-provenance gate; no run directory created |
| 2026-09-28 | 4 | Same command with process-scoped `safe.directory` at clean commit `a3d208f` | Exit 0 in 18.7 s; run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a`; values as above |
| 2026-09-28 | 4 | Independent check script: re-verify every artifact SHA-256, reload `calibration.json` with its compatibility checks, and regenerate all official and duplicate-sensitivity metrics from the prediction files | All hashes matched; every aggregate regenerated identically |
| 2026-09-28 | 4 | Reproducibility re-run of the same protocol through `evaluate()` into a scratch directory outside `runs/` (not a result) | Prediction CSVs, calibrated CSVs, and `calibration.json` byte-identical; metrics, duplicate sensitivity, and calibration identical |

## Robustness (Milestone 5)

**Protocol** `kw10-robustness-v1` ([docs/robustness-protocol.md](docs/robustness-protocol.md)) and its code were committed at `e024ee8` before any real robustness run.

**Run** `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a`:
- clean commit `e024ee8`, 8 threads, batch size 256, 323 s;
- checkpoint `d21f8928…d267e` (the Milestone 4 checkpoint, unchanged; no retraining, tuning, or recalibration);
- official test split, 4,074 rows, duplicates retained;
- raw probabilities (`T = 1`).

**Clean baseline:** the Milestone 4 official raw test metrics, from run `20260927T220409Z_kw10-cnn-baseline-eval_5b44b01a` (report SHA-256 `2f9a0f5bdbbdb89cf1053c604e821ccd139950a73d6fe07ab4e059872d56b237`). Accuracy 0.9639, macro F1 0.9639, NLL 0.1011, ECE 0.0082.
- **Clean-path check:** the robustness inference path reproduced all 4,074 Milestone 4 test logits bit-for-bit before any corruption was applied.

**Codec backend** (pinned and checked): soundfile/libsndfile, `soundfile` 0.14.0, libsndfile 1.2.2, Ogg Opus. NumPy 2.5.3 provides the per-example random generators.

**Accuracy and macro F1 degradation** are in the robustness results table above (degradation = clean − corrupted; positive means worse).

**NLL, ECE, and the largest confusions** (degradation = clean − corrupted; for NLL and ECE, negative means worse):

| Corruption | Severity | NLL | Δ NLL | ECE | Δ ECE | Largest confusions (true → predicted, count) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| noise | snr_20db | 0.2710 | -0.1699 | 0.0317 | -0.0235 | up → off 118; on → off 38; left → yes 23 |
| noise | snr_10db | 0.4245 | -0.3234 | 0.0413 | -0.0331 | up → off 160; left → yes 76; on → off 53 |
| noise | snr_0db | 0.9863 | -0.8852 | 0.0518 | -0.0435 | left → yes 197; up → off 131; go → off 95 |
| reverb | rt60_0.3s_drr_+6db | 0.1311 | -0.0301 | 0.0127 | -0.0044 | down → no 18; off → up 16; go → no 16 |
| reverb | rt60_0.6s_drr_0db | 0.2477 | -0.1467 | 0.0133 | -0.0051 | up → off 44; on → off 41; left → yes 23 |
| reverb | rt60_1.0s_drr_-6db | 0.5017 | -0.4006 | 0.0354 | -0.0272 | up → off 102; on → off 102; right → left 44 |
| resample | via_8000hz | 0.1471 | -0.0460 | 0.0149 | -0.0067 | off → up 22; stop → up 22; down → no 15 |
| resample | via_4000hz | 0.4508 | -0.3497 | 0.0483 | -0.0400 | stop → go 129; yes → left 60; down → go 58 |
| resample | via_2000hz | 2.0112 | -1.9102 | 0.3077 | -0.2994 | up → off 306; right → go 216; stop → off 190 |
| clipping | at_0.5_peak | 0.1588 | -0.0577 | 0.0172 | -0.0090 | off → up 25; go → no 24; down → no 19 |
| clipping | at_0.25_peak | 0.3096 | -0.2085 | 0.0301 | -0.0219 | go → no 31; off → up 30; down → no 26 |
| clipping | at_0.1_peak | 0.6271 | -0.5261 | 0.0693 | -0.0611 | right → no 42; go → no 39; right → left 37 |
| codec | opus_level_0.9 | 0.1021 | -0.0010 | 0.0094 | -0.0012 | go → no 16; off → up 14; down → no 11 |
| codec | opus_level_0.95 | 0.1026 | -0.0015 | 0.0081 | +0.0001 | off → up 17; go → no 14; on → off 12 |
| codec | opus_level_1.0 | 0.1979 | -0.0968 | 0.0140 | -0.0058 | up → off 19; off → up 18; on → off 16 |

**Reading the results:**
- **Largest accuracy degradation:** resample/via_2000hz +0.5000; noise/snr_0db +0.2960; clipping/at_0.1_peak +0.1487; reverb/rt60_1.0s_drr_-6db +0.1301; resample/via_4000hz +0.1134.
- **Resampling through 2 kHz** (1 kHz bandwidth) is the most damaging condition. Predictions collapse toward `go` (1309 predictions) and `off` (998).
- **Opus** at `compression_level` 0.9 and 0.95 (mean about 35 and 24 kbit/s) leaves accuracy essentially unchanged. At 0.95 the corrupted accuracy is **higher** than clean: 3,935 vs 3,927 correct of 4,074, a negative degradation of -0.0020. This is reported as measured. With one corruption draw and no confidence intervals, it is not claimed as an improvement. At 0.9: 3,926 correct. Level 1.0 (about 12.7 kbit/s) costs 0.0285 accuracy (3,811 correct).
- **ECE is worse than clean** in 14 of 15 conditions.
- **Recurring confusions:** `up → off` is the largest confusion at 20 dB and 10 dB SNR (118, 160) and ties for largest under the most severe reverb (102, with `on → off`). `left → yes` grows with noise (23, 76, 197) and becomes the largest at 0 dB SNR.

**Secondary: duplicate sensitivity** (one row per `audio_sha256`, 3,953 rows; degradation against the Milestone 4 dedup clean raw metrics; not the official result):

| Corruption | Severity | Rows | Accuracy | Δ acc | Macro F1 | Δ F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| noise | snr_20db | 3953 | 0.9122 | +0.0524 | 0.9144 | +0.0503 |
| noise | snr_10db | 3953 | 0.8548 | +0.1098 | 0.8605 | +0.1042 |
| noise | snr_0db | 3953 | 0.6671 | +0.2975 | 0.6604 | +0.3042 |
| reverb | rt60_0.3s_drr_+6db | 3953 | 0.9598 | +0.0048 | 0.9599 | +0.0048 |
| reverb | rt60_0.6s_drr_0db | 3953 | 0.9147 | +0.0498 | 0.9158 | +0.0489 |
| reverb | rt60_1.0s_drr_-6db | 3953 | 0.8325 | +0.1321 | 0.8355 | +0.1291 |
| resample | via_8000hz | 3953 | 0.9514 | +0.0132 | 0.9515 | +0.0131 |
| resample | via_4000hz | 3953 | 0.8520 | +0.1126 | 0.8516 | +0.1131 |
| resample | via_2000hz | 3953 | 0.4640 | +0.5006 | 0.4271 | +0.5376 |
| clipping | at_0.5_peak | 3953 | 0.9484 | +0.0162 | 0.9485 | +0.0162 |
| clipping | at_0.25_peak | 3953 | 0.9074 | +0.0572 | 0.9078 | +0.0568 |
| clipping | at_0.1_peak | 3953 | 0.8181 | +0.1465 | 0.8158 | +0.1488 |
| codec | opus_level_0.9 | 3953 | 0.9643 | +0.0003 | 0.9644 | +0.0003 |
| codec | opus_level_0.95 | 3953 | 0.9664 | -0.0018 | 0.9664 | -0.0018 |
| codec | opus_level_1.0 | 3953 | 0.9368 | +0.0278 | 0.9370 | +0.0276 |

**Failures and non-runs:** none. All 15 conditions ran on all 4,074 test clips, and no clip failed validation.

**Limitations:**
- The corruptions are synthetic and procedural: white noise, an exponential-decay RIR, one resampler design, peak-relative symmetric clipping, and one codec family at three `compression_level` settings.
- One corruption draw per clip, one model, and one seed, so there are no confidence intervals. Small differences, such as the Opus 0.9 and 0.95 rows, should not be over-read.
- Codec bitrate is only indirectly controlled and varies per clip (reported above).
- Bit-exactness is guaranteed only on the same platform and library versions.
- The duplicate, near-duplicate, closed-set, and clean-trained limitations of Milestone 4 carry over.

**Artifacts** are under the git-ignored `runs/20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a/`:
- `conditions/<corruption>/<severity>/predictions.csv` and `corruption_manifest.csv` (per-clip seed, output SHA-256, and measurements);
- `report.json` (full precision, confusion matrices, per-class metrics, measurement summaries, artifact SHA-256) and `report.md`;
- `summary.json`.

### Commands (Milestone 5)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 5 | `uv add "soundfile>=0.13"` (later tightened to `>=0.14`) and `uv lock` | Added `soundfile` 0.14.0 (bundled libsndfile 1.2.2), `cffi` 2.1.1, and `pycparser` 3.0 to the lockfile |
| 2026-09-28 | 5 | Backend probe on synthetic signals only: Opus encode/decode at several `compression_level` values | Decoded 16 kHz mono with exact frame counts and lag 0. Decoded samples were identical across repeated encodes; the encoded bytes differ only in the Ogg serial number and CRC. |
| 2026-09-28 | 5 | `uv run python scripts/check.py` on the tree committed as `e024ee8` | First attempt: the dist check failed, because its isolated `uv sync` picked a cached Python 3.10 with no cached `cffi` cp310 wheel. Fixed by pinning `--python sys.executable` in `check_dist.py`. Final: format 75 files, lint passed, mypy no issues in 60 files, pytest 435 passed, dist 31-file wheel with 4 smoke tests passed; exit 0. |
| 2026-09-28 | 5 | `uv run python scripts/evaluate_robustness.py --config configs/kw10_cnn_baseline_robustness.yaml` (git refuses this checkout) | Exit 1: `ProvenanceError` at the checkpoint-provenance gate; no run directory created |
| 2026-09-28 | 5 | Same command with process-scoped `safe.directory` at clean commit `e024ee8` | Exit 0 in 5 min 29 s; run `20260927T223417Z_kw10-cnn-baseline-robustness_6cf68b9a`; 15 conditions completed |
| 2026-09-28 | 5 | Independent verification script (outside the package) | All 30 artifact hashes matched. All 4,074 source test files re-hashed unchanged. Every condition's metrics and degradation regenerated identically from its predictions file. For 25 random clips per condition, the corruption recomputed from source matched the recorded seed and output SHA-256. |

## Streaming and ONNX (Milestone 6)

**Protocols** `kw10-streaming-v1` and `kw10-onnx-v1` ([docs/streaming-onnx-protocol.md](docs/streaming-onnx-protocol.md)) were committed at `15797c9`, and their implementation at `978a012`, both before any real-data export or parity run. The tolerances were sized from synthetic probes only (commands below).

**Run** `20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b`:
- clean commit `978a012`; 203 s wall time (195 s inside the run);
- checkpoint `d21f8928…d267e` (the Milestone 4 and 5 checkpoint, unchanged; no retraining, tuning, or recalibration);
- manifest `f811a20d…ab6131c`; official test split, 4,074 rows.

**Toolchain and runtime** (pinned and checked):
- export: torch 2.14.0+cpu, onnx 1.23.0, onnxscript 0.7.2 (dynamo exporter);
- inference: ONNX Runtime 1.30.0 on `CPUExecutionProvider` only, with `ORT_ENABLE_ALL`, sequential execution, and 1 intra-op and 1 inter-op thread;
- PyTorch reference: 8 torch threads.

**Exported model:** `onnx/model.onnx`, SHA-256 `e1db0406461abb4649cd6ab36a76b2369067402a19ef2d80823825a9b0ea9e8b`, 341,090 bytes.
- **Graph:** opset 18 (default domain only), IR version 10, 31 nodes, 29 initializers. Operators: `Add`, `BatchNormalization`, `Conv`, `Gather`, `Gemm`, `Log`, `MatMul`, `MaxPool`, `Pad`, `Pow`, `ReduceMean`, `Relu`, `Reshape`, `STFT`, `Transpose`, `Unsqueeze`.
- **Contract:** `waveform` float32 `[batch, 16000]` → `logits` float32 `[batch, 10]`.
- **Determinism:** exported twice in the run with identical bytes. The independent check re-exported it from the checkpoint and also got identical bytes.
- **Metadata** records the checkpoint SHA-256, labels, feature fingerprint, manifest, source `978a012` (clean), and toolchain.

**Clean-path gate:** the PyTorch path (batch size 256) reproduced all 4,074 Milestone 4 raw test logits bit-for-bit.

**Parity** (ONNX Runtime vs PyTorch, both one window at a time; tolerances: logits 1e-2, probabilities 1e-3, aggregates 1e-3):

| Set | Streams | Windows | max \|Δ logit\| | mean \|Δ logit\| | max \|Δ probability\| | max \|Δ aggregate\| | Top-1 flips (near-tie) | Violations | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A: each test clip as a stream | 4,074 | 4,074 | 1.507e-4 | 5.57e-6 | 1.16e-5 | 1.16e-5 | 0 (0) | 0 | yes |
| B: concatenations of 8 consecutive test clips | 510 | 35,629 | 1.764e-4 | 5.19e-6 | 3.61e-5 | 1.37e-5 | 0 (0) | 0 | yes |

- The largest logit difference is about 57 times below the logit tolerance, and the largest probability difference about 28 times below the probability tolerance.
- Set B pushed its audio in 22,311 seeded chunks. It produced 177 final partial windows, and every streamed window was bit-identical to the offline reference windows. In set A, every clip yielded exactly one window, bit-identical to its Milestone 4 clip window.

**Streaming consistency** (set A, all 4,074 test clips; stream decision vs the Milestone 4 clip-level prediction):

| Path | Agreement with Milestone 4 | Accuracy (correct) |
| --- | ---: | ---: |
| ONNX Runtime streaming, batch 1 | 4,074 / 4,074 (1.0000) | 0.9639 (3,927) |
| PyTorch streaming, batch 1 | 4,074 / 4,074 (1.0000) | 0.9639 (3,927) |
| Milestone 4 clip-level, batch 256 | n/a | 0.9639 (3,927) |

- The PyTorch batch-1 logits differ from the Milestone 4 batch-256 logits by up to 9.5e-6. This is the batch-size dependence noted in the protocol. No decision changed.

**Failures and non-runs:** none. No tolerance was violated and no gate failed in the formal run.

**Limitations:**
- Parity was measured on clean test audio only. Corrupted audio is covered only by synthetic probes and tests.
- Speech Commands clips are isolated utterances of at most 1 s, so every clip-as-stream has exactly one window. That makes the 100% streaming consistency a check that the streaming path reproduces the clip protocol, not a measurement of continuous-audio behaviour. The multi-window streams (set B) are artificial concatenations with no per-window ground truth, so they are used for numerical parity only.
- The results are for one model, one runtime version (ONNX Runtime 1.30.0), and one platform: Windows 11, CPython 3.14.2, AMD Ryzen 7 5800H. The CI environment locks ONNX Runtime 1.23.2 on Python 3.10, which has only been exercised by the synthetic tests.
- Closed set: every window gets one of the 10 labels. There is no silence or unknown-word rejection.
- No latency was measured in this milestone.

**Artifacts** are under the git-ignored `runs/20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b/`:
- `onnx/model.onnx` and `onnx/model.onnx.json`;
- `parity/clips.csv` (per test clip: labels, Milestone 4 prediction, PyTorch and ONNX logits, both stream decisions);
- `parity/stream_windows.csv` (per set-B window: timestamps and both backends' logits) and `parity/streams.csv`;
- `report.json`, `report.md`, and `summary.json`.

### Commands (Milestone 6)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 6 | `uv pip compile` for `onnx`, `onnxruntime`, `onnxscript` targeting CPython 3.14 on Windows and CPython 3.10 on Linux | Resolved onnx 1.23.0, onnxscript 0.7.2, and onnxruntime 1.30.0 (Windows, Python 3.14) or 1.23.2 (Linux, Python 3.10) |
| 2026-09-28 | 6 | Export probe on synthetic signals, random weights, in an ephemeral `uv run --with …` environment (the project environment was not modified) | The legacy TorchScript exporter failed ("STFT does not currently support complex types"). The dynamo exporter exported `torch.stft` as ONNX `STFT` at opset 18. It first failed on a cp1252 console printing a ✅; that did not recur with `verbose=False`. ONNX Runtime matched PyTorch to 3.9e-7 on the logits. |
| 2026-09-28 | 6 | Tolerance-sizing probes: pinned checkpoint weights on synthetic signals only (tones, noise, bursts, band-limited, and all 15 `kw10-robustness-v1` corruptions of synthetic clips); no dataset audio | Export bytes identical across two exports. ONNX vs PyTorch max \|Δlogit\| 1.7e-3 and \|Δprob\| 2.7e-4 on pure tones; at most 1.3e-4 and 1.2e-5 on speech-like, band-limited, and corrupted signals; top-1 always agreed. PyTorch batch 1 vs 64 differed by up to 9.5e-6. These set the tolerances (1e-2, 1e-3). |
| 2026-09-28 | 6 | `uv add "onnx>=1.23" "onnxruntime>=1.23" "onnxscript>=0.7"` | Locked onnx 1.23.0, onnxscript 0.7.2, onnx-ir 1.0.0, protobuf 7.36.2, ml-dtypes 0.6.0, flatbuffers, and onnxruntime 1.30.0 (Python ≥ 3.11). The lock picked onnxruntime 1.24.3 for Python 3.10, which has **no cp310 wheels**. |
| 2026-09-28 | 6 | `[tool.uv] required-environments` (Linux and Windows x86-64), then `uv lock --upgrade-package onnxruntime` | Did not help: 1.24.3 was still selected for Python < 3.11. The setting was removed. |
| 2026-09-28 | 6 | Marker-split dependency (`onnxruntime>=1.23,<1.24; python_version < '3.11'`) and `uv lock` | onnxruntime 1.23.2 (cp310 wheels for Linux, Windows, and macOS) for Python 3.10, and 1.30.0 otherwise. A lockfile scan found a CPython 3.10 Linux x86-64 artifact for every package; the only package flagged was the macOS-only torch build. |
| 2026-09-28 | 6 | `uv run python scripts/check.py` on the tree committed as `978a012` (Milestone 2 integration test included) | Format: 88 files; lint passed; mypy: no issues in 72 files; pytest: 534 passed; dist: 39-file wheel, 4 smoke tests passed in the isolated environment; exit 0 |
| 2026-09-28 | 6 | `uv run python scripts/export_onnx.py --config configs/kw10_cnn_baseline_onnx.yaml` (git refuses this checkout) | Exit 1: `ProvenanceError` at the checkpoint-provenance gate; no run directory created |
| 2026-09-28 | 6 | Same command with process-scoped `safe.directory` at clean commit `978a012` | Exit 0 in 203 s; run `20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b`; parity passed |
| 2026-09-28 | 6 | Independent verification script (outside the package's run code) | All checks passed: <ul><li>all 7 artifact hashes matched;</li><li>the model reloaded with every expectation;</li><li>a re-export from the checkpoint was byte-identical;</li><li>every set A and set B parity statistic, aggregate, and consistency count regenerated identically from the CSVs;</li><li>window counts matched the protocol formula;</li><li>all 4,074 source test files re-hashed unchanged;</li><li>100 sampled clip windows and 100 sampled stream windows, recomputed from source audio with fresh ONNX Runtime and PyTorch backends, gave identical logits.</li></ul> |

## CPU latency benchmark (Milestone 7)

**Protocol** `kw10-latency-v1` ([docs/benchmark-protocol.md](docs/benchmark-protocol.md)) was committed at `52ea06e`, and its implementation at `95fd765`, both before any latency measurement.

**Run** `20260928T004340Z_kw10-onnx-latency_1df4024f`:
- clean commit `95fd765`; 230 s wall time;
- the protocol plan: 500 warm-up and 5,000 timed steps per block, 3 repetitions, interleaved `repetition → threads → path`, with a fresh session or classifier per block.

**Model and runtime:**
- the Milestone 6 export `e1db0406…9e8b`, from run `20260928T002359Z_kw10-cnn-baseline-onnx_7da9328b`, whose export report `7fddcac2…fa60` is pinned and records passed parity; checkpoint `d21f8928…d267e`;
- ONNX Runtime 1.30.0 on `CPUExecutionProvider`, with `ORT_ENABLE_ALL`, sequential execution, and 1 inter-op thread;
- PyTorch 2.14.0+cpu for the reference path; batch size 1; window 16,000 and hop 1,600 samples.

**Machine** (recorded by the run):
- AMD Ryzen 7 5800H with Radeon Graphics: nominal 3,194 MHz; 8 physical cores (`GetLogicalProcessorInformationEx`) and 16 logical cores;
- Windows 11 (`Windows-11-10.0.26200-SP0`); CPython 3.14.2; numpy 2.5.3;
- timer `time.perf_counter_ns` (QueryPerformanceCounter, resolution 1e-7 s);
- no CPU affinity, normal priority, garbage collector enabled; power plan and background load not controlled.

**Audio:** the first 562 official test clips in manifest order (`down/022cd682_nohash_0.wav` … `go/7bae88ed_nohash_0.wav`), concatenated into 8,814,400 samples (550.9 s). Every block sees the same 5,500 windows.

**Sanity checks (before timing):**
- ONNX streaming logits for the first 20 windows were bit-identical to ONNX Runtime on the offline windows;
- they agreed with PyTorch to max |Δlogit| 1.6e-5, within the `kw10-onnx-v1` tolerances;
- every timed step emitted exactly one window.

**Results** (per-window latency in ms, nearest-rank percentiles over 15,000 pooled timed steps; the primary configuration is the ONNX streaming step with 1 thread):

| Path | Threads | p50 | p95 | p99 | mean | min | max | Per-repetition p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **ONNX streaming step** (primary; feature extraction included) | 1 | **2.430** | **2.707** | 2.958 | 2.468 | 2.276 | 3.914 | 2.420 / 2.677; 2.434 / 2.716; 2.435 / 2.719 |
| ONNX `session.run` alone | 1 | 2.314 | 2.523 | 2.748 | 2.338 | 2.178 | 3.630 | 2.311 / 2.518; 2.317 / 2.528; 2.313 / 2.518 |
| PyTorch eager streaming step (reference) | 1 | 1.953 | 2.441 | 2.715 | 2.038 | 1.741 | 3.855 | 1.942 / 2.443; 1.965 / 2.459; 1.952 / 2.428 |
| ONNX streaming step | 4 | 2.515 | 3.056 | 3.261 | 2.568 | 2.287 | 3.896 | 2.485 / 2.972; 2.520 / 3.092; 2.541 / 3.071 |
| ONNX `session.run` alone | 4 | 2.419 | 2.958 | 3.128 | 2.473 | 2.183 | 3.696 | 2.425 / 2.982; 2.413 / 2.941; 2.420 / 2.934 |
| PyTorch eager streaming step (reference) | 4 | 1.410 | 1.896 | 2.095 | 1.474 | 1.196 | 2.563 | 1.424 / 1.908; 1.404 / 1.911; 1.405 / 1.871 |

**Reading the results:**
- **Streaming overhead:** the primary path (the whole streaming step) took a median of 2.43 ms per 100 ms hop on this machine. Its p50 is about 0.12 ms above `session.run` alone (2.430 vs 2.314 ms); that is the cost of the bookkeeping (chunk conversion, buffering, window copy, softmax, aggregation).
- **PyTorch was faster than the ONNX export** in both thread settings, as measured: p50 1.95 vs 2.43 ms with 1 thread, and 1.41 vs 2.52 ms with 4 threads. The comparison is between two local paths; it is not a statement about optimised deployments of either.
- **Threads:** 4 intra-op threads did not reduce ONNX Runtime latency (p50 2.43 → 2.52 ms); they did reduce PyTorch latency (1.95 → 1.41 ms).
- **Spread:** the three repetitions agree within 0.06 ms at p50 and 0.12 ms at p95 for every configuration. The widest spreads are both from the 4-thread ONNX streaming step.
- **No real-time or production claim** is made from these numbers.

**Diagnostic, not part of `kw10-latency-v1`:** an operator profile on synthetic windows, 1 thread, looked at where the ONNX time goes.
- ONNX Runtime's `STFT` kernel accounted for 81% of kernel time (about 2.0 ms per window). The exported CNN's `Conv` kernels took about 0.25 ms.
- In PyTorch, the FFT (`aten::_fft_r2c`) took about 0.06 ms, and pooling and convolution dominated.

So the export's in-graph STFT is the bottleneck on this runtime. Expressing the front end differently, or computing features outside ONNX Runtime, might reduce it. That would need a new protocol version and a new parity run, and **was not done**.

**Recording issue (disclosed):** `report.json` field `environment.torch_threads.intra_op_default` reads 4. The environment was captured *after* the timing blocks, when the last PyTorch block had set 4 threads. The process default at run start was 8, as recorded by `start_run` in `metadata.json`. The per-configuration thread counts used for timing are recorded correctly in `timings.csv` and in the result keys. The timings are unaffected. Commit `1d1f800` fixed the code afterwards to capture the environment before timing; this run was not re-run or edited.

**Limitations:**
- One laptop, with uncontrolled power, thermal, and background conditions; other machines and runs will differ.
- Single stream at batch 1, including Python-level overhead.
- Audio I/O, decoding, and model loading are excluded.
- The PyTorch path is an unoptimised eager reference.
- One ONNX Runtime version.
- This is not a production service-level agreement and not a real-time claim.

**Artifacts** are under the git-ignored `runs/20260928T004340Z_kw10-onnx-latency_1df4024f/`:
- `timings.csv`: all 90,000 timed steps, raw nanoseconds;
- `report.json`: statistics, environment, sanity checks, and plan;
- `report.md` and `summary.json`.

### Commands (Milestone 7 benchmark)

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 7 | `uv run python scripts/check.py` on the tree committed as `95fd765` (Milestone 2 integration test included) | Format: 97 files; lint passed; mypy: no issues in 80 files; pytest: 555 passed; dist: 44-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-28 | 7 | `uv run python scripts/benchmark_latency.py --config configs/kw10_onnx_benchmark.yaml` (git refuses this checkout) | Exit 1: `ProvenanceError` at the checkpoint-provenance gate; no run directory created |
| 2026-09-28 | 7 | Same command with process-scoped `safe.directory` at clean commit `95fd765` | Exit 0 in 230 s; run `20260928T004340Z_kw10-onnx-latency_1df4024f`; values as above |
| 2026-09-28 | 7 | Independent verification script (outside the package's run code) | All checks passed: <ul><li>artifact hashes matched;</li><li>the export pins matched the Milestone 6 files;</li><li>the plan equals the protocol;</li><li>90,000 rows, in the interleaved block order, each block holding windows 500–5,499;</li><li>every pooled and per-repetition statistic regenerated identically with an independent nearest-rank implementation.</li></ul> |
| 2026-09-28 | 7 | Diagnostic operator profile (ONNX Runtime profiling and `torch.profiler`), synthetic windows, 1 thread; not a protocol measurement | ONNX Runtime `STFT` 81.1% of kernel time (about 2.0 ms/run), `Conv` 9.9%. PyTorch `aten::_fft_r2c` about 0.06 ms/run; max-pool and convolution dominate PyTorch self time. |

## Demo and final verification (Milestone 7)

**Demo runs:** these are demo outputs, not results. The demo (`scripts/demo.py`, [docs/demo.md](docs/demo.md)) was committed at `f044b20`. It runs the Milestone 6 export `e1db0406…9e8b`, pinned through its export report `7fddcac2…fa60`, with ONNX Runtime 1.30.0 on the CPU provider and 1 thread. It writes a self-contained HTML page and JSON under the git-ignored `artifacts/demo/`. The runs below executed at clean commit `f044b20` and are described in the walkthrough in `docs/demo.md`.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 7 | `uv run python scripts/check.py` on the tree committed as `f044b20` (Milestone 2 integration test included) | Format: 104 files; lint passed; mypy: no issues in 86 files; pytest: 568 passed; dist: 48-file wheel, 4 smoke tests passed; exit 0 |
| 2026-09-28 | 7 | `uv run python scripts/demo.py --audio data/raw/speech_commands_v0.02/yes/022cd682_nohash_0.wav --corruption noise --severity snr_0db --output artifacts/demo/walkthrough_yes_noise_snr_0db.html` | Exit 0: one window. Clean decision `yes` (1.000); with noise at an achieved SNR of 0.00 dB, `yes` (0.999). HTML and JSON written; source `f044b20`, clean. |
| 2026-09-28 | 7 | `uv run python scripts/demo.py --audio artifacts/demo/left_stop_go.wav --corruption resample --severity via_2000hz --output artifacts/demo/walkthrough_left_stop_go_resample_2000hz.html` | Exit 0. The input is three test clips (`left`, `stop`, `go`, speaker `022cd682`) concatenated into 48,000 samples by a scratch script. <ul><li>21 windows.</li><li>Clean window labels: `left` for windows 0–6, `stop` 7–15, `up` 16–18, `go` 19–20.</li><li>Clean stream decision `up` (0.502), the mean of the last 5 windows.</li><li>Resampled via 2 kHz: 0 of 21 window labels unchanged, and stream decision `off` (0.301).</li></ul> |
| 2026-09-28 | 7 | Structural check of a demo page with Python's `html.parser` | Tags balanced; no `<script>`; no `src`/`href` other than `data:` URIs; 2 SVG timelines. The page was not inspected visually: the browser pane cannot run page tools on `file://` pages, and a local preview server was not used. |

**CPython 3.10 CI rehearsal:** local, on Windows; the GitHub Actions workflow itself has still not run.

| Date | Milestone | Command | Outcome |
| --- | --- | --- | --- |
| 2026-09-28 | 7 | `git clone` of this checkout into a scratch directory, with process-scoped `safe.directory` | First attempt: exit 128, "dubious ownership" for the source `.git` path. It succeeded after adding that path to the process-scoped override. |
| 2026-09-28 | 7 | Fresh clone of `f044b20`: `UV_PYTHON=3.10 uv sync --locked`, then `UV_PYTHON=3.10 uv run python scripts/check.py` | Sync exit 0: CPython 3.10.21, onnxruntime 1.23.2, torch 2.14.0+cpu, onnx 1.23.0, onnxscript 0.7.2, numpy 2.2.6. **Check exit 1:** mypy found 10 errors in 6 files under numpy 2.2 stubs, several in code from Milestones 2–5, so CI type checking would have failed since then. Format, lint, pytest (567 passed, 1 skipped), and dist passed. |
| 2026-09-28 | 7 | Typing fix (commit `b106a2b`): explicit `NDArray` annotations, and `np.asarray(..., float64)` on already-float64 arrays in the resample corruption | mypy clean under numpy 2.5.3 and 2.2.6. `uv run python scripts/check.py` (CPython 3.14.2): 104 files formatted, lint passed, mypy no issues in 86 files, 568 passed, 48-file wheel; exit 0. 90 sampled `resample` outputs re-hashed identical to the Milestone 5 run. |
| 2026-09-28 | 7 | Fresh clone of `b106a2b`: `UV_PYTHON=3.10 uv sync --locked`, then `UV_PYTHON=3.10 uv run python scripts/check.py` | All five checks passed on CPython 3.10.21 with onnxruntime 1.23.2 and numpy 2.2.6. Format 104 files; lint passed; mypy no issues in 86 files; pytest 567 passed, 1 skipped (the real-data integration test: no dataset in the clone); dist: 48-file wheel, 4 smoke tests; exit 0. |
