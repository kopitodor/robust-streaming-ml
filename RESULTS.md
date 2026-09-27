# Results

The experiment results table holds the Milestone 4 clean evaluation, protocol `kw10-clean-eval-v1` (docs/evaluation-protocol.md). The robustness and latency tables stay empty until their protocols run.

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
| | | | | | | | | | | |

## Latency measurements

| Run ID | Model/runtime | CPU/hardware | OS | Threads | Window/hop | Warm-up | Timed iterations | Batch size | p50 ms | p95 ms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | |

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
