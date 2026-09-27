# Results

No experiment results exist yet. Do not populate the experiment, robustness, or latency tables until runs are reproducible and artifacts are retained.

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
| | | | | | | | | | | | |

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
