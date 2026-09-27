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
