# Review findings

## P1

### Run provenance can silently omit the source revision

**File:** `src/robust_streaming_ml/run_dir.py:80-98, 128, 170` (also observed in `RESULTS.md:20`)

`_git_info` converts every Git failure into `null`, and `create_run_dir` writes that incomplete metadata without failing or requiring an explicit revision. The recorded Milestone 1 check already demonstrates this path. A later training, evaluation, or benchmark run could therefore be reported without a code revision, making it impossible to reproduce the exact implementation behind a result.

**Concrete fix:** Require a resolved Git commit when creating a portfolio result run. If Git cannot provide one, fail `create_run_dir` with its diagnostic, or require an explicit, validated revision value and record it in `metadata.json`.

### Run metadata claims a configured seed state without proving it was applied

**File:** `src/robust_streaming_ml/run_dir.py:161-170`; `src/robust_streaming_ml/seed.py:50-67`

`create_run_dir` records `config.seed` and `config.deterministic`, but it neither calls `seed_everything` nor accepts a verified `SeedState`. Consequently, a caller can create metadata saying deterministic seed 17 while running with different global RNG and PyTorch determinism state. That breaks the traceability required for later training, corruption generation, evaluation, and benchmark comparisons.

**Concrete fix:** Provide one run-start API that calls `seed_everything(config.seed, deterministic=config.deterministic)` before collecting metadata, then records the returned state and the observed PyTorch determinism state; require later entry points to use it.

## P2

### Repository provenance behavior lacks a successful-path test

**File:** `tests/test_run_dir.py:83-96`

The tests only verify that Git metadata is absent outside a repository. They do not create or use a repository to assert that a commit and dirty-state value are recorded, nor do they test the current failure behavior described in `RESULTS.md`. This leaves the most important provenance branch unverified and allows future changes to silently discard revision metadata.

**Concrete fix:** Add tests using a temporary initialized Git repository with a committed fixture: assert the recorded commit matches `HEAD`, assert dirty-state handling, and assert that an unavailable revision follows the chosen fail-or-explicit-revision policy.

### The package import test does not verify the built distribution

**File:** `tests/test_import.py:8-14`; `pyproject.toml:1-3, 33-34`

The import test runs from the checkout and does not build or install the wheel. It therefore cannot catch a packaging regression such as omitted package data, an incorrect `src` package selection, or a broken wheel build. Later reproducible evaluation and benchmarking commands depend on installable, pinned project code rather than accidental source-tree imports.

**Concrete fix:** Add the build frontend to the development tooling and a check that builds the wheel into a temporary directory, installs it into an isolated environment, and runs the import smoke test there.
