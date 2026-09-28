# Contributing

Keep changes reproducible, scoped, and evidence-based.

- Do not commit datasets, model weights, generated audio, credentials, or large experiment artifacts.
- Add or update tests for behavior changes, especially data validation, split logic, feature shapes, and export parity.
- Version configuration and document random seeds, dataset provenance, evaluation protocol, and environment details for any measured result.
- Do not add a result or claim unless its artifacts and method can be reproduced from the repository's documented process.
- Keep pull requests small; explain assumptions, limitations, and validation performed.
- Run `uv run python scripts/check.py` before committing: format, lint, strict type check, tests, and the offline wheel check must all pass. Add new package modules to `REQUIRED_WHEEL_FILES` in `scripts/check_dist.py` and to `tests/test_import.py`.
- Commit a protocol (`docs/*-protocol.md`, with its fixed parameters and tolerances) and the code that implements it before the first real run it governs. Commit measured results separately, after verifying them. Never tune, select, or refit on the held-out test split.
- Formal runs go through `start_run`: they need a clean tree and a git revision it can resolve. Do not change global git configuration. If git reports "dubious ownership", use the process-scoped override described in the README.
- `RESULTS.md` holds only executed commands, real run IDs, verified artifacts, and measured values, including failures, non-runs, and known recording issues.
- Lock new dependencies in `uv.lock`, and keep the lock installable on the CI interpreter (CPython 3.10, Linux x86-64).
