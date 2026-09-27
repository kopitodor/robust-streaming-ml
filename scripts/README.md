# Scripts

Explicit, documented entry points for acquisition, validation, training, evaluation, export, benchmarking, and demo launch will live here.

- `acquire_speech_commands.py`: downloads (if needed), verifies, and extracts Speech Commands v0.02 under `data/raw/`: `uv run python scripts/acquire_speech_commands.py`.
- `check.py` — runs formatting check, lint, type check, tests, and the distribution check: `uv run python scripts/check.py`.
- `check_dist.py` — builds the wheel offline, installs it into an isolated temporary environment, and runs the import smoke tests there: `uv run python scripts/check_dist.py`.
