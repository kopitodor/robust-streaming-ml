# Scripts

Explicit, documented entry points for acquisition, validation, training, evaluation, export, benchmarking, and demo launch will live here.

- `acquire_speech_commands.py`: downloads (if needed), verifies, and extracts Speech Commands v0.02 under `data/raw/`: `uv run python scripts/acquire_speech_commands.py`.
- `evaluate_baseline.py`: formal clean evaluation of the pinned checkpoint through `start_run` (fails closed): `uv run python scripts/evaluate_baseline.py --config configs/kw10_cnn_baseline_eval.yaml`.
- `train_baseline.py`: trains the baseline from a `TrainingConfig` YAML through `start_run` (fails closed): `uv run python scripts/train_baseline.py --config configs/kw10_cnn_baseline.yaml`.
- `build_speech_commands_manifest.py`: validates the extraction against the data contract and writes the manifest, sidecar, and reports under `data/` (fails closed): `uv run python scripts/build_speech_commands_manifest.py`.
- `check.py` — runs formatting check, lint, type check, tests, and the distribution check: `uv run python scripts/check.py`.
- `check_dist.py` — builds the wheel offline, installs it into an isolated temporary environment, and runs the import smoke tests there: `uv run python scripts/check_dist.py`.
