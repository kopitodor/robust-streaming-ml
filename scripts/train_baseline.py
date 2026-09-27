"""Train the baseline CNN on the validated Speech Commands 10-keyword manifest.

Usage (from the repository root, after building the manifest)::

    uv run python scripts/train_baseline.py --config configs/kw10_cnn_baseline.yaml

Checks config/manifest/report compatibility and the source revision first,
then starts a run with ``start_run`` and writes logs, checkpoints, and a
summary under ``<run.output_root>/<run_id>/``. Only the train and validation
splits are used. Exits 1 (fail closed) on any compatibility, provenance, data
integrity, or training error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_streaming_ml.checkpoint import CheckpointError
from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.training.config import TrainingConfig
from robust_streaming_ml.training.data import DataCompatibilityError, DataIntegrityError
from robust_streaming_ml.training.loop import TrainingError, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline keyword-spotting CNN.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow uncommitted changes (development runs only; recorded as dirty)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=TrainingConfig)
        result = train(config, allow_dirty_source=args.allow_dirty_source)
    except (
        ConfigError,
        DataCompatibilityError,
        DataIntegrityError,
        ProvenanceError,
        TrainingError,
        CheckpointError,
    ) as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = result.summary
    print(f"run: {result.run.path}")
    print(json.dumps({k: summary[k] for k in ("best", "epochs_completed", "num_parameters")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
