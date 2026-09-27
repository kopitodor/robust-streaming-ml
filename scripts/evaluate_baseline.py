"""Formal clean evaluation (protocol kw10-clean-eval-v1) of the pinned baseline checkpoint.

Usage (from the repository root)::

    uv run python scripts/evaluate_baseline.py --config configs/kw10_cnn_baseline_eval.yaml

Checks every gate in docs/evaluation-protocol.md (pinned checkpoint and its
provenance, validated manifest, clean source) before ``start_run``. It then
predicts on validation, fits the temperature on validation only, predicts on
test, and writes predictions, ``calibration.json``, and the report under the
new run directory. Exits 1 (fail closed) on any gate or evaluation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.evaluation.calibration import CalibrationError
from robust_streaming_ml.evaluation.config import EvaluationConfig
from robust_streaming_ml.evaluation.predictions import PredictionError
from robust_streaming_ml.evaluation.run import EvaluationError, evaluate
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.training.data import DataCompatibilityError, DataIntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal clean evaluation of the baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow uncommitted changes (development only; recorded as dirty)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=EvaluationConfig)
        result = evaluate(config, allow_dirty_source=args.allow_dirty_source)
    except (
        ConfigError,
        EvaluationError,
        CalibrationError,
        PredictionError,
        DataCompatibilityError,
        DataIntegrityError,
        ProvenanceError,
    ) as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run: {result.run.path}")
    print(json.dumps(result.summary["headline"], indent=2))
    print(f"temperature: {result.summary['temperature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
