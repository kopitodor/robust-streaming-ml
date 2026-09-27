"""Robustness suite (protocol kw10-robustness-v1) on the official held-out test split.

Usage (from the repository root)::

    uv run python scripts/evaluate_robustness.py --config configs/kw10_cnn_baseline_robustness.yaml

Checks every gate in docs/robustness-protocol.md first: the pinned checkpoint
and its provenance, the validated manifest, the pinned Milestone 4 baseline,
the exact codec backend, and a clean source. After ``start_run`` it reproduces
the Milestone 4 test logits bit-for-bit and evaluates all 5 corruptions x 3
severities with raw probabilities. Exits 1 (fail closed) on any error; a
failure inside the run is recorded in ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.evaluation.predictions import PredictionError
from robust_streaming_ml.evaluation.run import EvaluationError
from robust_streaming_ml.robustness.config import RobustnessConfig
from robust_streaming_ml.robustness.corruptions import CorruptionError
from robust_streaming_ml.robustness.run import RobustnessError, evaluate_robustness
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.training.data import DataCompatibilityError, DataIntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled-corruption robustness suite.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow uncommitted changes (development only; recorded as dirty)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=RobustnessConfig)
        result = evaluate_robustness(config, allow_dirty_source=args.allow_dirty_source)
    except (
        ConfigError,
        RobustnessError,
        CorruptionError,
        EvaluationError,
        PredictionError,
        DataCompatibilityError,
        DataIntegrityError,
        ProvenanceError,
    ) as exc:
        print(f"robustness evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run: {result.run.path}")
    print(json.dumps(result.summary["headline"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
