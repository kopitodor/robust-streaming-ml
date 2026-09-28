"""ONNX export and parity (protocols kw10-onnx-v1 and kw10-streaming-v1).

Usage (from the repository root)::

    uv run python scripts/export_onnx.py --config configs/kw10_cnn_baseline_onnx.yaml

Checks every gate in docs/streaming-onnx-protocol.md first: the pinned
checkpoint and its provenance, the validated manifest, the pinned Milestone 4
baseline, the exact export toolchain and ONNX Runtime version, and a clean
source. After ``start_run`` it exports the model (twice, byte-identical),
reproduces the Milestone 4 test logits bit-for-bit on the PyTorch path, and
compares ONNX Runtime with PyTorch on the test clips and on concatenation
streams against the predeclared tolerances. Exits 1 on any gate failure,
error, or parity violation; a parity failure still writes the full report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_streaming_ml.checkpoint import CheckpointError
from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.evaluation.predictions import PredictionError
from robust_streaming_ml.evaluation.run import EvaluationError
from robust_streaming_ml.export.config import ExportConfig
from robust_streaming_ml.export.onnx_model import OnnxModelError
from robust_streaming_ml.export.run import ExportError, export_and_verify
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.streaming.windowing import StreamingError
from robust_streaming_ml.training.data import DataCompatibilityError, DataIntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ONNX export and PyTorch/ONNX parity.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow uncommitted changes (development only; recorded as dirty)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=ExportConfig)
        result = export_and_verify(config, allow_dirty_source=args.allow_dirty_source)
    except (
        ConfigError,
        ExportError,
        OnnxModelError,
        StreamingError,
        CheckpointError,
        EvaluationError,
        PredictionError,
        DataCompatibilityError,
        DataIntegrityError,
        ProvenanceError,
    ) as exc:
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run: {result.run.path}")
    print(json.dumps(result.summary["headline"], indent=2))
    if not result.passed:
        print("parity FAILED: tolerances violated; see report.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
