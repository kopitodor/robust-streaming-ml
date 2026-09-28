"""CPU latency benchmark (protocol kw10-latency-v1).

Usage (from the repository root)::

    uv run python scripts/benchmark_latency.py --config configs/kw10_onnx_benchmark.yaml

Checks every gate in docs/benchmark-protocol.md first: the pinned checkpoint
and its provenance, the validated manifest, the pinned Milestone 6 export
(passed parity), the exact ONNX Runtime version, and a clean source. After
``start_run`` it times the ONNX streaming step, ``session.run`` alone, and the
PyTorch reference step with 1 and 4 threads (500 warm-up and 5,000 timed
steps x 3 repetitions each) and writes raw timings and nearest-rank p50/p95.
The numbers describe this machine only. Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robust_streaming_ml.benchmark.config import BenchmarkConfig
from robust_streaming_ml.benchmark.run import BenchmarkError, run_benchmark
from robust_streaming_ml.checkpoint import CheckpointError
from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.evaluation.run import EvaluationError
from robust_streaming_ml.export.onnx_model import OnnxModelError
from robust_streaming_ml.export.run import ExportError
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.streaming.windowing import StreamingError
from robust_streaming_ml.training.data import DataCompatibilityError, DataIntegrityError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU latency benchmark (kw10-latency-v1).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="allow uncommitted changes (development only; recorded as dirty)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=BenchmarkConfig)
        result = run_benchmark(config, allow_dirty_source=args.allow_dirty_source)
    except (
        ConfigError,
        BenchmarkError,
        ExportError,
        OnnxModelError,
        StreamingError,
        CheckpointError,
        EvaluationError,
        DataCompatibilityError,
        DataIntegrityError,
        ProvenanceError,
    ) as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run: {result.run.path}")
    print(json.dumps(result.summary["headline_ms"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
