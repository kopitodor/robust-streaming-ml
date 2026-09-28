"""Local streaming keyword-spotting demo (offline; nothing is uploaded).

Usage (from the repository root)::

    uv run python scripts/demo.py --config configs/kw10_demo.yaml --audio path/to/clip.wav
    uv run python scripts/demo.py --config configs/kw10_demo.yaml --audio path/to/clip.wav \\
        --corruption noise --severity snr_10db --open
    uv run python scripts/demo.py --list-corruptions

Streams one local 16 kHz mono 16-bit PCM WAV file through the pinned,
parity-verified ONNX model (``kw10-streaming-v1``: 1 s windows, 100 ms hop),
optionally also through one ``kw10-robustness-v1`` corruption. Prints the
per-window predictions and writes a self-contained HTML page and a JSON
record under ``artifacts/demo/`` (git-ignored). ``--open`` opens the local
page in the default browser. Demo output is not a measured result, and the
model has no silence or unknown-word rejection. Exits 1 on any error.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.demo.app import DemoError, run_demo
from robust_streaming_ml.demo.config import DemoConfig
from robust_streaming_ml.export.onnx_model import OnnxModelError
from robust_streaming_ml.export.run import ExportError
from robust_streaming_ml.robustness.corruptions import CORRUPTIONS, CorruptionError
from robust_streaming_ml.streaming.windowing import StreamingError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local streaming keyword-spotting demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/kw10_demo.yaml"))
    parser.add_argument("--audio", type=Path, help="local 16 kHz mono 16-bit PCM WAV file")
    parser.add_argument("--corruption", choices=[c.name for c in CORRUPTIONS])
    parser.add_argument("--severity", help="a severity of the chosen corruption")
    parser.add_argument("--output", type=Path, help="HTML path (default: under artifacts/demo)")
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    parser.add_argument("--list-corruptions", action="store_true")
    args = parser.parse_args(argv)
    if args.list_corruptions:
        for spec in CORRUPTIONS:
            print(f"{spec.name}: {', '.join(s.name for s in spec.severities)}")
        return 0
    if args.audio is None:
        parser.error("--audio is required")
    try:
        config = load_config(args.config, schema=DemoConfig)
        result = run_demo(
            config,
            args.audio,
            corruption=args.corruption,
            severity=args.severity,
            output=args.output,
        )
    except (
        ConfigError,
        DemoError,
        ExportError,
        OnnxModelError,
        CorruptionError,
        StreamingError,
    ) as exc:
        print(f"demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    record = result.record
    labels = record["model"]["labels"]
    print("Local demo output for one file: not a measured result; not production-ready.")
    print("Closed set: every window gets one of the 10 keywords (no silence/unknown rejection).")
    print(f"model {record['model']['onnx_sha256'][:12]}... (parity passed)  labels: {labels}")
    for name, run in record["runs"].items():
        if run is None:
            continue
        print(f"\n[{name}] window  start-end s   label (conf)       aggregate (conf)   ms")
        for w in run["windows"]:
            print(
                f"  {w['index']:>6}  {w['start_s']:5.2f}-{w['end_s']:5.2f}  "
                f"{w['label']:>6} ({w['confidence']:.3f})   {w['aggregate_label']:>6} "
                f"({w['aggregate_confidence']:.3f})   {w['inference_ms']:.2f}"
                + ("  [final, zero-padded]" if w["final_partial"] else "")
            )
        decision = run["decision"]
        verdict = (
            f"{decision['label']} ({decision['confidence']:.3f})" if decision else "none (empty)"
        )
        print(f"  stream decision: {verdict}")
    print(f"\nHTML: {result.html_path}\nJSON: {result.json_path}")
    if args.open:
        webbrowser.open(result.html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
