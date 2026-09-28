"""Live-microphone mode of the local demo (localhost only; nothing is uploaded).

Usage (from the repository root)::

    uv run python scripts/live_demo.py
    uv run python scripts/live_demo.py --port 8765 --open

Loads the pinned, parity-verified ONNX export from ``configs/kw10_demo.yaml``
(the same fail-closed checks as the file demo). It then serves a browser UI
and JSON API on http://127.0.0.1:8765/. Only loopback addresses can be bound.

Press "Start listening" in the page and allow the microphone. The browser
sends ordered 100 ms chunks of mono device-rate audio to this process, which
converts them to 16 kHz 16-bit PCM with a documented deterministic resampler.
It then runs the existing ``kw10-streaming-v1`` engine and returns every window
with its label, confidence, timestamp, last-5-window aggregate, stability, and
inference time. Audio is kept in memory only; nothing is written to disk.
Stop with Ctrl+C. See docs/live-demo.md.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from robust_streaming_ml.config import ConfigError, load_config
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.demo.app import DemoError, load_pinned_model, model_identity
from robust_streaming_ml.demo.config import DemoConfig
from robust_streaming_ml.export.onnx_model import OnnxModelError
from robust_streaming_ml.export.run import ExportError
from robust_streaming_ml.live.server import LiveDemoServer, LiveServerError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live-microphone keyword demo (localhost only).")
    parser.add_argument("--config", type=Path, default=Path("configs/kw10_demo.yaml"))
    parser.add_argument("--host", help="loopback address to bind (default: live.host, 127.0.0.1)")
    parser.add_argument("--port", type=int, help="port (default: live.port, 8765; 0 = any free)")
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, schema=DemoConfig)
        model, export_report = load_pinned_model(config)
        spec = SPEECH_COMMANDS_KW10
        info = {
            "model": model_identity(config, model, export_report),
            "dataset": {
                "name": spec.archive.name,
                "version": spec.archive.version,
                "subset": spec.manifest_name,
                "license": spec.metadata.get("license_name"),
                "attribution": spec.metadata.get("attribution_text"),
            },
        }
        server = LiveDemoServer(
            model,
            info=info,
            host=args.host or config.live.host,
            port=config.live.port if args.port is None else args.port,
            intra_op_threads=config.demo.intra_op_threads,
            max_sessions=config.live.max_sessions,
        )
    except (ConfigError, DemoError, ExportError, OnnxModelError, LiveServerError) as exc:
        print(f"live demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Live keyword demo: closed set of 10 commands; not general speech recognition;")
    print("not production software. Audio stays on this computer (in memory only).")
    print(
        f"model {model.sha256[:12]}... (parity passed), export run {export_report['export_run_id']}"
    )
    print(f"Open {server.url} and press 'Start listening'. Stop with Ctrl+C.", flush=True)
    if args.open:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
