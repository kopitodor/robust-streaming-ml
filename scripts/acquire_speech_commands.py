"""Acquire Speech Commands v0.02: download (if needed), verify, and extract.

Usage (from the repository root)::

    uv run python scripts/acquire_speech_commands.py [--data-dir data]

The archive goes to ``<data-dir>/raw/speech_commands_v0.02.tar.gz`` and must
match the SHA-256 pinned in docs/data-contract.md. The contents are extracted
to ``<data-dir>/raw/speech_commands_v0.02/``. Both locations are git-ignored.
Re-running reuses a verified archive and a completed extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robust_streaming_ml.data.acquisition import SPEECH_COMMANDS_V0_02, AcquisitionError, acquire


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    last_percent = -1

    def progress(received: int, total: int) -> None:
        nonlocal last_percent
        percent = received * 100 // total
        if percent >= last_percent + 10:
            last_percent = percent
            print(f"downloaded {percent}%", flush=True)

    spec = SPEECH_COMMANDS_V0_02
    print(f"{spec.name} {spec.version}: expecting SHA-256 {spec.sha256}", flush=True)
    try:
        dest = acquire(spec, args.data_dir, progress=progress)
    except AcquisitionError as exc:
        print(f"acquisition failed: {exc}", file=sys.stderr)
        return 1
    print(f"archive verified; extracted to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
