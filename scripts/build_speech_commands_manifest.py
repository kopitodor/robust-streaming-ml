"""Validate the extracted Speech Commands v0.02 subset and build its manifest.

Usage (from the repository root, after scripts/acquire_speech_commands.py)::

    uv run python scripts/build_speech_commands_manifest.py [--data-dir data]

Reads ``<data-dir>/raw/speech_commands_v0.02/`` and the pinned archive in
``<data-dir>/raw/``. Writes, all git-ignored:

- ``<data-dir>/manifests/speech_commands_v0.02_kw10.csv`` and ``.json`` (sidecar);
- ``<data-dir>/reports/speech_commands_v0.02_kw10_validation.json`` and ``.md``.

Fails closed (exit 1) on any data-contract violation: no manifest is left in
place and the report lists every violation. The code revision must be
resolvable by git (see README, "Configuration, seeds, and run directories").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import DatasetValidationError, build_manifest
from robust_streaming_ml.run_dir import ProvenanceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Speech Commands and build manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=None, help="file-reading threads")
    args = parser.parse_args(argv)

    spec = SPEECH_COMMANDS_KW10
    raw = args.data_dir / "raw"
    try:
        built = build_manifest(
            spec,
            raw / spec.archive.extract_dirname,
            raw / spec.archive.filename,
            manifest_dir=args.data_dir / "manifests",
            report_dir=args.data_dir / "reports",
            workers=args.workers,
        )
    except (DatasetValidationError, ProvenanceError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    report = built.result.report
    print(f"manifest: {built.manifest_path} ({built.row_count} rows)")
    print(f"manifest SHA-256: {built.manifest_sha256}")
    print(f"counts by split: {report['counts']['by_split']}")
    print(f"warnings: {report['warnings']['by_code'] or 'none'}")
    print(f"report: {built.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
