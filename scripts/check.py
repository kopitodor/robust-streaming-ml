"""Run every repository check: format, lint, types, tests, distribution.

Usage: ``python scripts/check.py`` (from the repository root, inside the dev
environment, e.g. ``uv run python scripts/check.py``). Exits non-zero if any
check fails; all checks run regardless so one invocation reports everything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PY = sys.executable
CHECKS: list[tuple[str, list[str]]] = [
    ("format", [PY, "-m", "ruff", "format", "--check", "."]),
    ("lint", [PY, "-m", "ruff", "check", "."]),
    ("types", [PY, "-m", "mypy"]),
    ("tests", [PY, "-m", "pytest"]),
    ("dist", [PY, str(Path(__file__).with_name("check_dist.py"))]),
]


def main() -> int:
    failed: list[str] = []
    for name, args in CHECKS:
        print(f"==> {name}: {' '.join(args[1:])}", flush=True)
        if subprocess.run(args, check=False).returncode != 0:
            failed.append(name)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
