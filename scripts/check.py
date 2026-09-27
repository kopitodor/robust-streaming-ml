"""Run every repository check: formatting, linting, type checking, tests.

Usage: ``python scripts/check.py`` (from the repository root, inside the dev
environment, e.g. ``uv run python scripts/check.py``). Exits non-zero if any
check fails; all checks run regardless so one invocation reports everything.
"""

from __future__ import annotations

import subprocess
import sys

CHECKS: list[tuple[str, list[str]]] = [
    ("format", ["ruff", "format", "--check", "."]),
    ("lint", ["ruff", "check", "."]),
    ("types", ["mypy"]),
    ("tests", ["pytest"]),
]


def main() -> int:
    failed: list[str] = []
    for name, args in CHECKS:
        print(f"==> {name}: {' '.join(args)}", flush=True)
        if subprocess.run([sys.executable, "-m", *args], check=False).returncode != 0:
            failed.append(name)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
