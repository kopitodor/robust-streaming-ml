"""Build the wheel, install it into an isolated environment, and smoke-test it.

Usage: ``uv run python scripts/check_dist.py`` from the repository root, after
``uv sync``. Requires ``uv`` on PATH. Runs fully offline:

1. ``uv build --wheel --no-build-isolation --offline`` builds the wheel with
   the locked ``hatchling`` from the dev environment;
2. the wheel's contents are checked (package modules and ``py.typed`` present,
   no tests or stray top-level packages);
3. ``uv sync --frozen --offline --no-dev --group smoke --no-install-project``
   creates a fresh temporary environment with the locked runtime
   dependencies and pytest, from the local uv cache only;
4. the wheel is installed there with ``--no-deps``;
5. ``tests/test_import.py`` is copied to an empty directory and run with that
   environment's interpreter in isolated mode, after confirming the package
   is imported from the environment's site-packages, not the source tree.

Exits non-zero on any failure. Temporary files are removed afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "robust_streaming_ml"
REQUIRED_WHEEL_FILES = {
    f"{PACKAGE}/__init__.py",
    f"{PACKAGE}/config.py",
    f"{PACKAGE}/data/__init__.py",
    f"{PACKAGE}/data/acquisition.py",
    f"{PACKAGE}/py.typed",
    f"{PACKAGE}/run_dir.py",
    f"{PACKAGE}/seed.py",
}
LOCATION_CHECK = """
import sys, sysconfig
from pathlib import Path
import robust_streaming_ml
location = Path(robust_streaming_ml.__file__).resolve()
site = Path(sysconfig.get_paths()["purelib"]).resolve()
print(f"imported {location}")
if site not in location.parents:
    sys.exit(f"package imported from {location}, not from {site}")
"""


def run(args: list[str | Path], **kwargs: object) -> None:
    print("$", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kwargs)  # type: ignore[call-overload]


def check_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    missing = REQUIRED_WHEEL_FILES - names
    if missing:
        raise SystemExit(f"wheel is missing files: {sorted(missing)}")
    top_level = {name.split("/", 1)[0] for name in names}
    unexpected = {t for t in top_level if t != PACKAGE and not t.endswith(".dist-info")}
    if unexpected:
        raise SystemExit(f"wheel contains unexpected top-level entries: {sorted(unexpected)}")
    print(f"wheel contents ok: {len(names)} files")


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required on PATH", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="rsml-dist-", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        dist, env_dir, smoke = tmp_path / "dist", tmp_path / "venv", tmp_path / "smoke"

        run(
            [uv, "build", "--wheel", "--no-build-isolation", "--offline", "--out-dir", dist],
            cwd=ROOT,
        )
        wheels = sorted(dist.glob("*.whl"))
        if len(wheels) != 1:
            print(f"expected exactly one wheel, found {wheels}", file=sys.stderr)
            return 1
        wheel = wheels[0]
        check_wheel_contents(wheel)

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
        run(
            [
                uv,
                "sync",
                "--frozen",
                "--offline",
                "--no-dev",
                "--group",
                "smoke",
                "--no-install-project",
                "--quiet",
            ],
            cwd=ROOT,
            env=env,
        )
        run([uv, "pip", "install", "--offline", "--no-deps", "--python", env_dir, wheel])

        scripts = env_dir / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        smoke.mkdir()
        shutil.copy2(ROOT / "tests" / "test_import.py", smoke / "test_import.py")
        (smoke / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        run([python, "-I", "-c", LOCATION_CHECK], cwd=smoke)
        run([python, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=smoke)

    print("Distribution check passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit status {exc.returncode}", file=sys.stderr)
        sys.exit(1)
