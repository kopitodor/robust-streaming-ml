"""Download, verify, and safely extract a pinned dataset archive.

Implements the acquisition step of docs/data-contract.md section 1:

- the archive is downloaded to ``<name>.part`` and only moved into place
  after its size and SHA-256 match the pinned values;
- an existing archive is reused only if it matches, and is never
  overwritten silently;
- extraction rejects unsafe members (absolute paths, ``..``, links, devices),
  extracts into a temporary sibling directory, checks the pinned files inside,
  and then atomically renames it into place with a completion record.

This module does not decode audio, build manifests, or validate labels and
splits; those are separate steps of the contract.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "SPEECH_COMMANDS_V0_02",
    "AcquisitionError",
    "ArchiveSpec",
    "acquire",
    "download_archive",
    "extract_archive",
    "sha256_file",
    "verify_archive",
]

EXTRACTION_RECORD = ".acquisition.json"
_CHUNK = 1 << 20


class AcquisitionError(RuntimeError):
    """Raised when an archive cannot be obtained, verified, or extracted safely."""


@dataclass(frozen=True)
class ArchiveSpec:
    """A pinned dataset archive and the files inside it that must match."""

    name: str
    version: str
    filename: str
    source_url: str
    retrieval_url: str
    sha256: str
    size_bytes: int
    pinned_members: dict[str, str] = field(default_factory=dict)

    @property
    def extract_dirname(self) -> str:
        return f"{self.name}_{self.version}"


# Values verified on 2026-09-27; see docs/datasets/speech_commands_v0.02.md.
SPEECH_COMMANDS_V0_02 = ArchiveSpec(
    name="speech_commands",
    version="v0.02",
    filename="speech_commands_v0.02.tar.gz",
    source_url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    retrieval_url=(
        "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    ),
    sha256="af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    size_bytes=2428923189,
    pinned_members={
        "LICENSE": "fd3a263fe19ed8faa9068b43abaebafc02c77897b0c6fc09abc04bb592e5f16e",
        "README.md": "b415da1f2d880bcce33b2958fcba9bba46013c1bc5592fa4aaa85759d9308b6f",
        "validation_list.txt": "5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2",
        "testing_list.txt": "2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5",
    },
)

ProgressCallback = Callable[[int, int], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: ArchiveSpec) -> None:
    """Raise :class:`AcquisitionError` unless ``path`` matches the pinned size and hash."""
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise AcquisitionError(f"{path}: size {size} bytes, expected {spec.size_bytes}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise AcquisitionError(f"{path}: SHA-256 {actual}, expected {spec.sha256}")


def download_archive(
    spec: ArchiveSpec,
    raw_dir: Path,
    *,
    url: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Return the verified archive in ``raw_dir``, downloading it if absent.

    An existing archive is verified, never replaced: if it does not match, this
    raises and the caller must remove it deliberately.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / spec.filename
    if target.exists():
        verify_archive(target, spec)
        return target

    part = target.with_name(target.name + ".part")
    source = url if url is not None else spec.retrieval_url
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(source, timeout=60) as response, part.open("wb") as out:
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received > spec.size_bytes:
                    raise AcquisitionError(
                        f"{source}: received more than the expected {spec.size_bytes} bytes"
                    )
                if progress is not None:
                    progress(received, spec.size_bytes)
    except (OSError, AcquisitionError) as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, AcquisitionError):
            raise
        raise AcquisitionError(f"download from {source} failed: {exc}") from exc

    if received != spec.size_bytes or digest.hexdigest() != spec.sha256:
        part.unlink(missing_ok=True)
        raise AcquisitionError(
            f"{source}: downloaded {received} bytes with SHA-256 {digest.hexdigest()}, "
            f"expected {spec.size_bytes} bytes with SHA-256 {spec.sha256}"
        )
    part.replace(target)
    return target


def _normalized_name(member: tarfile.TarInfo) -> str | None:
    """Validated POSIX member path relative to the archive root, or None for the root."""
    name = member.name
    if name in (".", "./"):
        return None
    if name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or ".." in path.parts or not path.parts:
        raise AcquisitionError(f"unsafe path in archive: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise AcquisitionError(f"unsupported member type in archive: {member.name!r}")
    return path.as_posix()


def _safe_members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    seen: set[str] = set()
    for member in tf:
        name = _normalized_name(member)
        if name is None:
            continue
        key = name.lower()  # also catches collisions on case-insensitive filesystems
        if key in seen and member.isfile():
            raise AcquisitionError(f"duplicate path in archive: {member.name!r}")
        seen.add(key)
        member.name = name
        member.mode = 0o755 if member.isdir() else 0o644
        yield member


def extract_archive(spec: ArchiveSpec, archive: Path, data_dir: Path) -> Path:
    """Safely extract ``archive`` to ``data_dir/<name>_<version>/`` and return that path.

    A completed extraction (with a matching completion record) is reused. A
    directory without a matching record is an error rather than being reused
    or deleted.
    """
    dest = data_dir / spec.extract_dirname
    record_path = dest / EXTRACTION_RECORD
    if dest.exists():
        if record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get("archive_sha256") == spec.sha256:
                return dest
        raise AcquisitionError(
            f"{dest} exists but is not a completed extraction of this archive; "
            "remove it to re-extract"
        )

    if not hasattr(tarfile, "data_filter"):
        raise AcquisitionError("safe extraction needs tarfile filters (Python 3.10.12 or newer)")
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = data_dir / f".{spec.extract_dirname}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    file_count = 0
    try:
        with tarfile.open(archive, "r:*") as tf:
            for member in _safe_members(tf):
                tf.extract(member, staging, filter="data")
                file_count += member.isfile()
        for rel, expected in spec.pinned_members.items():
            path = staging / rel
            if not path.is_file():
                raise AcquisitionError(f"pinned file missing from archive: {rel}")
            actual = sha256_file(path)
            if actual != expected:
                raise AcquisitionError(f"pinned file {rel}: SHA-256 {actual}, expected {expected}")
        record: dict[str, Any] = {
            "dataset_name": spec.name,
            "dataset_version": spec.version,
            "archive_filename": spec.filename,
            "archive_sha256": spec.sha256,
            "source_url": spec.source_url,
            "file_count": file_count,
            "pinned_members": spec.pinned_members,
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / EXTRACTION_RECORD).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest


def acquire(
    spec: ArchiveSpec,
    data_dir: Path,
    *,
    url: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download (if needed), verify, and extract ``spec`` under ``data_dir``.

    The archive is kept in ``data_dir/raw/``; the extraction goes to
    ``data_dir/raw/<name>_<version>/``. Returns the extraction directory.
    """
    raw_dir = data_dir / "raw"
    archive = download_archive(spec, raw_dir, url=url, progress=progress)
    return extract_archive(spec, archive, raw_dir)
