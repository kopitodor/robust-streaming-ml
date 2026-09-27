"""Validate an extracted dataset against the data contract and build its manifest.

Implements docs/data-contract.md sections 2-7. :func:`validate_dataset`
collects every contract finding without writing anything; :func:`build_manifest`
runs it and writes outputs. It fails closed: if there is any error, no manifest
or sidecar is left in place (stale ones are removed), a failed validation report
is written, and :class:`DatasetValidationError` is raised. Nothing is repaired,
relabelled, dropped, or resampled.

Duplicate-content protocol: ``sha256`` is the SHA-256 of a file's bytes;
``audio_sha256`` is the SHA-256 of the ``data`` chunk's sample bytes
(little-endian signed 16-bit mono after format validation), excluding every
header and other chunk. Duplicate rules are applied to ``audio_sha256``, so files
whose audio is identical but whose headers differ are still caught.
Near-duplicates (re-encoded, trimmed, or gain-changed audio) are not detected.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from robust_streaming_ml import run_dir
from robust_streaming_ml.data.acquisition import EXTRACTION_RECORD, ArchiveSpec
from robust_streaming_ml.data.manifest import (
    COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    SPLITS,
    Issue,
    ManifestRow,
    check_rows,
    encode_manifest,
    format_duration,
    parse_filename,
    read_manifest,
)
from robust_streaming_ml.data.wav import WavError, parse_wav

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "BuildResult",
    "DatasetSpec",
    "DatasetValidationError",
    "ValidationResult",
    "build_manifest",
    "validate_dataset",
]

REPORT_SCHEMA_VERSION = 1
# Report item caps guard against pathological inputs; real findings are listed in full.
MAX_ITEMS_PER_CODE = 10000
MARKDOWN_ITEMS_PER_CODE = 20
CONTENT_HASH_PROTOCOL = (
    "audio_sha256 = SHA-256 of the WAVE data chunk sample bytes (PCM_16 little-endian, mono), "
    "excluding RIFF/fmt headers and all other chunks; sha256 = SHA-256 of the whole file"
)
_SHA_CHUNK = 1 << 20


class DatasetValidationError(RuntimeError):
    """Raised when the dataset violates the data contract."""


@dataclass(frozen=True)
class DatasetSpec:
    """Contract parameters for one dataset subset."""

    archive: ArchiveSpec
    labels: tuple[str, ...]
    manifest_name: str
    metadata: Mapping[str, str]
    validation_list: str = "validation_list.txt"
    testing_list: str = "testing_list.txt"
    license_file: str = "LICENSE"
    sample_rate_hz: int = 16000
    num_channels: int = 1
    sample_format: str = "PCM_16"
    max_frames: int = 16000
    near_silent_peak_abs: int = 32
    # SHA-256 the manifest must have; None only before the first verified build.
    expected_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        required = {self.validation_list, self.testing_list, self.license_file}
        missing = required - set(self.archive.pinned_members)
        if missing:
            raise ValueError(f"archive spec must pin {sorted(missing)}")
        if len(set(self.labels)) != len(self.labels) or not self.labels:
            raise ValueError("labels must be non-empty and unique")

    @property
    def known_top_level_files(self) -> frozenset[str]:
        return frozenset(self.archive.pinned_members) | {EXTRACTION_RECORD}


@dataclass(frozen=True)
class _Audio:
    format_tag: int
    sample_format: str
    num_channels: int
    sample_rate_hz: int
    num_frames: int
    audio_sha256: str
    peak_abs: int | None
    extra_chunks: tuple[str, ...]


@dataclass(frozen=True)
class _FileResult:
    relative_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    error: Issue | None = None
    audio: _Audio | None = None


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_dataset`. ``rows`` is empty unless ``passed``."""

    spec: DatasetSpec
    rows: list[ManifestRow]
    errors: list[Issue]
    warnings: list[Issue]
    report: dict[str, Any]
    list_sha256: dict[str, str] = field(default_factory=dict)
    manifest_bytes: bytes = b""

    @property
    def manifest_sha256(self) -> str:
        return _sha256_bytes(self.manifest_bytes) if self.manifest_bytes else ""

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class BuildResult:
    manifest_path: Path
    sidecar_path: Path
    report_path: Path
    report_markdown_path: Path
    manifest_sha256: str
    row_count: int
    result: ValidationResult


# --- helpers ------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _os_error(exc: OSError) -> str:
    return f"{type(exc).__name__}: {exc.strerror or exc}"


def _nearest_rank(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile: the ceil(q*n)-th smallest value (1-based)."""
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]


class _HashingReader(io.RawIOBase):
    """Read-through wrapper that hashes and counts every byte read."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self.sha256 = hashlib.sha256()
        self.size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._raw.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        self.sha256.update(data)
        self.size += n
        return n


# --- scanning -----------------------------------------------------------------


def _scan(
    root: Path, spec: DatasetSpec
) -> tuple[list[tuple[str, str, str, int]], list[str], dict[str, Any], list[Issue]]:
    """Find candidate files in the label directories.

    Returns ``(candidates, label_dir_files, ignored, errors)`` where candidates
    are ``(relative_path, label, speaker_id, utterance_index)`` and
    ``label_dir_files`` lists every regular file in the label directories.
    """
    errors: list[Issue] = []
    candidates: list[tuple[str, str, str, int]] = []
    label_dir_files: list[str] = []
    top = {entry.name: entry for entry in os.scandir(root)}
    ignored_dirs: dict[str, int] = {}
    ignored_files: list[str] = []

    for name in sorted(top):
        entry = top[name]
        if name in spec.labels:
            continue
        if entry.is_symlink():
            errors.append(Issue("unexpected_entry", name, "symbolic link at top level"))
        elif entry.is_dir():
            ignored_dirs[name] = sum(len(files) for _, _, files in os.walk(entry.path))
        elif name not in spec.known_top_level_files:
            ignored_files.append(name)

    for label in spec.labels:
        label_entry = top.get(label)
        if label_entry is None:
            errors.append(Issue("missing_label_dir", label, "label directory not found"))
            continue
        if label_entry.is_symlink() or not label_entry.is_dir():
            errors.append(Issue("missing_label_dir", label, "not a regular directory"))
            continue
        wav_count = 0
        for child in sorted(os.scandir(label_entry.path), key=lambda e: e.name):
            rel = f"{label}/{child.name}"
            if child.is_symlink() or not child.is_file():
                errors.append(Issue("unexpected_entry", rel, "not a regular file"))
                continue
            label_dir_files.append(rel)
            parsed = parse_filename(child.name)
            if parsed is None:
                errors.append(Issue("bad_filename", rel, "expected <8 hex>_nohash_<integer>.wav"))
                continue
            wav_count += 1
            candidates.append((rel, label, parsed[0], parsed[1]))
        if wav_count == 0:
            errors.append(Issue("empty_label_dir", label, "no valid .wav files"))

    ignored = {"directories": ignored_dirs, "top_level_files": ignored_files}
    return candidates, label_dir_files, ignored, errors


def _check_pinned_files(root: Path, spec: DatasetSpec) -> tuple[dict[str, str], list[Issue]]:
    hashes: dict[str, str] = {}
    errors: list[Issue] = []
    for rel, expected in sorted(spec.archive.pinned_members.items()):
        path = root / rel
        if not path.is_file() or path.is_symlink():
            errors.append(Issue("pinned_file_missing", rel, "pinned file not found"))
            continue
        actual = _sha256_bytes(path.read_bytes())
        hashes[rel] = actual
        if actual != expected:
            errors.append(Issue("pinned_file_mismatch", rel, f"SHA-256 {actual} != {expected}"))
    return hashes, errors


def _parse_split_list(
    root: Path, list_name: str, spec: DatasetSpec, existing: set[str]
) -> tuple[set[str], int, list[Issue]]:
    """Return ``(kept_paths, ignored_other_label_count, errors)`` for one list file."""
    errors: list[Issue] = []
    path = root / list_name
    if not path.is_file():
        return set(), 0, []  # reported by the pinned-file check
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        return set(), 0, [Issue("list_not_utf8", list_name, str(exc))]
    seen: set[str] = set()
    kept: set[str] = set()
    ignored = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        entry = line.replace("\\", "/")
        where = f"{list_name}:{number}"
        if entry.startswith("/") or (len(entry) > 1 and entry[1] == ":"):
            errors.append(Issue("list_absolute_path", where, entry))
            continue
        parts = PurePosixPath(entry).parts
        if any(p in ("", ".", "..") for p in entry.split("/")):
            errors.append(Issue("list_noncanonical_path", where, entry))
            continue
        if entry in seen:
            errors.append(Issue("list_duplicate_entry", where, entry))
            continue
        seen.add(entry)
        if parts[0] not in spec.labels:
            ignored += 1
            continue
        if len(parts) != 2 or parse_filename(parts[1]) is None:
            errors.append(Issue("list_bad_entry", where, entry))
            continue
        if entry not in existing:
            errors.append(Issue("list_missing_file", where, entry))
            continue
        kept.add(entry)
    return kept, ignored, errors


# --- per-file inspection ------------------------------------------------------


def _inspect(root: Path, rel: str) -> _FileResult:
    try:
        data = (root / rel).read_bytes()
    except OSError as exc:
        return _FileResult(rel, error=Issue("file_unreadable", rel, _os_error(exc)))
    sha = _sha256_bytes(data)
    try:
        wav = parse_wav(data)
    except WavError as exc:
        return _FileResult(rel, sha, len(data), error=Issue("wav_malformed", rel, str(exc)))
    peak: int | None = None
    if wav.sample_format == "PCM_16":
        samples = wav.samples_int16()
        peak = int(np.abs(samples.astype(np.int32)).max()) if samples.size else 0
    audio = _Audio(
        format_tag=wav.format_tag,
        sample_format=wav.sample_format,
        num_channels=wav.num_channels,
        sample_rate_hz=wav.sample_rate_hz,
        num_frames=wav.num_frames,
        audio_sha256=_sha256_bytes(wav.sample_bytes),
        peak_abs=peak,
        extra_chunks=wav.extra_chunks,
    )
    return _FileResult(rel, sha, len(data), audio=audio)


def _audio_issues(spec: DatasetSpec, result: _FileResult) -> list[Issue]:
    audio = result.audio
    assert audio is not None
    rel = result.relative_path
    issues = []
    if audio.sample_rate_hz != spec.sample_rate_hz:
        issues.append(Issue("wav_sample_rate", rel, f"{audio.sample_rate_hz} Hz"))
    if audio.num_channels != spec.num_channels:
        issues.append(Issue("wav_channels", rel, f"{audio.num_channels} channels"))
    if audio.sample_format != spec.sample_format:
        issues.append(Issue("wav_sample_format", rel, audio.sample_format))
    if audio.num_frames == 0:
        issues.append(Issue("wav_zero_frames", rel, "no audio frames"))
    elif audio.num_frames > spec.max_frames:
        issues.append(Issue("wav_too_long", rel, f"{audio.num_frames} > {spec.max_frames} frames"))
    return issues


# --- archive cross-check ------------------------------------------------------


def _hash_archive(spec: DatasetSpec, archive: Path) -> tuple[dict[str, str], list[Issue], int]:
    """Stream the archive: verify its pinned SHA-256 and hash files in label dirs."""
    members: dict[str, str] = {}
    errors: list[Issue] = []
    name = archive.name
    try:
        with archive.open("rb") as raw:
            reader = _HashingReader(raw)
            buffered = io.BufferedReader(reader, buffer_size=_SHA_CHUNK)
            with tarfile.open(fileobj=buffered, mode="r|gz") as tf:
                for member in tf:
                    member_name = member.name[2:] if member.name.startswith("./") else member.name
                    parts = member_name.split("/")
                    if len(parts) < 2 or parts[0] not in spec.labels:
                        continue
                    if member.isdir():
                        continue
                    if not member.isfile():
                        errors.append(Issue("archive_unexpected_member", member_name, "not a file"))
                        continue
                    extracted = tf.extractfile(member)
                    assert extracted is not None
                    members[member_name] = _sha256_bytes(extracted.read())
            while buffered.read(_SHA_CHUNK):  # hash any bytes after the tar end marker
                pass
    except (OSError, tarfile.TarError, EOFError) as exc:
        return members, [Issue("archive_unreadable", name, f"{type(exc).__name__}: {exc}")], 0
    actual = reader.sha256.hexdigest()
    if actual != spec.archive.sha256 or reader.size != spec.archive.size_bytes:
        errors.append(
            Issue(
                "archive_checksum_mismatch",
                name,
                f"{reader.size} bytes, SHA-256 {actual}; expected "
                f"{spec.archive.size_bytes} bytes, SHA-256 {spec.archive.sha256}",
            )
        )
    return members, errors, len(members)


def _compare_with_archive(
    archive_hashes: Mapping[str, str], extracted_hashes: Mapping[str, str | None]
) -> list[Issue]:
    errors = []
    for rel in sorted(set(archive_hashes) - set(extracted_hashes)):
        errors.append(Issue("extraction_missing_file", rel, "in archive, not in extraction"))
    for rel in sorted(set(extracted_hashes) - set(archive_hashes)):
        errors.append(Issue("extraction_extra_file", rel, "in extraction, not in archive"))
    for rel in sorted(set(archive_hashes) & set(extracted_hashes)):
        if extracted_hashes[rel] is not None and extracted_hashes[rel] != archive_hashes[rel]:
            errors.append(Issue("extraction_modified_file", rel, "content differs from archive"))
    return errors


# --- validation ---------------------------------------------------------------


def _issue_summary(issues: Sequence[Issue]) -> dict[str, Any]:
    by_code = Counter(i.code for i in issues)
    items: list[dict[str, str]] = []
    omitted = 0
    shown: Counter[str] = Counter()
    for issue in sorted(issues):
        if shown[issue.code] < MAX_ITEMS_PER_CODE:
            shown[issue.code] += 1
            items.append({"code": issue.code, "path": issue.path, "detail": issue.detail})
        else:
            omitted += 1
    return {
        "total": len(issues),
        "by_code": dict(sorted(by_code.items())),
        "items": items,
        "items_omitted": omitted,
    }


def _duration_stats(frames: Sequence[int], sample_rate: int, full: int) -> dict[str, Any]:
    if not frames:
        return {"count": 0}
    ordered = sorted(frames)
    stats: dict[str, Any] = {
        "count": len(ordered),
        "min_frames": ordered[0],
        "p5_frames": _nearest_rank(ordered, 0.05),
        "median_frames": _nearest_rank(ordered, 0.5),
        "max_frames": ordered[-1],
        "under_full_length": sum(1 for f in ordered if f < full),
    }
    for key in ("min", "p5", "median", "max"):
        stats[f"{key}_s"] = format_duration(stats[f"{key}_frames"], sample_rate)
    return stats


def _duplicates(
    rows: Sequence[ManifestRow],
) -> tuple[list[dict[str, Any]], list[list[str]], list[Issue], list[Issue]]:
    """Group rows by ``audio_sha256`` and apply the contract's duplicate rules.

    Returns ``(audio_groups, identical_file_groups, errors, warnings)``.
    """
    audio_groups: dict[str, list[ManifestRow]] = defaultdict(list)
    file_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        audio_groups[row.audio_sha256].append(row)
        file_groups[row.sha256].append(row.relative_path)
    groups: list[dict[str, Any]] = []
    errors: list[Issue] = []
    warnings: list[Issue] = []
    for digest, group in sorted(audio_groups.items()):
        if len(group) < 2:
            continue
        paths = sorted(r.relative_path for r in group)
        splits = sorted({r.split for r in group})
        labels = sorted({r.label for r in group})
        groups.append(
            {
                "audio_sha256": digest,
                "paths": paths,
                "splits": splits,
                "labels": labels,
                "speakers": sorted({r.speaker_id for r in group}),
            }
        )
        members = ", ".join(paths)
        if len(splits) > 1:
            errors += [Issue("duplicate_audio_across_splits", p, members) for p in paths]
        if len(labels) > 1:
            errors += [Issue("duplicate_audio_across_labels", p, members) for p in paths]
        if len(splits) == 1 and len(labels) == 1:
            warnings += [Issue("duplicate_audio_within_split_label", p, members) for p in paths]
    identical_files = sorted(sorted(g) for g in file_groups.values() if len(g) > 1)
    return groups, identical_files, errors, warnings


def _duplicate_summary(
    groups: Sequence[dict[str, Any]], identical_files: Sequence[list[str]]
) -> dict[str, Any]:
    single_split = {s: [g for g in groups if g["splits"] == [s]] for s in SPLITS}
    return {
        "audio_groups": list(groups[:MAX_ITEMS_PER_CODE]),
        "audio_group_count": len(groups),
        "audio_groups_cross_speaker": sum(1 for g in groups if len(g["speakers"]) > 1),
        "audio_groups_by_split": {s: len(single_split[s]) for s in SPLITS},
        "files_in_audio_groups_by_split": {
            s: sum(len(g["paths"]) for g in single_split[s]) for s in SPLITS
        },
        "identical_file_groups": list(identical_files[:MAX_ITEMS_PER_CODE]),
        "identical_file_group_count": len(identical_files),
    }


def validate_dataset(
    spec: DatasetSpec, extraction_dir: Path, archive_path: Path, *, workers: int | None = None
) -> ValidationResult:
    """Check ``extraction_dir`` (and ``archive_path``) against the contract.

    Collects every error and warning; writes nothing. ``rows`` is populated
    only if there are no errors.
    """
    errors: list[Issue] = []
    warnings: list[Issue] = []
    root = extraction_dir

    candidates, label_dir_files, ignored, scan_errors = _scan(root, spec)
    errors += scan_errors
    pinned_hashes, pinned_errors = _check_pinned_files(root, spec)
    errors += pinned_errors

    existing = {c[0] for c in candidates}
    validation, ignored_val, val_errors = _parse_split_list(
        root, spec.validation_list, spec, existing
    )
    test, ignored_test, test_errors = _parse_split_list(root, spec.testing_list, spec, existing)
    errors += val_errors + test_errors
    for rel in sorted(validation & test):
        errors.append(Issue("list_overlap", rel, "listed in both validation and test"))
    ignored["list_entries_other_labels"] = {"validation": ignored_val, "test": ignored_test}

    def split_of(rel: str) -> str:
        if rel in validation:
            return "validation"
        if rel in test:
            return "test"
        return "train"

    n_workers = workers or min(16, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=n_workers + 1) as pool:
        archive_future = pool.submit(_hash_archive, spec, archive_path)
        results = {
            r.relative_path: r for r in pool.map(lambda p: _inspect(root, p), label_dir_files)
        }
        archive_hashes, archive_errors, archive_count = archive_future.result()
    errors += archive_errors
    if not any(e.code == "archive_unreadable" for e in archive_errors):
        errors += _compare_with_archive(
            archive_hashes, {rel: r.sha256 for rel, r in results.items()}
        )

    label_index = {label: i for i, label in enumerate(spec.labels)}
    rows: list[ManifestRow] = []
    speakers: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    frames: dict[tuple[str, str], list[int]] = defaultdict(list)
    extra_chunks: Counter[str] = Counter()
    for rel, label, speaker, index in candidates:
        split = split_of(rel)
        speakers[speaker].add(split)
        counts[split][label] += 1
        result = results[rel]
        if result.error is not None:
            errors.append(result.error)
            continue
        audio = result.audio
        assert audio is not None and result.sha256 is not None and result.size_bytes is not None
        extra_chunks.update(audio.extra_chunks)
        file_issues = _audio_issues(spec, result)
        if file_issues:
            errors += file_issues
            continue
        assert audio.peak_abs is not None
        frames[(split, label)].append(audio.num_frames)
        if audio.peak_abs == 0:
            warnings.append(Issue("all_zero_audio", rel, "every sample is 0"))
        elif audio.peak_abs <= spec.near_silent_peak_abs:
            warnings.append(
                Issue(
                    "near_silent_audio",
                    rel,
                    f"peak_abs {audio.peak_abs} <= {spec.near_silent_peak_abs}",
                )
            )
        rows.append(
            ManifestRow(
                relative_path=rel,
                split=split,
                label=label,
                label_index=label_index[label],
                speaker_id=speaker,
                utterance_index=index,
                sha256=result.sha256,
                audio_sha256=audio.audio_sha256,
                size_bytes=result.size_bytes,
                sample_rate_hz=audio.sample_rate_hz,
                num_channels=audio.num_channels,
                sample_format=audio.sample_format,
                num_frames=audio.num_frames,
                duration_s=format_duration(audio.num_frames, audio.sample_rate_hz),
                peak_abs=audio.peak_abs,
                is_all_zero=audio.peak_abs == 0,
            )
        )

    present_labels = {c[1] for c in candidates}  # absent labels already have an error
    for split in SPLITS:
        for label in spec.labels:
            if counts[split][label] == 0 and label in present_labels:
                errors.append(Issue("split_label_empty", label, f"no files in split {split}"))

    leaking = {s: sorted(v) for s, v in speakers.items() if len(v) > 1}
    for speaker, splits in sorted(leaking.items()):
        errors.append(Issue("speaker_in_multiple_splits", speaker, ", ".join(splits)))

    duplicate_groups, identical_files, dup_errors, dup_warnings = _duplicates(rows)
    errors += dup_errors
    warnings += dup_warnings

    errors += check_rows(
        rows,
        spec.labels,
        sample_rate_hz=spec.sample_rate_hz,
        num_channels=spec.num_channels,
        sample_format=spec.sample_format,
        max_frames=spec.max_frames,
    )

    manifest_bytes = b""
    if not errors:
        manifest_bytes = encode_manifest(rows)
        digest = _sha256_bytes(manifest_bytes)
        expected = spec.expected_manifest_sha256
        if expected is not None and digest != expected:
            errors.append(
                Issue(
                    "manifest_checksum_mismatch",
                    f"{spec.manifest_name}.csv",
                    f"generated SHA-256 {digest}, pinned {expected}",
                )
            )
            manifest_bytes = b""

    errors = sorted(set(errors))
    warnings = sorted(set(warnings))
    passed = not errors
    by_split = {s: sum(counts[s].values()) for s in SPLITS}
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "dataset": {
            "name": spec.archive.name,
            "version": spec.archive.version,
            "archive_sha256": spec.archive.sha256,
            "manifest_name": spec.manifest_name,
        },
        "labels": list(spec.labels),
        "contract": {
            "sample_rate_hz": spec.sample_rate_hz,
            "num_channels": spec.num_channels,
            "sample_format": spec.sample_format,
            "min_frames": 1,
            "max_frames": spec.max_frames,
            "near_silent_peak_abs": spec.near_silent_peak_abs,
            "content_hash_protocol": CONTENT_HASH_PROTOCOL,
            "percentile_method": "nearest-rank",
        },
        "files": {
            "label_dir_files": len(label_dir_files),
            "candidates": len(candidates),
            "manifest_rows": len(rows) if passed else 0,
        },
        "counts": {
            "by_split": by_split,
            "by_split_label": {s: {lab: counts[s][lab] for lab in spec.labels} for s in SPLITS},
            "total": sum(by_split.values()),
        },
        "speakers": {
            "total": len(speakers),
            "by_split": {s: sum(1 for v in speakers.values() if s in v) for s in SPLITS},
            "in_multiple_splits": len(leaking),
        },
        "split_lists": {
            name: {
                "sha256": pinned_hashes.get(name),
                "entries_kept": len(kept),
                "entries_other_labels": other,
            }
            for name, kept, other in (
                (spec.validation_list, validation, ignored_val),
                (spec.testing_list, test, ignored_test),
            )
        },
        "durations": {
            s: {
                lab: _duration_stats(frames[(s, lab)], spec.sample_rate_hz, spec.max_frames)
                for lab in spec.labels
            }
            for s in SPLITS
        },
        "duplicates": _duplicate_summary(duplicate_groups, identical_files),
        "extra_wav_chunks": dict(sorted(extra_chunks.items())),
        "manifest": {
            "file": f"{spec.manifest_name}.csv",
            "sha256": _sha256_bytes(manifest_bytes) if passed else None,
            "pinned_sha256": spec.expected_manifest_sha256,
            "row_count": len(rows) if passed else 0,
        },
        "archive_crosscheck": {
            "archive_file": archive_path.name,
            "label_files_compared": archive_count,
        },
        "ignored": ignored,
        "errors": _issue_summary(errors),
        "warnings": _issue_summary(warnings),
    }
    return ValidationResult(
        spec=spec,
        rows=sorted(rows, key=lambda r: r.relative_path) if passed else [],
        errors=errors,
        warnings=warnings,
        report=report,
        manifest_bytes=manifest_bytes,
        list_sha256={
            "validation_list_sha256": pinned_hashes.get(spec.validation_list, ""),
            "testing_list_sha256": pinned_hashes.get(spec.testing_list, ""),
        },
    )


# --- outputs ------------------------------------------------------------------


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _markdown_report(report: Mapping[str, Any]) -> bytes:
    lines = [
        f"# Validation report: {report['dataset']['manifest_name']}",
        "",
        f"- Status: **{report['status']}**",
        f"- Archive SHA-256: `{report['dataset']['archive_sha256']}`",
        f"- Labels: {', '.join(report['labels'])}",
        f"- Errors: {report['errors']['total']}; warnings: {report['warnings']['total']}",
    ]
    if report["manifest"]["sha256"]:
        lines.append(
            f"- Manifest: `{report['manifest']['file']}`, {report['manifest']['row_count']} rows, "
            f"SHA-256 `{report['manifest']['sha256']}`"
        )
    lines += ["", "## Counts", "", "| Label | " + " | ".join(SPLITS) + " |"]
    lines.append("| --- |" + " ---: |" * len(SPLITS))
    by_split_label = report["counts"]["by_split_label"]
    for label in report["labels"]:
        cells = " | ".join(str(by_split_label[s][label]) for s in SPLITS)
        lines.append(f"| {label} | {cells} |")
    totals = " | ".join(str(report["counts"]["by_split"][s]) for s in SPLITS)
    lines.append(f"| **total** | {totals} |")
    speakers = report["speakers"]
    lines += [
        "",
        f"Speakers: {speakers['total']} total; per split "
        + ", ".join(f"{s} {speakers['by_split'][s]}" for s in SPLITS)
        + f"; in more than one split: {speakers['in_multiple_splits']}.",
        "",
        "## Durations (frames; nearest-rank percentiles)",
        "",
        "| Split | Label | Count | Min | P5 | Median | Max | Under full length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        for label in report["labels"]:
            st = report["durations"][split][label]
            if st["count"]:
                lines.append(
                    f"| {split} | {label} | {st['count']} | {st['min_frames']} | "
                    f"{st['p5_frames']} | {st['median_frames']} | {st['max_frames']} | "
                    f"{st['under_full_length']} |"
                )
    for kind in ("errors", "warnings"):
        summary = report[kind]
        lines += ["", f"## {kind.capitalize()} ({summary['total']})", ""]
        if not summary["total"]:
            lines.append("None.")
            continue
        for code, count in summary["by_code"].items():
            lines.append(f"- `{code}`: {count}")
        lines += ["", "| Code | Path | Detail |", "| --- | --- | --- |"]
        shown: Counter[str] = Counter()
        for item in summary["items"]:
            if shown[item["code"]] >= MARKDOWN_ITEMS_PER_CODE:
                continue
            shown[item["code"]] += 1
            detail = item["detail"].replace("|", "\\|")
            lines.append(f"| `{item['code']}` | `{item['path']}` | {detail} |")
        hidden = summary["total"] - sum(shown.values())
        if hidden:
            lines.append(f"\n({hidden} further items are listed in the JSON report only.)")
    dup = report["duplicates"]
    lines += [
        "",
        "## Duplicate audio content",
        "",
        f"{dup['audio_group_count']} groups of identical audio (`audio_sha256`); "
        f"{dup['audio_groups_cross_speaker']} span more than one speaker; "
        f"{dup['identical_file_group_count']} groups of byte-identical files.",
        "",
        "| Split | Groups | Files in groups |",
        "| --- | ---: | ---: |",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {dup['audio_groups_by_split'][split]} | "
            f"{dup['files_in_audio_groups_by_split'][split]} |"
        )
    ignored = report["ignored"]
    lines += [
        "",
        "## Ignored",
        "",
        "- Directories (file counts): "
        + ", ".join(f"`{k}` {v}" for k, v in ignored["directories"].items()),
        "- Top-level files: " + (", ".join(f"`{f}`" for f in ignored["top_level_files"]) or "none"),
        "- Split-list entries for other labels: "
        + ", ".join(f"{k} {v}" for k, v in ignored["list_entries_other_labels"].items()),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _check_extraction_record(spec: DatasetSpec, extraction_dir: Path) -> None:
    record_path = extraction_dir / EXTRACTION_RECORD
    if not record_path.is_file():
        raise DatasetValidationError(
            f"{extraction_dir} has no {EXTRACTION_RECORD}; run acquisition first"
        )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(f"{record_path} is unreadable: {exc}") from exc
    if not isinstance(record, dict) or record.get("archive_sha256") != spec.archive.sha256:
        raise DatasetValidationError(
            f"{record_path} records archive {record.get('archive_sha256')}, "
            f"expected {spec.archive.sha256}"
        )


def build_manifest(
    spec: DatasetSpec,
    extraction_dir: Path,
    archive_path: Path,
    *,
    manifest_dir: Path,
    report_dir: Path,
    workers: int | None = None,
) -> BuildResult:
    """Validate the dataset and write manifest, sidecar, and reports (fail closed)."""
    source = run_dir.current_source_revision()
    _check_extraction_record(spec, extraction_dir)
    if not archive_path.is_file():
        raise DatasetValidationError(f"archive not found: {archive_path}")

    manifest_path = manifest_dir / f"{spec.manifest_name}.csv"
    sidecar_path = manifest_dir / f"{spec.manifest_name}.json"
    report_path = report_dir / f"{spec.manifest_name}_validation.json"
    report_md_path = report_dir / f"{spec.manifest_name}_validation.md"

    result = validate_dataset(spec, extraction_dir, archive_path, workers=workers)
    report = result.report
    if not result.passed:
        for stale in (manifest_path, sidecar_path):
            stale.unlink(missing_ok=True)
        _write_atomic(report_path, _json_bytes(report))
        _write_atomic(report_md_path, _markdown_report(report))
        codes = ", ".join(f"{c} ({n})" for c, n in report["errors"]["by_code"].items())
        raise DatasetValidationError(
            f"{len(result.errors)} contract violation(s): {codes}; see {report_path}"
        )

    manifest_bytes = result.manifest_bytes
    manifest_sha = result.manifest_sha256
    report_bytes = _json_bytes(report)
    _write_atomic(manifest_path, manifest_bytes)
    if read_manifest(manifest_path, expected_sha256=manifest_sha) != result.rows:
        manifest_path.unlink(missing_ok=True)
        raise DatasetValidationError("manifest round-trip check failed")
    _write_atomic(report_path, report_bytes)
    _write_atomic(report_md_path, _markdown_report(report))

    archive = spec.archive
    sidecar: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "row_count": len(result.rows),
        "columns": list(COLUMNS),
        "dataset_name": archive.name,
        "dataset_version": archive.version,
        "source_url": archive.source_url,
        "retrieval_url": archive.retrieval_url,
        "archive_filename": archive.filename,
        "archive_sha256": archive.sha256,
        "archive_size_bytes": archive.size_bytes,
        "license_file_sha256": archive.pinned_members[spec.license_file],
        **dict(spec.metadata),
        "labels": list(spec.labels),
        **result.list_sha256,
        "counts": report["counts"],
        "speakers": report["speakers"],
        "warnings": report["warnings"],
        "ignored": report["ignored"],
        "content_hash_protocol": CONTENT_HASH_PROTOCOL,
        "validation_report_file": report_path.name,
        "validation_report_sha256": _sha256_bytes(report_bytes),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"commit": source.commit, "dirty": source.dirty},
    }
    _write_atomic(sidecar_path, _json_bytes(sidecar))
    return BuildResult(
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
        report_path=report_path,
        report_markdown_path=report_md_path,
        manifest_sha256=manifest_sha,
        row_count=len(result.rows),
        result=result,
    )
