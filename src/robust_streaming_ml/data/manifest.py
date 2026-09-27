"""Manifest rows: schema, deterministic CSV encoding, strict decoding, row checks.

The CSV format is fixed by docs/data-contract.md section 7: UTF-8, comma
separated, one header row, ``\\n`` line endings, rows sorted by
``relative_path`` (Python string order), and the column order in
:data:`COLUMNS`. Encoding the same rows always yields the same bytes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

__all__ = [
    "COLUMNS",
    "FILENAME_PATTERN",
    "MANIFEST_SCHEMA_VERSION",
    "SPLITS",
    "Issue",
    "ManifestError",
    "ManifestRow",
    "check_rows",
    "decode_manifest",
    "encode_manifest",
    "format_duration",
    "parse_filename",
    "read_manifest",
]

MANIFEST_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
# <8 lowercase hex speaker id>_nohash_<canonical non-negative integer>.wav
FILENAME_PATTERN = re.compile(r"^(?P<speaker>[0-9a-f]{8})_nohash_(?P<index>0|[1-9][0-9]*)\.wav$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest file does not match the schema."""


@dataclass(frozen=True, order=True)
class Issue:
    """One validation finding. ``code`` is stable; ``detail`` is human-readable."""

    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class ManifestRow:
    relative_path: str
    split: str
    label: str
    label_index: int
    speaker_id: str
    utterance_index: int
    sha256: str
    audio_sha256: str
    size_bytes: int
    sample_rate_hz: int
    num_channels: int
    sample_format: str
    num_frames: int
    duration_s: str
    peak_abs: int
    is_all_zero: bool


COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))


def format_duration(num_frames: int, sample_rate_hz: int) -> str:
    """``num_frames / sample_rate_hz`` rounded to 6 decimals, as fixed-point text."""
    return f"{num_frames / sample_rate_hz:.6f}"


def parse_filename(name: str) -> tuple[str, int] | None:
    """Return ``(speaker_id, utterance_index)`` for a canonical filename, else None."""
    match = FILENAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    return match.group("speaker"), int(match.group("index"))


def _cell(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def encode_manifest(rows: Iterable[ManifestRow]) -> bytes:
    """Encode rows as manifest CSV bytes, sorted by ``relative_path``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(COLUMNS)
    for row in sorted(rows, key=lambda r: r.relative_path):
        writer.writerow([_cell(v) for v in astuple(row)])
    return buffer.getvalue().encode("utf-8")


def _parse_int(text: str, column: str, line: int) -> int:
    if not re.fullmatch(r"0|-?[1-9][0-9]*", text):
        raise ManifestError(f"line {line}: {column} is not a canonical integer: {text!r}")
    return int(text)


def decode_manifest(data: bytes) -> list[ManifestRow]:
    """Strictly decode manifest CSV bytes; re-encoding the result must give ``data``."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"manifest is not UTF-8: {exc}") from exc
    if "\r" in text:
        raise ManifestError("manifest must use \\n line endings")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or tuple(header) != COLUMNS:
        raise ManifestError(f"manifest header must be {','.join(COLUMNS)}")
    rows: list[ManifestRow] = []
    int_columns = {f.name for f in fields(ManifestRow) if f.type in ("int", int)}
    for line, record in enumerate(reader, start=2):
        if len(record) != len(COLUMNS):
            raise ManifestError(f"line {line}: expected {len(COLUMNS)} fields, got {len(record)}")
        values: dict[str, object] = {}
        for column, text_value in zip(COLUMNS, record, strict=True):
            if column in int_columns:
                values[column] = _parse_int(text_value, column, line)
            elif column == "is_all_zero":
                if text_value not in ("true", "false"):
                    raise ManifestError(f"line {line}: is_all_zero must be true or false")
                values[column] = text_value == "true"
            else:
                values[column] = text_value
        rows.append(ManifestRow(**values))  # type: ignore[arg-type]
    if encode_manifest(rows) != data:
        raise ManifestError("manifest is not in canonical form (order, quoting, or formatting)")
    return rows


def read_manifest(path: Path, *, expected_sha256: str | None = None) -> list[ManifestRow]:
    """Read and strictly decode a manifest file, optionally checking its SHA-256."""
    data = path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256:
            raise ManifestError(f"{path}: SHA-256 {actual}, expected {expected_sha256}")
    return decode_manifest(data)


def check_rows(
    rows: Sequence[ManifestRow],
    labels: Sequence[str],
    *,
    sample_rate_hz: int,
    num_channels: int,
    sample_format: str,
    max_frames: int,
) -> list[Issue]:
    """Row-level integrity checks, independent of how the rows were produced.

    Checks for duplicate rows, duplicate relative paths (exact and
    case-insensitive), label/label_index/split validity, consistency of the
    path with the label, speaker id, and utterance index, hash formats, audio
    format fields, duration bounds, and duration/all-zero consistency.
    """
    issues: list[Issue] = []
    row_counts = Counter(rows)
    for row, count in sorted(row_counts.items(), key=lambda kv: kv[0].relative_path):
        if count > 1:
            issues.append(Issue("row_duplicate", row.relative_path, f"row appears {count} times"))
    path_counts = Counter(r.relative_path for r in rows)
    for path, count in sorted(path_counts.items()):
        if count > 1:
            issues.append(Issue("row_duplicate_path", path, f"relative_path appears {count} times"))
    folded: dict[str, set[str]] = {}
    for path in path_counts:
        folded.setdefault(path.casefold(), set()).add(path)
    for variants in folded.values():
        if len(variants) > 1:
            for path in sorted(variants):
                issues.append(
                    Issue("row_case_collision", path, f"differs only by case: {sorted(variants)}")
                )

    label_index = {label: i for i, label in enumerate(labels)}
    for row in sorted(set(rows), key=lambda r: r.relative_path):
        path = row.relative_path

        def bad(code: str, detail: str, path: str = path) -> None:
            issues.append(Issue(code, path, detail))

        if row.split not in SPLITS:
            bad("row_invalid_split", f"split {row.split!r} not in {SPLITS}")
        if row.label not in label_index:
            bad("row_invalid_label", f"label {row.label!r} not in labels")
        elif row.label_index != label_index[row.label]:
            bad("row_label_index_mismatch", f"label_index {row.label_index} for {row.label!r}")
        parts = path.split("/")
        parsed = parse_filename(parts[-1]) if len(parts) == 2 else None
        if parsed is None:
            bad("row_bad_path", "relative_path must be <label>/<speaker>_nohash_<n>.wav")
        else:
            if parts[0] != row.label:
                bad("row_path_label_mismatch", f"directory {parts[0]!r} != label {row.label!r}")
            if parsed != (row.speaker_id, row.utterance_index):
                bad(
                    "row_identity_mismatch",
                    f"path gives {parsed}, row has {(row.speaker_id, row.utterance_index)}",
                )
        for column in ("sha256", "audio_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(row, column)):
                bad("row_bad_hash", f"{column} is not 64 lowercase hex characters")
        if (row.sample_rate_hz, row.num_channels, row.sample_format) != (
            sample_rate_hz,
            num_channels,
            sample_format,
        ):
            bad(
                "row_bad_format",
                f"{row.sample_rate_hz} Hz, {row.num_channels} ch, {row.sample_format}",
            )
        if not 1 <= row.num_frames <= max_frames:
            bad("row_bad_num_frames", f"num_frames {row.num_frames} not in [1, {max_frames}]")
        elif row.duration_s != format_duration(row.num_frames, row.sample_rate_hz or 1):
            bad("row_bad_duration", f"duration_s {row.duration_s} inconsistent with num_frames")
        if row.size_bytes < 44 + 2 * max(row.num_frames, 0):
            bad("row_bad_size", f"size_bytes {row.size_bytes} too small for {row.num_frames}")
        if not 0 <= row.peak_abs <= 32768 or row.is_all_zero != (row.peak_abs == 0):
            bad("row_bad_peak", f"peak_abs {row.peak_abs}, is_all_zero {row.is_all_zero}")
    return sorted(issues)
