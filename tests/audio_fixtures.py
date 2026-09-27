"""Synthetic WAV files and toy dataset extractions for data-validation tests."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.data.acquisition import EXTRACTION_RECORD, ArchiveSpec
from robust_streaming_ml.data.validation import DatasetSpec

TOY_LABELS = ("yes", "no")
TOY_SPEAKERS = {
    "train": ("aaaaaaaa", "bbbbbbbb"),
    "validation": ("cccccccc",),
    "test": ("dddddddd",),
}


def tone(num_frames: int, seed: int) -> npt.NDArray[np.int16]:
    """Deterministic, seed-specific samples with peak well above near-silence."""
    n = np.arange(num_frames, dtype=np.int64)
    return (((n * (seed + 3)) % 2000) - 1000).astype(np.int16)


def wav_bytes(
    samples: npt.NDArray[np.integer] | bytes,
    *,
    rate: int = 16000,
    channels: int = 1,
    bits: int = 16,
    format_tag: int = 1,
    byte_rate: int | None = None,
    block_align: int | None = None,
    fmt_size: int = 16,
    chunks_before: Sequence[tuple[bytes, bytes]] = (),
    chunks_after: Sequence[tuple[bytes, bytes]] = (),
    order: str = "fmt,data",
    riff_size_delta: int = 0,
    trailing: bytes = b"",
    data_size: int | None = None,
) -> bytes:
    """Build a WAVE file; every structural field can be overridden to make it malformed."""
    data = samples if isinstance(samples, bytes) else samples.astype("<i2").tobytes()
    align = block_align if block_align is not None else channels * ((bits + 7) // 8)
    rate_bytes = byte_rate if byte_rate is not None else rate * align
    fmt_body = struct.pack("<HHIIHH", format_tag, channels, rate, rate_bytes, align, bits)
    fmt_body = fmt_body[:fmt_size] + b"\0" * max(0, fmt_size - 16)

    def chunk(cid: bytes, body: bytes, size: int | None = None) -> bytes:
        declared = len(body) if size is None else size
        return struct.pack("<4sI", cid, declared) + body + (b"\0" if len(body) % 2 else b"")

    parts = {"fmt": chunk(b"fmt ", fmt_body), "data": chunk(b"data", data, data_size)}
    body = b"".join(chunk(c, b) for c, b in chunks_before)
    body += b"".join(parts[name] for name in order.split(",") if name)
    body += b"".join(chunk(c, b) for c, b in chunks_after)
    riff = b"WAVE" + body
    return struct.pack("<4sI", b"RIFF", len(riff) + riff_size_delta) + riff + trailing


def default_files(num_frames: int = 1600) -> dict[str, bytes]:
    """Two labels; train speakers a (2 utterances) and b, validation c, test d."""
    files: dict[str, bytes] = {}
    seed = 0
    for label in TOY_LABELS:
        for speakers in TOY_SPEAKERS.values():
            for speaker in speakers:
                count = 2 if speaker == "aaaaaaaa" else 1
                for index in range(count):
                    seed += 1
                    files[f"{label}/{speaker}_nohash_{index}.wav"] = wav_bytes(
                        tone(num_frames, seed)
                    )
    return files


def default_lists(files: Mapping[str, bytes]) -> tuple[list[str], list[str]]:
    validation = sorted(p for p in files if "cccccccc" in p)
    testing = sorted(p for p in files if "dddddddd" in p)
    # Entries for labels outside the subset are ignored by the contract.
    return [*validation, "bed/eeeeeeee_nohash_0.wav"], [*testing, "bed/ffffffff_nohash_0.wav"]


@dataclass
class Toy:
    spec: DatasetSpec
    root: Path
    archive: Path


def build_toy(
    tmp_path: Path,
    *,
    files: Mapping[str, bytes] | None = None,
    validation: Sequence[str] | None = None,
    testing: Sequence[str] | None = None,
    validation_text: str | None = None,
    testing_text: str | None = None,
    extra_top: Mapping[str, bytes] | None = None,
    **spec_overrides: object,
) -> Toy:
    """Write a toy extraction and a matching pinned archive; return the spec and paths."""
    files = dict(default_files() if files is None else files)
    default_val, default_test = default_lists(files)
    val_lines = default_val if validation is None else list(validation)
    test_lines = default_test if testing is None else list(testing)
    top: dict[str, bytes] = {
        "LICENSE": b"toy license\n",
        "README.md": b"toy readme\n",
        "validation_list.txt": (
            validation_text if validation_text is not None else "\n".join(val_lines) + "\n"
        ).encode("utf-8"),
        "testing_list.txt": (
            testing_text if testing_text is not None else "\n".join(test_lines) + "\n"
        ).encode("utf-8"),
        ".DS_Store": b"junk",
        "_background_noise_/noise.wav": wav_bytes(tone(100, 99)),
        "bed/eeeeeeee_nohash_0.wav": wav_bytes(tone(100, 98)),
    }
    top.update(extra_top or {})
    content = {**top, **files}

    root = tmp_path / "raw" / "toy_v1"
    for rel, data in content.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = tmp_path / "raw" / "toy_v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for rel in sorted(content):
            info = tarfile.TarInfo(f"./{rel}")
            info.size = len(content[rel])
            tf.addfile(info, io.BytesIO(content[rel]))
    archive_bytes = archive.read_bytes()
    pinned = {
        name: hashlib.sha256(top[name]).hexdigest()
        for name in ("LICENSE", "README.md", "validation_list.txt", "testing_list.txt")
    }
    archive_spec = ArchiveSpec(
        name="toy",
        version="v1",
        filename=archive.name,
        source_url="http://example.invalid/toy_v1.tar.gz",
        retrieval_url="https://example.invalid/toy_v1.tar.gz",
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        size_bytes=len(archive_bytes),
        pinned_members=pinned,
    )
    (root / EXTRACTION_RECORD).write_text(
        json.dumps({"archive_sha256": archive_spec.sha256}), encoding="utf-8"
    )
    spec = DatasetSpec(
        archive=archive_spec,
        labels=TOY_LABELS,
        manifest_name="toy_v1_kw2",
        metadata={"license_name": "Toy License", "attribution_text": "Toy attribution"},
        **spec_overrides,  # type: ignore[arg-type]
    )
    return Toy(spec=spec, root=root, archive=archive)
