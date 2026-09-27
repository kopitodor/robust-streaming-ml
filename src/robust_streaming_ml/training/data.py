"""Manifest-backed training data: compatibility checks, audio loading, loaders.

Only a validated, pinned manifest is accepted (:func:`load_validated_manifest`),
and only the ``train`` and ``validation`` splits can be loaded: asking for
``test`` raises. Each audio file is re-verified against its manifest row
(file SHA-256, strict WAV parse, sample rate, channels, format, frame count)
before it is placed in the fixed window. Nothing is resampled, cropped, or
repaired; any mismatch raises :class:`DataIntegrityError`.

Determinism: :class:`EpochSampler` yields ``(epoch, index)`` pairs, shuffled
with ``derive_seed(seed, "shuffle", epoch)``. :class:`WaveformDataset` draws
augmentation from ``derive_seed(seed, "augment", epoch, index)``. The batches
therefore depend only on the seed and epoch, not on the number of DataLoader
workers. Workers are also seeded (:func:`seed_worker`) from the loader's
seeded generator.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from robust_streaming_ml.data.manifest import (
    SPLITS,
    ManifestError,
    ManifestRow,
    check_rows,
    read_manifest,
)
from robust_streaming_ml.data.validation import DatasetSpec
from robust_streaming_ml.data.wav import WavError, parse_wav
from robust_streaming_ml.features import WindowError, apply_window
from robust_streaming_ml.seed import derive_seed, torch_generator
from robust_streaming_ml.training.config import (
    AugmentationSection,
    DataSection,
    TrainingConfig,
    WindowSection,
)

__all__ = [
    "TRAINING_SPLITS",
    "DataCompatibilityError",
    "DataIntegrityError",
    "EpochSampler",
    "SplitAudio",
    "ValidatedManifest",
    "WaveformDataset",
    "check_manifest_inputs",
    "load_split_audio",
    "load_validated_manifest",
    "make_loader",
    "read_split_audio",
    "seed_worker",
    "time_shift",
]

TRAINING_SPLITS = ("train", "validation")


class DataCompatibilityError(RuntimeError):
    """The manifest, report, or config are not a compatible, validated set."""


class DataIntegrityError(RuntimeError):
    """An audio file does not match its manifest row."""


@dataclass(frozen=True)
class ValidatedManifest:
    path: Path
    sha256: str
    dataset: str
    labels: tuple[str, ...]
    rows: tuple[ManifestRow, ...]

    def rows_for(self, split: str) -> list[ManifestRow]:
        return [r for r in self.rows if r.split == split]

    @property
    def rows_by_split(self) -> dict[str, int]:
        return {s: sum(1 for r in self.rows if r.split == s) for s in SPLITS}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validated_manifest(config: TrainingConfig, spec: DatasetSpec) -> ValidatedManifest:
    """Check that config, manifest, sidecar, and validation report agree; load rows.

    Raises :class:`DataCompatibilityError` without side effects on any mismatch.
    """
    return check_manifest_inputs(config.data, config.window, spec)


def check_manifest_inputs(
    data: DataSection, window: WindowSection, spec: DatasetSpec
) -> ValidatedManifest:
    """The manifest compatibility gate shared by training and evaluation."""

    def fail(message: str) -> DataCompatibilityError:
        return DataCompatibilityError(f"{data.manifest_path}: {message}")

    if data.dataset != spec.manifest_name:
        raise fail(f"config dataset {data.dataset!r} != spec {spec.manifest_name!r}")
    if spec.expected_manifest_sha256 is None:
        raise fail("dataset spec has no pinned manifest SHA-256; validate and pin it first")
    if data.manifest_sha256 != spec.expected_manifest_sha256:
        raise fail(
            f"config manifest_sha256 {data.manifest_sha256} != pinned "
            f"{spec.expected_manifest_sha256}"
        )
    if list(spec.labels) != data.labels:
        raise fail(f"config labels {data.labels} != dataset labels {list(spec.labels)}")
    if window.sample_rate_hz != spec.sample_rate_hz:
        raise fail(f"window sample rate {window.sample_rate_hz} != dataset {spec.sample_rate_hz}")
    if window.num_samples < spec.max_frames:
        raise fail(
            f"window holds {window.num_samples} samples but clips may have "
            f"{spec.max_frames}; cropping is not allowed"
        )

    try:
        report = json.loads(data.validation_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise fail(f"validation report unreadable: {exc}") from exc
    if report.get("status") != "passed":
        raise fail(f"validation report status is {report.get('status')!r}, not 'passed'")
    if (report.get("manifest") or {}).get("sha256") != data.manifest_sha256:
        raise fail("validation report does not describe this manifest")
    if (report.get("dataset") or {}).get("archive_sha256") != spec.archive.sha256:
        raise fail("validation report is for a different archive")

    sidecar_path = data.manifest_path.with_suffix(".json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise fail(f"manifest sidecar unreadable: {exc}") from exc
    if sidecar.get("manifest_sha256") != data.manifest_sha256:
        raise fail("manifest sidecar describes a different manifest")
    if sidecar.get("validation_report_sha256") != _sha256_file(data.validation_report_path):
        raise fail("manifest sidecar does not match the validation report file")

    try:
        rows = read_manifest(data.manifest_path, expected_sha256=data.manifest_sha256)
    except (OSError, ManifestError) as exc:
        raise fail(str(exc)) from exc
    issues = check_rows(
        rows,
        spec.labels,
        sample_rate_hz=spec.sample_rate_hz,
        num_channels=spec.num_channels,
        sample_format=spec.sample_format,
        max_frames=spec.max_frames,
    )
    if issues:
        raise fail(f"{len(issues)} row check issue(s), first: {issues[0]}")
    counts = {s: sum(1 for r in rows if r.split == s) for s in SPLITS}
    empty = [s for s, n in counts.items() if n == 0]
    if empty:
        raise fail(f"splits with no rows: {empty}")
    return ValidatedManifest(
        path=data.manifest_path,
        sha256=data.manifest_sha256,
        dataset=data.dataset,
        labels=tuple(data.labels),
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class SplitAudio:
    """Windowed int16 waveforms ``[N, num_samples]`` and labels ``[N]`` in manifest order."""

    split: str
    waveforms: torch.Tensor
    labels: torch.Tensor
    relative_paths: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.relative_paths)


def _read_row(audio_root: Path, row: ManifestRow, window: WindowSection) -> np.ndarray:
    path = audio_root / row.relative_path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise DataIntegrityError(f"{row.relative_path}: file SHA-256 differs from the manifest")
    try:
        wav = parse_wav(data)
    except WavError as exc:
        raise DataIntegrityError(f"{row.relative_path}: {exc}") from exc
    observed = (wav.sample_rate_hz, wav.num_channels, wav.sample_format, wav.num_frames)
    expected = (row.sample_rate_hz, row.num_channels, row.sample_format, row.num_frames)
    if observed != expected:
        raise DataIntegrityError(f"{row.relative_path}: audio {observed} != manifest {expected}")
    if wav.sample_rate_hz != window.sample_rate_hz:
        raise DataIntegrityError(
            f"{row.relative_path}: {wav.sample_rate_hz} Hz; resampling is not allowed"
        )
    try:
        return apply_window(wav.samples_int16(), window)
    except WindowError as exc:
        raise DataIntegrityError(f"{row.relative_path}: {exc}") from exc


def load_split_audio(manifest: ValidatedManifest, split: str, config: TrainingConfig) -> SplitAudio:
    """Read, verify, and window every file of ``split`` (``train`` or ``validation``)."""
    if split not in TRAINING_SPLITS:
        raise DataCompatibilityError(
            f"split {split!r} cannot be loaded for training; the test split is held out"
        )
    return read_split_audio(manifest, split, config.data, config.window)


def read_split_audio(
    manifest: ValidatedManifest, split: str, data: DataSection, window: WindowSection
) -> SplitAudio:
    """Read, verify against the manifest, and window every file of ``split``.

    Callers enforce which splits they may use (training: train/validation;
    evaluation: validation/test).
    """
    if split not in SPLITS:
        raise DataCompatibilityError(f"unknown split {split!r}")
    rows = manifest.rows_for(split)
    label_index = {label: i for i, label in enumerate(manifest.labels)}
    root = data.audio_root
    with ThreadPoolExecutor(max_workers=data.load_workers) as pool:
        arrays = list(pool.map(lambda r: _read_row(root, r, window), rows))
    waveforms = torch.from_numpy(np.stack(arrays)) if arrays else torch.empty(0, 0)
    labels = torch.tensor([label_index[r.label] for r in rows], dtype=torch.int64)
    return SplitAudio(
        split=split,
        waveforms=waveforms,
        labels=labels,
        relative_paths=tuple(r.relative_path for r in rows),
    )


def time_shift(waveform: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift right (``shift > 0``) or left (``shift < 0``), filling with zeros."""
    out = torch.zeros_like(waveform)
    n = waveform.shape[-1]
    if shift >= n or -shift >= n:
        return out
    if shift > 0:
        out[..., shift:] = waveform[..., : n - shift]
    elif shift < 0:
        out[..., : n + shift] = waveform[..., -shift:]
    else:
        out.copy_(waveform)
    return out


class WaveformDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style dataset indexed by ``(epoch, index)``; returns ``(int16 waveform, label)``."""

    def __init__(
        self,
        audio: SplitAudio,
        *,
        seed: int,
        sample_rate_hz: int,
        augmentation: AugmentationSection | None = None,
    ) -> None:
        self.audio = audio
        self.seed = seed
        self.augmentation = augmentation if augmentation and augmentation.enabled else None
        self.max_shift = (
            round(self.augmentation.time_shift_max_ms * sample_rate_hz / 1000)
            if self.augmentation
            else 0
        )

    def __len__(self) -> int:
        return len(self.audio)

    def shift_for(self, epoch: int, index: int) -> int:
        generator = torch_generator(derive_seed(self.seed, "augment", epoch, index))
        return int(torch.randint(-self.max_shift, self.max_shift + 1, (1,), generator=generator))

    def __getitem__(self, key: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, index = key
        waveform = self.audio.waveforms[index]
        if self.augmentation is not None and self.max_shift > 0:
            waveform = time_shift(waveform, self.shift_for(epoch, index))
        return waveform, self.audio.labels[index]


class EpochSampler(Sampler[tuple[int, int]]):
    """Yields ``(epoch, index)``, shuffled with ``derive_seed(seed, "shuffle", epoch)``."""

    def __init__(self, size: int, *, seed: int, shuffle: bool) -> None:
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def order(self) -> list[int]:
        if not self.shuffle:
            return list(range(self.size))
        generator = torch_generator(derive_seed(self.seed, "shuffle", self.epoch))
        return [int(i) for i in torch.randperm(self.size, generator=generator)]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch = self.epoch
        return iter([(epoch, i) for i in self.order()])

    def __len__(self) -> int:
        return self.size


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed Python and NumPy from the worker's torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)  # noqa: NPY002 - seeding the worker's legacy global RNG


def make_loader(
    dataset: WaveformDataset,
    sampler: EpochSampler,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    """Deterministic DataLoader: seeded generator, seeded workers, fixed batch order."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers else None,
        generator=torch_generator(derive_seed(seed, "loader")),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
