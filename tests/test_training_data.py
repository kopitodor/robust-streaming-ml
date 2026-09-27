from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from robust_streaming_ml.data.wav import parse_wav
from robust_streaming_ml.training.config import AugmentationSection
from robust_streaming_ml.training.data import (
    DataCompatibilityError,
    DataIntegrityError,
    EpochSampler,
    WaveformDataset,
    load_split_audio,
    load_validated_manifest,
    make_loader,
    seed_worker,
    time_shift,
)
from tests.audio_fixtures import tone, wav_bytes
from tests.training_fixtures import ToyTraining, make_toy_training


@pytest.fixture
def toy(tmp_path: Path, package_repo: Path) -> ToyTraining:
    return make_toy_training(tmp_path)


# --- manifest compatibility gate -----------------------------------------------


def test_validated_manifest_loads(toy: ToyTraining) -> None:
    manifest = load_validated_manifest(toy.config(), toy.spec)
    assert manifest.sha256 == toy.built.manifest_sha256
    assert manifest.labels == ("yes", "no")
    assert manifest.rows_by_split == {"train": 14, "validation": 2, "test": 2}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"data.dataset": "other"}, "config dataset"),
        ({"data.manifest_sha256": "0" * 64}, "pinned"),
        ({"data.labels": ["no", "yes"]}, "config labels"),
        ({"window.sample_rate_hz": 22050, "features.f_max_hz": 8000.0}, "sample rate"),
        ({"window.num_samples": 8000}, "cropping is not allowed"),
    ],
)
def test_config_incompatibilities_are_rejected(
    toy: ToyTraining, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(DataCompatibilityError, match=message):
        load_validated_manifest(toy.config(**overrides), toy.spec)


def test_unpinned_dataset_spec_is_rejected(toy: ToyTraining) -> None:
    unpinned = dataclasses.replace(toy.spec, expected_manifest_sha256=None)
    with pytest.raises(DataCompatibilityError, match="no pinned manifest"):
        load_validated_manifest(toy.config(), unpinned)


def test_failed_validation_report_is_rejected(toy: ToyTraining) -> None:
    report = json.loads(toy.built.report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    toy.built.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataCompatibilityError, match="status is 'failed'"):
        load_validated_manifest(toy.config(), toy.spec)


def test_report_for_another_manifest_is_rejected(toy: ToyTraining) -> None:
    report = json.loads(toy.built.report_path.read_text(encoding="utf-8"))
    report["manifest"]["sha256"] = "0" * 64
    toy.built.report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataCompatibilityError, match="does not describe this manifest"):
        load_validated_manifest(toy.config(), toy.spec)


def test_report_changed_after_the_sidecar_was_written_is_rejected(toy: ToyTraining) -> None:
    text = toy.built.report_path.read_text(encoding="utf-8")
    toy.built.report_path.write_text(text + " ", encoding="utf-8")
    with pytest.raises(DataCompatibilityError, match="sidecar does not match"):
        load_validated_manifest(toy.config(), toy.spec)


def test_missing_sidecar_is_rejected(toy: ToyTraining) -> None:
    toy.built.sidecar_path.unlink()
    with pytest.raises(DataCompatibilityError, match="sidecar unreadable"):
        load_validated_manifest(toy.config(), toy.spec)


def test_modified_manifest_is_rejected(toy: ToyTraining) -> None:
    data = toy.built.manifest_path.read_bytes().replace(b",train,", b",test,", 1)
    toy.built.manifest_path.write_bytes(data)
    with pytest.raises(DataCompatibilityError, match="SHA-256"):
        load_validated_manifest(toy.config(), toy.spec)


# --- audio loading --------------------------------------------------------------


def test_split_audio_is_windowed_verified_and_labelled(toy: ToyTraining) -> None:
    config = toy.config()
    manifest = load_validated_manifest(config, toy.spec)
    audio = load_split_audio(manifest, "train", config)
    rows = manifest.rows_for("train")
    assert audio.waveforms.shape == (14, 16000) and audio.waveforms.dtype == torch.int16
    assert audio.relative_paths == tuple(r.relative_path for r in rows)
    assert audio.labels.tolist() == [r.label_index for r in rows]
    for i, row in enumerate(rows):
        original = parse_wav((toy.toy.root / row.relative_path).read_bytes()).samples_int16()
        np.testing.assert_array_equal(audio.waveforms[i, : row.num_frames].numpy(), original)
        assert not audio.waveforms[i, row.num_frames :].any()


def test_test_split_cannot_be_loaded(toy: ToyTraining) -> None:
    config = toy.config()
    manifest = load_validated_manifest(config, toy.spec)
    with pytest.raises(DataCompatibilityError, match="held out"):
        load_split_audio(manifest, "test", config)


def test_training_splits_never_open_test_files(
    toy: ToyTraining, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = toy.config()
    manifest = load_validated_manifest(config, toy.spec)
    opened: list[str] = []
    original = Path.read_bytes

    def tracking(self: Path) -> bytes:
        opened.append(self.as_posix())
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracking)
    load_split_audio(manifest, "train", config)
    load_split_audio(manifest, "validation", config)
    test_paths = {r.relative_path for r in manifest.rows_for("test")}
    assert test_paths and not any(p.endswith(t) for p in opened for t in test_paths)


def test_audio_changed_after_validation_is_rejected(toy: ToyTraining) -> None:
    config = toy.config()
    manifest = load_validated_manifest(config, toy.spec)
    row = manifest.rows_for("train")[0]
    (toy.toy.root / row.relative_path).write_bytes(wav_bytes(tone(row.num_frames, 999)))
    with pytest.raises(DataIntegrityError, match="SHA-256 differs"):
        load_split_audio(manifest, "train", config)


def test_audio_that_does_not_match_its_row_is_rejected(toy: ToyTraining) -> None:
    config = toy.config()
    manifest = load_validated_manifest(config, toy.spec)
    row = manifest.rows_for("train")[0]
    bad = dataclasses.replace(row, num_frames=row.num_frames - 1)
    with pytest.raises(DataIntegrityError, match="!= manifest"):
        load_split_audio(dataclasses.replace(manifest, rows=(bad,)), "train", config)


# --- augmentation, sampling, loaders ---------------------------------------------


def test_time_shift() -> None:
    x = torch.arange(1, 6, dtype=torch.int16)
    assert time_shift(x, 2).tolist() == [0, 0, 1, 2, 3]
    assert time_shift(x, -2).tolist() == [3, 4, 5, 0, 0]
    assert time_shift(x, 0).tolist() == [1, 2, 3, 4, 5]
    assert time_shift(x, 5).tolist() == [0, 0, 0, 0, 0]
    assert time_shift(x, -9).tolist() == [0, 0, 0, 0, 0]


def _audio(toy: ToyTraining) -> object:
    config = toy.config()
    return load_split_audio(load_validated_manifest(config, toy.spec), "train", config)


def test_augmentation_is_off_by_default_and_deterministic_when_on(toy: ToyTraining) -> None:
    audio = _audio(toy)
    plain = WaveformDataset(audio, seed=7, sample_rate_hz=16000)  # type: ignore[arg-type]
    assert torch.equal(plain[(1, 0)][0], audio.waveforms[0])  # type: ignore[attr-defined]
    aug = AugmentationSection(enabled=True, time_shift_max_ms=50.0)
    a = WaveformDataset(audio, seed=7, sample_rate_hz=16000, augmentation=aug)  # type: ignore[arg-type]
    b = WaveformDataset(audio, seed=7, sample_rate_hz=16000, augmentation=aug)  # type: ignore[arg-type]
    shifts = [a.shift_for(e, i) for e in (1, 2, 3) for i in range(len(a))]
    assert shifts == [b.shift_for(e, i) for e in (1, 2, 3) for i in range(len(b))]
    assert all(-800 <= s <= 800 for s in shifts) and len(set(shifts)) > 1
    assert torch.equal(a[(2, 3)][0], time_shift(audio.waveforms[3], a.shift_for(2, 3)))  # type: ignore[attr-defined]
    disabled = AugmentationSection(enabled=False)
    off = WaveformDataset(audio, seed=7, sample_rate_hz=16000, augmentation=disabled)  # type: ignore[arg-type]
    assert off.augmentation is None


def test_sampler_is_a_seeded_permutation_per_epoch() -> None:
    sampler = EpochSampler(20, seed=7, shuffle=True)
    sampler.set_epoch(1)
    first = list(sampler)
    assert sorted(i for _, i in first) == list(range(20)) and {e for e, _ in first} == {1}
    again = EpochSampler(20, seed=7, shuffle=True)
    again.set_epoch(1)
    assert list(again) == first
    sampler.set_epoch(2)
    assert [i for _, i in sampler] != [i for _, i in first]
    ordered = EpochSampler(5, seed=7, shuffle=False)
    assert [i for _, i in ordered] == [0, 1, 2, 3, 4]


def _batches(toy: ToyTraining, workers: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    audio = _audio(toy)
    aug = AugmentationSection(enabled=True, time_shift_max_ms=50.0)
    dataset = WaveformDataset(audio, seed=7, sample_rate_hz=16000, augmentation=aug)  # type: ignore[arg-type]
    sampler = EpochSampler(len(dataset), seed=7, shuffle=True)
    loader = make_loader(dataset, sampler, batch_size=4, num_workers=workers, seed=7)
    out = []
    for epoch in (1, 2):
        sampler.set_epoch(epoch)
        out += [(w.clone(), y.clone()) for w, y in loader]
    return out


def test_batches_do_not_depend_on_worker_count(toy: ToyTraining) -> None:
    serial = _batches(toy, 0)
    parallel = _batches(toy, 2)
    assert len(serial) == len(parallel) == 8
    for (w0, y0), (w1, y1) in zip(serial, parallel, strict=True):
        assert torch.equal(w0, w1) and torch.equal(y0, y1)


def test_seed_worker_seeds_python_and_numpy() -> None:
    torch.manual_seed(123)
    seed_worker(0)
    a = (random.random(), np.random.rand())  # noqa: NPY002
    torch.manual_seed(123)
    seed_worker(0)
    assert (random.random(), np.random.rand()) == a  # noqa: NPY002
