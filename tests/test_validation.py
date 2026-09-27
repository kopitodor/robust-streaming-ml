from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robust_streaming_ml import run_dir
from robust_streaming_ml.data.manifest import read_manifest
from robust_streaming_ml.data.validation import (
    DatasetValidationError,
    ValidationResult,
    build_manifest,
    validate_dataset,
)
from robust_streaming_ml.run_dir import ProvenanceError
from tests.audio_fixtures import Toy, build_toy, default_files, tone, wav_bytes
from tests.conftest import git

TRAIN_EXTRA = "yes/aaaaaaaa_nohash_7.wav"  # an additional train file (train speaker)


@pytest.fixture
def package_repo(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", git_repo)
    return git_repo


def validate(toy: Toy, **kwargs: Any) -> ValidationResult:
    return validate_dataset(toy.spec, toy.root, toy.archive, **kwargs)


def error_codes(toy: Toy) -> set[str]:
    return set(validate(toy).report["errors"]["by_code"])


def with_file(path: str, data: bytes) -> dict[str, bytes]:
    return {**default_files(), path: data}


def listed_validation() -> list[str]:
    """The default validation list, fixed before any files are added."""
    return sorted(p for p in default_files() if "cccccccc" in p)


def build(toy: Toy, out: Path) -> Any:
    return build_manifest(
        toy.spec, toy.root, toy.archive, manifest_dir=out / "manifests", report_dir=out / "reports"
    )


# --- passing dataset ----------------------------------------------------------


def test_valid_toy_dataset_passes(tmp_path: Path) -> None:
    result = validate(build_toy(tmp_path))
    report = result.report
    assert result.passed, report["errors"]
    assert report["status"] == "passed"
    assert len(result.rows) == 10
    assert report["counts"]["by_split_label"] == {
        "train": {"yes": 3, "no": 3},
        "validation": {"yes": 1, "no": 1},
        "test": {"yes": 1, "no": 1},
    }
    assert report["speakers"] == {
        "total": 4,
        "by_split": {"train": 2, "validation": 1, "test": 1},
        "in_multiple_splits": 0,
    }
    assert report["warnings"]["total"] == 0
    assert report["archive_crosscheck"]["label_files_compared"] == 10
    assert report["ignored"] == {
        "directories": {"_background_noise_": 1, "bed": 1},
        "top_level_files": [".DS_Store"],
        "list_entries_other_labels": {"validation": 1, "test": 1},
    }


def test_rows_carry_identity_audio_and_hashes(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    rows = {r.relative_path: r for r in validate(toy).rows}
    r = rows["no/aaaaaaaa_nohash_1.wav"]
    data = (toy.root / r.relative_path).read_bytes()
    assert (r.split, r.label, r.label_index, r.speaker_id, r.utterance_index) == (
        "train", "no", 1, "aaaaaaaa", 1,
    )  # fmt: skip
    assert r.sha256 == hashlib.sha256(data).hexdigest()
    assert r.audio_sha256 == hashlib.sha256(data[44:]).hexdigest()
    assert (r.size_bytes, r.num_frames, r.duration_s) == (len(data), 1600, "0.100000")
    assert (r.sample_rate_hz, r.num_channels, r.sample_format) == (16000, 1, "PCM_16")
    assert r.peak_abs == 1000 and r.is_all_zero is False


def test_train_is_everything_not_listed(tmp_path: Path) -> None:
    rows = validate(build_toy(tmp_path)).rows
    for r in rows:
        expected = {"cccccccc": "validation", "dddddddd": "test"}.get(r.speaker_id, "train")
        assert r.split == expected, r.relative_path


def test_list_whitespace_blank_lines_and_backslashes_are_normalised(tmp_path: Path) -> None:
    text = "\n  yes\\cccccccc_nohash_0.wav  \n\nno/cccccccc_nohash_0.wav\r\n"
    result = validate(build_toy(tmp_path, validation_text=text))
    assert result.passed, result.report["errors"]
    assert result.report["counts"]["by_split"]["validation"] == 2


def test_worker_count_does_not_change_the_report(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    assert validate(toy, workers=1).report == validate(toy, workers=8).report


# --- warnings (rows kept) -----------------------------------------------------


def test_all_zero_audio_is_a_warning_and_kept(tmp_path: Path) -> None:
    toy = build_toy(tmp_path, files=with_file(TRAIN_EXTRA, wav_bytes(np.zeros(800, np.int16))))
    result = validate(toy)
    assert result.passed
    assert result.report["warnings"]["by_code"] == {"all_zero_audio": 1}
    row = next(r for r in result.rows if r.relative_path == TRAIN_EXTRA)
    assert row.is_all_zero is True and row.peak_abs == 0


@pytest.mark.parametrize(("peak", "warned"), [(1, True), (32, True), (33, False)])
def test_near_silent_threshold(tmp_path: Path, peak: int, warned: bool) -> None:
    samples = np.zeros(800, np.int16)
    samples[10] = -peak
    result = validate(build_toy(tmp_path, files=with_file(TRAIN_EXTRA, wav_bytes(samples))))
    assert result.passed
    assert ("near_silent_audio" in result.report["warnings"]["by_code"]) is warned


def test_duplicate_audio_within_split_and_label_is_a_warning(tmp_path: Path) -> None:
    files = default_files()
    files[TRAIN_EXTRA] = files["yes/aaaaaaaa_nohash_0.wav"]
    result = validate(build_toy(tmp_path, files=files))
    assert result.passed
    assert result.report["warnings"]["by_code"] == {"duplicate_audio_within_split_label": 2}
    dup = result.report["duplicates"]
    assert dup["audio_group_count"] == 1
    assert dup["identical_file_group_count"] == 1
    assert dup["audio_groups_cross_speaker"] == 0
    assert dup["audio_groups_by_split"] == {"train": 1, "validation": 0, "test": 0}
    assert len(result.rows) == 11  # both rows kept


# --- layout and filename errors -----------------------------------------------


def test_missing_label_directory(tmp_path: Path) -> None:
    files = {p: d for p, d in default_files().items() if p.startswith("yes/")}
    assert "missing_label_dir" in error_codes(build_toy(tmp_path, files=files))


def test_empty_label_directory(tmp_path: Path) -> None:
    files = {p: d for p, d in default_files().items() if p.startswith("yes/")}
    toy = build_toy(tmp_path, files=files)
    (toy.root / "no").mkdir()
    assert "empty_label_dir" in error_codes(toy)


def test_subdirectory_in_label_directory(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / "yes" / "nested").mkdir()
    assert "unexpected_entry" in error_codes(toy)


@pytest.mark.parametrize(
    "name",
    ["notes.txt", "EEEEEEEE_nohash_0.wav", "aaaaaaaa_nohash_01.wav", "aaaaaaaa_nohash_9.WAV"],
)
def test_bad_filenames(tmp_path: Path, name: str) -> None:
    toy = build_toy(tmp_path, files=with_file(f"yes/{name}", wav_bytes(tone(100, 50))))
    assert "bad_filename" in error_codes(toy)


# --- pinned files and split lists ---------------------------------------------


def test_modified_pinned_file(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / "LICENSE").write_bytes(b"changed")
    assert "pinned_file_mismatch" in error_codes(toy)


def test_missing_pinned_file(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / "testing_list.txt").unlink()
    codes = error_codes(toy)
    assert "pinned_file_missing" in codes
    assert "speaker_in_multiple_splits" not in codes  # test files simply become train


@pytest.mark.parametrize(
    ("extra_line", "code"),
    [
        ("/yes/cccccccc_nohash_0.wav", "list_absolute_path"),
        ("C:/yes/cccccccc_nohash_0.wav", "list_absolute_path"),
        ("yes/../yes/cccccccc_nohash_0.wav", "list_noncanonical_path"),
        ("./yes/cccccccc_nohash_0.wav", "list_noncanonical_path"),
        ("yes//cccccccc_nohash_0.wav", "list_noncanonical_path"),
        ("yes/cccccccc_nohash_0.wav", "list_duplicate_entry"),
        ("yes/notes.txt", "list_bad_entry"),
        ("yes/sub/cccccccc_nohash_0.wav", "list_bad_entry"),
        ("yes/cccccccc_nohash_9.wav", "list_missing_file"),
        ("yes/dddddddd_nohash_0.wav", "list_overlap"),
    ],
)
def test_split_list_errors(tmp_path: Path, extra_line: str, code: str) -> None:
    files = default_files()
    validation = [*sorted(p for p in files if "cccccccc" in p), extra_line]
    assert code in error_codes(build_toy(tmp_path, validation=validation))


def test_list_not_utf8(tmp_path: Path) -> None:
    toy = build_toy(tmp_path, extra_top={"validation_list.txt": b"yes/\xff\xfe.wav\n"})
    assert "list_not_utf8" in error_codes(toy)


# --- audio format errors ------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (wav_bytes(tone(100, 50))[:-3], "wav_malformed"),
        (b"not a wav file", "wav_malformed"),
        (wav_bytes(tone(100, 50), rate=8000), "wav_sample_rate"),
        (wav_bytes(tone(200, 50), channels=2), "wav_channels"),
        (wav_bytes(b"\0\1\2" * 100, bits=24), "wav_sample_format"),
        (wav_bytes(b"\0\0\0\0" * 100, format_tag=3, bits=32), "wav_sample_format"),
        (wav_bytes(tone(100, 50), format_tag=0xFFFE), "wav_sample_format"),
        (wav_bytes(b""), "wav_zero_frames"),
        (wav_bytes(tone(16001, 50)), "wav_too_long"),
    ],
    ids=[
        "truncated",
        "not-riff",
        "rate-8000",
        "stereo",
        "pcm24",
        "float32",
        "extensible",
        "zero-frames",
        "16001-frames",
    ],
)
def test_audio_contract_violations(tmp_path: Path, data: bytes, code: str) -> None:
    result = validate(build_toy(tmp_path, files=with_file(TRAIN_EXTRA, data)))
    assert code in result.report["errors"]["by_code"]
    assert result.rows == []


def test_exactly_max_frames_is_valid(tmp_path: Path) -> None:
    assert validate(
        build_toy(tmp_path, files=with_file(TRAIN_EXTRA, wav_bytes(tone(16000, 50))))
    ).passed


def test_unreadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toy = build_toy(tmp_path)
    original = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        if self.name == "bbbbbbbb_nohash_0.wav" and self.parent.name == "yes":
            raise PermissionError(13, "Permission denied")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    assert "file_unreadable" in error_codes(toy)


# --- split integrity, duplicates, leakage -------------------------------------


def test_speaker_leakage_across_splits(tmp_path: Path) -> None:
    files = with_file("no/cccccccc_nohash_1.wav", wav_bytes(tone(90, 60)))  # not listed: train
    result = validate(build_toy(tmp_path, files=files, validation=listed_validation()))
    assert "speaker_in_multiple_splits" in result.report["errors"]["by_code"]
    assert result.report["speakers"]["in_multiple_splits"] == 1


def test_label_missing_from_a_split(tmp_path: Path) -> None:
    files = {p: d for p, d in default_files().items() if p != "no/dddddddd_nohash_0.wav"}
    assert "split_label_empty" in error_codes(build_toy(tmp_path, files=files))


def test_identical_file_across_splits(tmp_path: Path) -> None:
    files = default_files()
    files["yes/cccccccc_nohash_0.wav"] = files["yes/aaaaaaaa_nohash_0.wav"]
    assert "duplicate_audio_across_splits" in error_codes(build_toy(tmp_path, files=files))


def test_same_audio_different_header_across_splits(tmp_path: Path) -> None:
    files = default_files()
    samples = tone(1600, 1)  # the samples of yes/aaaaaaaa_nohash_0.wav
    files["yes/cccccccc_nohash_0.wav"] = wav_bytes(samples, chunks_after=[(b"LIST", b"INFO")])
    result = validate(build_toy(tmp_path, files=files))
    assert "duplicate_audio_across_splits" in result.report["errors"]["by_code"]
    assert result.report["duplicates"]["identical_file_group_count"] == 0
    assert result.report["extra_wav_chunks"] == {"LIST": 1}


def test_duplicate_audio_across_labels(tmp_path: Path) -> None:
    files = default_files()
    files["no/bbbbbbbb_nohash_0.wav"] = files["yes/bbbbbbbb_nohash_0.wav"]
    codes = error_codes(build_toy(tmp_path, files=files))
    assert "duplicate_audio_across_labels" in codes
    assert "duplicate_audio_across_splits" not in codes


def test_multiple_violations_are_all_reported(tmp_path: Path) -> None:
    files = with_file(TRAIN_EXTRA, wav_bytes(tone(100, 50), rate=8000))
    files["no/cccccccc_nohash_1.wav"] = wav_bytes(tone(90, 60))
    codes = error_codes(build_toy(tmp_path, files=files, validation=listed_validation()))
    assert {"wav_sample_rate", "speaker_in_multiple_splits"} <= codes


# --- archive cross-check ------------------------------------------------------


def test_extraction_file_modified_after_extraction(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    path = toy.root / "yes" / "bbbbbbbb_nohash_0.wav"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01  # still a valid WAV, different content
    path.write_bytes(bytes(data))
    assert error_codes(toy) == {"extraction_modified_file"}


def test_extraction_file_missing(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / "yes" / "bbbbbbbb_nohash_0.wav").unlink()
    assert error_codes(toy) == {"extraction_missing_file"}


def test_extraction_has_extra_file(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / TRAIN_EXTRA).write_bytes(wav_bytes(tone(100, 50)))
    assert error_codes(toy) == {"extraction_extra_file"}


def test_archive_checksum_mismatch(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    archive = dataclasses.replace(toy.spec.archive, sha256="0" * 64)
    toy.spec = dataclasses.replace(toy.spec, archive=archive)
    assert error_codes(toy) == {"archive_checksum_mismatch"}


def test_archive_unreadable(tmp_path: Path) -> None:
    toy = build_toy(tmp_path)
    toy.archive.write_bytes(b"not a gzip file")
    assert "archive_unreadable" in error_codes(toy)


# --- build_manifest outputs ---------------------------------------------------


def test_build_writes_manifest_sidecar_and_reports(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    built = build(toy, tmp_path / "out")
    data = built.manifest_path.read_bytes()
    assert built.manifest_sha256 == hashlib.sha256(data).hexdigest()
    rows = read_manifest(built.manifest_path, expected_sha256=built.manifest_sha256)
    assert rows == built.result.rows and len(rows) == 10

    sidecar = json.loads(built.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["manifest_sha256"] == built.manifest_sha256
    assert sidecar["row_count"] == 10
    assert sidecar["manifest_schema_version"] == 1
    assert sidecar["archive_sha256"] == toy.spec.archive.sha256
    assert sidecar["license_file_sha256"] == toy.spec.archive.pinned_members["LICENSE"]
    assert sidecar["license_name"] == "Toy License"
    assert sidecar["labels"] == ["yes", "no"]
    assert (
        sidecar["validation_list_sha256"] == toy.spec.archive.pinned_members["validation_list.txt"]
    )
    assert sidecar["testing_list_sha256"] == toy.spec.archive.pinned_members["testing_list.txt"]
    assert sidecar["source"] == {"commit": git(package_repo, "rev-parse", "HEAD"), "dirty": False}
    report_bytes = built.report_path.read_bytes()
    assert sidecar["validation_report_sha256"] == hashlib.sha256(report_bytes).hexdigest()

    report = json.loads(report_bytes)
    assert report["status"] == "passed"
    assert report["manifest"] == {
        "file": built.manifest_path.name,
        "sha256": built.manifest_sha256,
        "pinned_sha256": None,
        "row_count": 10,
    }
    markdown = built.report_markdown_path.read_text(encoding="utf-8")
    assert "Status: **passed**" in markdown and built.manifest_sha256 in markdown


def test_build_is_deterministic(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    first = build(toy, tmp_path / "a")
    second = build(toy, tmp_path / "b")
    for attr in ("manifest_path", "report_path", "report_markdown_path"):
        assert getattr(first, attr).read_bytes() == getattr(second, attr).read_bytes(), attr
    a = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    b = json.loads(second.sidecar_path.read_text(encoding="utf-8"))
    assert a.pop("created_at_utc") != "" and b.pop("created_at_utc") != ""
    assert a == b


def test_build_fails_closed_and_removes_stale_outputs(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    out = tmp_path / "out"
    built = build(toy, out)
    (toy.root / "no" / "cccccccc_nohash_1.wav").write_bytes(wav_bytes(tone(90, 60)))
    with pytest.raises(DatasetValidationError, match="contract violation"):
        build(toy, out)
    assert not built.manifest_path.exists()
    assert not built.sidecar_path.exists()
    report = json.loads(built.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["manifest"]["sha256"] is None
    assert {"speaker_in_multiple_splits", "extraction_extra_file"} <= set(
        report["errors"]["by_code"]
    )
    assert "Status: **failed**" in built.report_markdown_path.read_text(encoding="utf-8")


def test_build_requires_completed_extraction_record(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    (toy.root / ".acquisition.json").write_text('{"archive_sha256": "other"}', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="records archive"):
        build(toy, tmp_path / "out")
    (toy.root / ".acquisition.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="unreadable"):
        build(toy, tmp_path / "out")
    (toy.root / ".acquisition.json").unlink()
    with pytest.raises(DatasetValidationError, match="run acquisition first"):
        build(toy, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_build_requires_archive(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    toy.archive.unlink()
    with pytest.raises(DatasetValidationError, match="archive not found"):
        build(toy, tmp_path / "out")


def test_build_requires_source_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", tmp_path)
    toy = build_toy(tmp_path)
    with pytest.raises(ProvenanceError):
        build(toy, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_pinned_manifest_checksum_is_enforced(tmp_path: Path, package_repo: Path) -> None:
    toy = build_toy(tmp_path)
    digest = validate(toy).manifest_sha256
    pinned = dataclasses.replace(toy.spec, expected_manifest_sha256=digest)
    assert validate_dataset(pinned, toy.root, toy.archive).passed

    wrong = dataclasses.replace(toy.spec, expected_manifest_sha256="0" * 64)
    result = validate_dataset(wrong, toy.root, toy.archive)
    assert result.report["errors"]["by_code"] == {"manifest_checksum_mismatch": 1}
    assert result.rows == [] and result.manifest_bytes == b""
    toy.spec = wrong
    with pytest.raises(DatasetValidationError, match="manifest_checksum_mismatch"):
        build(toy, tmp_path / "out")
    assert not (tmp_path / "out" / "manifests" / "toy_v1_kw2.csv").exists()
