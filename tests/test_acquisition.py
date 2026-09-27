from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from robust_streaming_ml.data.acquisition import (
    EXTRACTION_RECORD,
    SPEECH_COMMANDS_V0_02,
    AcquisitionError,
    ArchiveSpec,
    acquire,
    download_archive,
    extract_archive,
    verify_archive,
)

FILES = {
    "LICENSE": b"license text\n",
    "validation_list.txt": b"yes/aa_nohash_0.wav\n",
    "yes/aa_nohash_0.wav": b"RIFF-fake-audio",
    "no/bb_nohash_1.wav": b"RIFF-other-audio",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_tar(path: Path, members: list[tarfile.TarInfo | tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for member in members:
            if isinstance(member, tarfile.TarInfo):
                tf.addfile(member)
            else:
                name, data = member
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return path


def spec_for(archive: Path, **overrides: object) -> ArchiveSpec:
    data = archive.read_bytes()
    spec = ArchiveSpec(
        name="toy",
        version="v1",
        filename=archive.name,
        source_url="http://example.invalid/toy.tar.gz",
        retrieval_url=archive.resolve().as_uri(),
        sha256=sha(data),
        size_bytes=len(data),
        pinned_members={"LICENSE": sha(FILES["LICENSE"])},
    )
    return dataclasses.replace(spec, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    dot_dir = tarfile.TarInfo(".")
    dot_dir.type = tarfile.DIRTYPE
    members: list[tarfile.TarInfo | tuple[str, bytes]] = [dot_dir]
    members += [(f"./{name}", data) for name, data in FILES.items()]
    return make_tar(src / "toy.tar.gz", members)


# --- pinned specification ---------------------------------------------------


def test_speech_commands_spec_matches_verification_record() -> None:
    record = (
        Path(__file__).resolve().parents[1] / "docs" / "datasets" / "speech_commands_v0.02.md"
    ).read_text(encoding="utf-8")
    spec = SPEECH_COMMANDS_V0_02
    assert spec.sha256 in record
    assert str(spec.size_bytes) in record
    assert spec.source_url in record
    assert spec.retrieval_url in record
    for digest in spec.pinned_members.values():
        assert digest in record
    assert spec.extract_dirname == "speech_commands_v0.02"


# --- download and verification ----------------------------------------------


def test_download_verifies_and_places_archive(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    calls: list[tuple[int, int]] = []
    path = download_archive(spec, tmp_path / "raw", progress=lambda r, t: calls.append((r, t)))
    assert path == tmp_path / "raw" / archive.name
    assert path.read_bytes() == archive.read_bytes()
    assert calls[-1] == (spec.size_bytes, spec.size_bytes)
    assert not (tmp_path / "raw" / f"{archive.name}.part").exists()


def test_download_hash_mismatch_leaves_nothing(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, sha256="0" * 64)
    with pytest.raises(AcquisitionError, match="SHA-256"):
        download_archive(spec, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_download_size_overrun_is_rejected(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, size_bytes=5)
    with pytest.raises(AcquisitionError, match="more than the expected"):
        download_archive(spec, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_download_failure_is_reported(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, retrieval_url=(tmp_path / "missing.tar.gz").as_uri())
    with pytest.raises(AcquisitionError, match="download from"):
        download_archive(spec, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_existing_valid_archive_is_reused_without_download(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, retrieval_url="http://unreachable.invalid/toy.tar.gz")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / archive.name).write_bytes(archive.read_bytes())
    assert download_archive(spec, raw) == raw / archive.name


def test_existing_mismatched_archive_is_not_overwritten(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / archive.name).write_bytes(b"tampered")
    with pytest.raises(AcquisitionError, match="size"):
        download_archive(spec, raw)
    assert (raw / archive.name).read_bytes() == b"tampered"


def test_verify_archive_detects_same_size_corruption(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    corrupted = tmp_path / "corrupt.tar.gz"
    data = bytearray(archive.read_bytes())
    data[-1] ^= 0xFF
    corrupted.write_bytes(bytes(data))
    with pytest.raises(AcquisitionError, match="SHA-256"):
        verify_archive(corrupted, spec)


# --- extraction -------------------------------------------------------------


def test_extracts_files_with_normalized_paths_and_record(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    dest = extract_archive(spec, archive, tmp_path / "out")
    assert dest == tmp_path / "out" / "toy_v1"
    for name, data in FILES.items():
        assert (dest / name).read_bytes() == data
    record = json.loads((dest / EXTRACTION_RECORD).read_text(encoding="utf-8"))
    assert record["archive_sha256"] == spec.sha256
    assert record["file_count"] == len(FILES)
    assert not list((tmp_path / "out").glob(".*extracting"))


def test_completed_extraction_is_reused(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    dest = extract_archive(spec, archive, tmp_path / "out")
    marker = dest / "yes" / "aa_nohash_0.wav"
    marker.write_bytes(b"unchanged-on-rerun")
    assert extract_archive(spec, archive, tmp_path / "out") == dest
    assert marker.read_bytes() == b"unchanged-on-rerun"


def test_incomplete_existing_directory_is_an_error(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    (tmp_path / "out" / "toy_v1").mkdir(parents=True)
    with pytest.raises(AcquisitionError, match="not a completed extraction"):
        extract_archive(spec, archive, tmp_path / "out")


def test_pinned_member_mismatch_aborts_extraction(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, pinned_members={"LICENSE": "0" * 64})
    with pytest.raises(AcquisitionError, match="pinned file LICENSE"):
        extract_archive(spec, archive, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_missing_pinned_member_aborts_extraction(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive, pinned_members={"testing_list.txt": "0" * 64})
    with pytest.raises(AcquisitionError, match="missing"):
        extract_archive(spec, archive, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def _symlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


def _hardlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info


@pytest.mark.parametrize(
    ("member", "message"),
    [
        (("../escape.txt", b"x"), "unsafe path"),
        (("/abs/escape.txt", b"x"), "unsafe path"),
        (("yes/../../escape.txt", b"x"), "unsafe path"),
        (("yes\\..\\escape.txt", b"x"), "unsafe path"),
        (_symlink("link", "/etc/passwd"), "unsupported member type"),
        (_hardlink("hard", "LICENSE"), "unsupported member type"),
    ],
)
def test_unsafe_members_are_rejected(
    tmp_path: Path, member: tarfile.TarInfo | tuple[str, bytes], message: str
) -> None:
    bad = make_tar(tmp_path / "bad.tar.gz", [("LICENSE", FILES["LICENSE"]), member])
    spec = spec_for(bad)
    with pytest.raises(AcquisitionError, match=message):
        extract_archive(spec, bad, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []
    assert not (tmp_path / "escape.txt").exists()


def test_case_colliding_members_are_rejected(tmp_path: Path) -> None:
    bad = make_tar(
        tmp_path / "bad.tar.gz",
        [("LICENSE", FILES["LICENSE"]), ("yes/A.wav", b"1"), ("yes/a.wav", b"2")],
    )
    with pytest.raises(AcquisitionError, match="duplicate path"):
        extract_archive(spec_for(bad), bad, tmp_path / "out")


# --- end to end -------------------------------------------------------------


def test_acquire_downloads_then_extracts(archive: Path, tmp_path: Path) -> None:
    spec = spec_for(archive)
    data_dir = tmp_path / "data"
    dest = acquire(spec, data_dir)
    assert (data_dir / "raw" / archive.name).is_file()
    assert dest == data_dir / "raw" / "toy_v1"
    assert (dest / "yes" / "aa_nohash_0.wav").read_bytes() == FILES["yes/aa_nohash_0.wav"]
    assert acquire(spec, data_dir) == dest  # idempotent
